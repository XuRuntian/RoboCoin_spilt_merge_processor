import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd
import cv2

# ================= [新增代码 Start] =================
import re

# 1. 定义标准目标列名 (固定 16 维)
TARGET_NAMES = []
for side in ["left", "right"]:
    for i in range(1, 8):
        TARGET_NAMES.append(f"{side}_arm_joint_{i}_rad")
    TARGET_NAMES.append(f"{side}_gripper_open")

# 2. 辅助函数：提取关节索引
def get_joint_index(name, pattern):
    match = re.search(pattern, name)
    if match: return int(match.group(1))
    return None

# 3. 核心函数：生成标准化映射指令
def get_standardized_mapping(source_names):
    if not source_names: return None, []
    
    # 临时存储找到的索引
    found_indices = {"left": {"arm": {}, "grip": [], "hand": {}}, "right": {"arm": {}, "grip": [], "hand": {}}}

    for idx, name in enumerate(source_names):
        n = name.lower()
        if any(kw in n for kw in ["pos", "quat", "euler", "matrix", "pose", "eef"]): continue
        side = "left" if "left" in n else ("right" if "right" in n else None)
        if not side: continue

        if "arm_joint" in n:
            if (num := get_joint_index(n, r"joint_(\d+)")) is not None: found_indices[side]["arm"][num] = idx
        elif any(kw in n for kw in ["gripper_open", "gripper_width", "gripper_state"]):
            found_indices[side]["grip"].append(idx)
        elif "hand_joint" in n:
            if (num := get_joint_index(n, r"joint_(\d+)")) is not None: found_indices[side]["hand"][num] = idx

    mapping_instructions = []
    for side in ["left", "right"]:
        # 机械臂 1-7 (缺的补0)
        arm_map = found_indices[side]["arm"]
        for i in range(1, 8):
            mapping_instructions.append({"type": "direct", "index": arm_map[i]} if i in arm_map else {"type": "pad_zero"})
        
        # 夹爪 (优先用显式gripper，没有则用灵巧手 joint2+3 合成)
        grip_list, hand_map = found_indices[side]["grip"], found_indices[side]["hand"]
        if grip_list:
            mapping_instructions.append({"type": "direct", "index": grip_list[0]})
        elif 2 in hand_map and 3 in hand_map:
            mapping_instructions.append({"type": "sum", "indices": [hand_map[2], hand_map[3]]})
        elif 2 in hand_map:
            mapping_instructions.append({"type": "direct", "index": hand_map[2]})
        else:
            mapping_instructions.append({"type": "pad_zero"})

    return mapping_instructions, TARGET_NAMES

# 4. 辅助函数：应用映射到向量
def apply_mapping_to_vector(original_vector, mapping_instructions):
    if original_vector is None: return [0.0] * 16
    vec = original_vector.tolist() if isinstance(original_vector, np.ndarray) else (original_vector if isinstance(original_vector, list) else [])
    vec_len = len(vec)
    new_data = []
    for instr in mapping_instructions:
        if instr["type"] == "direct":
            new_data.append(vec[instr["index"]] if instr["index"] < vec_len else 0.0)
        elif instr["type"] == "pad_zero":
            new_data.append(0.0)
        elif instr["type"] == "sum":
            val = sum(vec[idx] for idx in instr["indices"] if idx < vec_len)
            new_data.append(val)
    return new_data

# 5. 辅助函数：应用映射到统计信息
def apply_mapping_to_stats(original_stats, mapping_instructions):
    new_stats = {}
    for key in ["min", "max", "mean", "std"]:
        if key not in original_stats: continue
        old_vals = original_stats[key]
        if not isinstance(old_vals, list): continue
        new_stats[key] = apply_mapping_to_vector(old_vals, mapping_instructions)
    if "count" in original_stats: new_stats["count"] = original_stats["count"]
    return new_stats
# ================= [新增代码 End] =================

# 全局保留字段列表
SYSTEM_RESERVED_FIELDS = [
    "episode_index", "frame_index", "timestamp", "index", "task_index", 
    "subtask_annotation", "scene_annotation", 
]

def get_resample_indices(source_length, src_fps, target_fps):
    """
    计算从 src_fps 重采样到 target_fps 需要保留的源帧索引
    使用最近邻插值以最小化时间漂移
    """
    if target_fps is None or src_fps is None or src_fps == target_fps:
        return None
    
    # 目标总时长保持不变，计算新的帧数
    duration = source_length / src_fps
    target_length = int(np.ceil(duration * target_fps)) # 使用ceil防止丢尾
    
    if target_length == 0:
        return []

    # 生成目标时间戳序列: 0, 0.033, 0.066 ...
    target_timestamps = np.arange(target_length) / target_fps
    
    # 映射回源帧索引: time * src_fps
    source_indices = np.round(target_timestamps * src_fps).astype(int)
    
    # 防止浮点误差导致的越界
    source_indices = np.clip(source_indices, 0, source_length - 1)
    
    return source_indices

def get_safe_indices(names_list):
    """
    根据列名列表，返回需要保留的索引列表。
    逻辑：保留关节(joint/rad)和夹爪(gripper)，剔除末端(end_pos/end_quat/eef)。
    """
    if not names_list:
        return []
    
    indices = []
    kept_names = []
    
    for i, name in enumerate(names_list):
        # 转换为小写比较
        n = name.lower()
        # === 过滤逻辑 ===
        # 如果包含末端位姿关键词，则丢弃
        if "end_pos" in n or "end_quat" in n or "eef_pose" in n or "robot_pos" in n or "robot_quat" in n:
            continue
            
        # 或者：只保留 explicit 的关键词 (根据你的需求调整)
        # if "joint" in n or "rad" in n or "gripper" in n:
        #     indices.append(i)
        #     kept_names.append(name)
        
        # 目前的逻辑：黑名单模式 (剔除明确不需要的，保留其他的)
        indices.append(i)
        kept_names.append(name)
        
    return indices, kept_names

def load_jsonl(file_path):
    """
    从JSONL文件加载数据
    (Load data from a JSONL file)

    Args:
        file_path (str): JSONL文件路径 (Path to the JSONL file)

    Returns:
        list: 包含文件中每行JSON对象的列表 (List containing JSON objects from each line)
    """
    data = []

    # Special handling for episodes_stats.jsonl
    if "episodes_stats.jsonl" in file_path:
        try:
            # Try to load the entire file as a JSON array
            with open(file_path) as f:
                content = f.read()
                # Check if the content starts with '[' and ends with ']'
                if content.strip().startswith("[") and content.strip().endswith("]"):
                    return json.loads(content)
                else:
                    # Try to add brackets and parse
                    try:
                        return json.loads("[" + content + "]")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error loading {file_path} as JSON array: {e}")

        # Fall back to line-by-line parsing
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path} line by line: {e}")
    else:
        # Standard JSONL parsing for other files
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        data.append(json.loads(line))

    return data


def save_jsonl(data, file_path):
    """
    将数据保存为JSONL格式
    (Save data in JSONL format)

    Args:
        data (list): 要保存的JSON对象列表 (List of JSON objects to save)
        file_path (str): 输出文件路径 (Path to the output file)
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ========= 新增：通用工具函数 =========

import subprocess # 记得在文件顶部导入这个

def process_video_pad(src_path, dst_path, target_w, target_h, target_fps=None): 
    """
    使用系统 FFmpeg 命令对视频进行缩放并填充黑边 (Letterbox)。
    这比 OpenCV 更稳健，能解决 AV1 解码问题，且速度更快。
    """
    # 检查 src_path 是否存在
    if not os.path.exists(src_path):
        raise IOError(f"源视频文件不存在: {src_path}")

    # 构建 FFmpeg 滤镜字符串
    # 1. scale: 缩放视频，保持长宽比，使其适应目标框 (force_original_aspect_ratio=decrease)
    # 2. pad: 在缩放后的视频周围填充黑边，使其达到目标分辨率，位置居中
    # 3. setsar: 强制设置像素比为 1:1，防止播放器拉伸
    vf_filter = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1"
    )

    # 构建完整的命令行
    # -y: 覆盖输出文件
    # -loglevel error: 减少日志输出，只显示错误
    # -c:v libx264: 重新编码为 H.264 (兼容性最好)
    # -pix_fmt yuv420p: 确保兼容所有播放器和训练框架
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", src_path,
        "-vf", vf_filter,
    ]
    if target_fps is not None:
        cmd.extend(["-r", str(target_fps)])
        
    cmd.extend([
        "-c:v", "libx264", 
        "-pix_fmt", "yuv420p",
        dst_path
    ])
    try:
        # 执行命令
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        # 如果 ffmpeg 报错，抛出异常，让外层捕获并处理
        error_msg = e.stderr.strip() if e.stderr else "未知错误"
        raise IOError(f"FFmpeg 处理失败: {error_msg}")
    except FileNotFoundError:
        raise IOError("未找到 ffmpeg 命令。请先安装: apt-get install ffmpeg")

    # 简单的后置检查
    if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        raise IOError(f"FFmpeg 未能生成有效的视频文件: {dst_path}")

def get_info(path):
    info_path = os.path.join(path, "meta", "info.json")
    with open(info_path) as f:
        return json.load(f)


def get_chunks_size(info, default=1000):
    return info.get("chunks_size", default)


def detect_folder_dim(folder, fallback_dim):
    try:
        info = get_info(folder)
        state_shape = info.get("features", {}).get("observation.state", {}).get("shape", [fallback_dim])
        action_shape = info.get("features", {}).get("action", {}).get("shape", [fallback_dim])
        detected_dim = max(state_shape[0] if state_shape else fallback_dim,
                           action_shape[0] if action_shape else fallback_dim)
        print(f"检测到文件夹 {folder} 的维度: {detected_dim}")
        return detected_dim
    except Exception as e:
        print(f"检测文件夹 {folder} 维度失败，使用默认值 {fallback_dim}: {e}")
        return fallback_dim


def get_video_keys(info):
    video_keys = []
    for feature_name, feature_info in info.get("features", {}).items():
        if feature_info.get("dtype") == "video":
            video_keys.append(feature_name)
    return video_keys


def pad_episode_stats(stats, from_dim, to_dim):
    # 根据目标维度对单个 episode 的统计数据进行零填充
    if to_dim is None or to_dim <= 0:
        return
    for feature_name in ["observation.state", "action"]:
        feature_stats = stats.get("stats", {}).get(feature_name)
        if not feature_stats:
            continue
        for stat_type in ["min", "max", "mean", "std"]:
            values = feature_stats.get(stat_type)
            if isinstance(values, list):
                cur_dim = len(values)
                if cur_dim < to_dim:
                    feature_stats[stat_type] = values + [0.0] * (to_dim - cur_dim)


def recalc_merged_stats_with_episode_stats(merged_stats, all_stats_data, target_dim):
    if not all_stats_data:
        return
    for feature_name in ["observation.state", "action"]:
        if feature_name not in merged_stats:
            continue
        all_mins, all_maxs, all_means, all_counts = [], [], [], []
        for episode_stats in all_stats_data:
            if feature_name in episode_stats:
                stats = episode_stats[feature_name]
                min_v = stats.get("min")
                max_v = stats.get("max")
                mean_v = stats.get("mean")
                cnt_v = stats.get("count")
                if isinstance(min_v, list) and isinstance(max_v, list) and isinstance(mean_v, list) and cnt_v is not None:
                    # 统一维度
                    if len(min_v) < target_dim:
                        pad_len = target_dim - len(min_v)
                        min_v = min_v + [0.0] * pad_len
                        max_v = max_v + [0.0] * pad_len
                        mean_v = mean_v + [0.0] * pad_len
                    all_mins.append(min_v)
                    all_maxs.append(max_v)
                    all_means.append(mean_v)
                    if isinstance(cnt_v, list):
                        all_counts.append(cnt_v[0])
                    else:
                        all_counts.append(cnt_v)
        if all_mins and all_maxs and all_means and all_counts:
            all_mins_array = np.array(all_mins)
            all_maxs_array = np.array(all_maxs)
            all_means_array = np.array(all_means)
            all_counts_array = np.array(all_counts)
            merged_stats[feature_name]["min"] = all_mins_array.min(axis=0).tolist()
            merged_stats[feature_name]["max"] = all_maxs_array.max(axis=0).tolist()
            total_count = float(all_counts_array.sum())
            if total_count > 0:
                weights = all_counts_array.reshape(-1, 1)
                weighted_means = (all_means_array * weights).sum(axis=0) / total_count
                merged_stats[feature_name]["mean"] = weighted_means.tolist()
            merged_stats[feature_name]["count"] = [int(total_count)]


def select_episodes(
    source_folders,
    max_entries=None,
    max_episodes=None,
    start_entries=None,
    start_episodes=None,
):
    episode_mapping = []
    all_episodes = []
    all_episodes_stats = []
    episode_to_frame_index = {}
    folder_dimensions = {}
    folder_task_mapping = {}
    folder_annotations_mapping = {}
    all_stats_data = []
    task_desc_to_new_index = {}
    all_unique_tasks = []
    subtask_desc_to_new_index = {}
    scene_desc_to_new_index = {}
    gripper_mode_desc_to_new_index = {}
    gripper_activity_desc_to_new_index = {}
    eef_direction_desc_to_new_index = {}
    eef_velocity_desc_to_new_index = {}
    eef_acc_mag_desc_to_new_index = {}
    all_subtasks = []
    all_scenes = []
    all_gripper_modes = []
    all_gripper_activities = []
    all_eef_direction = []
    all_eef_velocity = []
    all_eef_acc_mag = []
    total_frames = 0
    selected_total_episodes = 0
    skipped_frames = 0
    skipped_episodes = 0
    to_skip_episodes = max((start_episodes or 0) - 1, 0)

    first_info = get_info(source_folders[0])
    default_max_dim = int(first_info.get("features", {}).get("observation.state", {}).get("shape", [32])[0])

    # 预检测每个文件夹的维度
    for folder in source_folders:
        folder_dimensions[folder] = detect_folder_dim(folder, default_max_dim)

    for folder in source_folders:
        folder_task_mapping[folder] = {}
        tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
        folder_tasks = load_jsonl(tasks_path) if os.path.exists(tasks_path) else []
        for task in folder_tasks:
            desc = task.get("task")
            old_idx = task.get("task_index")
            if desc not in task_desc_to_new_index:
                new_idx = len(all_unique_tasks)
                task_desc_to_new_index[desc] = new_idx
                all_unique_tasks.append({"task_index": new_idx, "task": desc})
            folder_task_mapping[folder][old_idx] = task_desc_to_new_index[desc]

        folder_annotations_mapping[folder] = {"subtask_annotation": {}, "scene_annotation": {}, "gripper_mode": {}, "gripper_activity": {}, "eef_direction": {}, "eef_velocity": {}, "eef_acc_mag": {}}
        subtask_path = os.path.join(folder, "annotations", "subtask_annotations.jsonl")
        folder_subtasks = load_jsonl(subtask_path) if os.path.exists(subtask_path) else []
        for item in folder_subtasks:
            desc = item.get("subtask")
            old_idx = item.get("subtask_index")
            if desc not in subtask_desc_to_new_index:
                new_idx = len(all_subtasks)
                subtask_desc_to_new_index[desc] = new_idx
                all_subtasks.append({"subtask_index": new_idx, "subtask": desc})
            folder_annotations_mapping[folder]["subtask_annotation"][old_idx] = subtask_desc_to_new_index[desc]
        scene_path = os.path.join(folder, "annotations", "scene_annotations.jsonl")
        folder_scenes = load_jsonl(scene_path) if os.path.exists(scene_path) else []
        for item in folder_scenes:
            desc = item.get("scene")
            old_idx = item.get("scene_index")
            if desc not in scene_desc_to_new_index:
                new_idx = len(all_scenes)
                scene_desc_to_new_index[desc] = new_idx
                all_scenes.append({"scene_index": new_idx, "scene": desc})
            folder_annotations_mapping[folder]["scene_annotation"][old_idx] = scene_desc_to_new_index[desc]
        gm_path = os.path.join(folder, "annotations", "gripper_mode_annotation.jsonl")
        folder_gm = load_jsonl(gm_path) if os.path.exists(gm_path) else []
        for item in folder_gm:
            desc = item.get("gripper_mode")
            old_idx = item.get("gripper_mode_index")
            if desc not in gripper_mode_desc_to_new_index:
                new_idx = len(all_gripper_modes)
                gripper_mode_desc_to_new_index[desc] = new_idx
                all_gripper_modes.append({"gripper_mode_index": new_idx, "gripper_mode": desc})
            folder_annotations_mapping[folder]["gripper_mode"][old_idx] = gripper_mode_desc_to_new_index[desc]
        ga_path = os.path.join(folder, "annotations", "gripper_activity_annotation.jsonl")
        folder_ga = load_jsonl(ga_path) if os.path.exists(ga_path) else []
        for item in folder_ga:
            desc = item.get("gripper_activity")
            old_idx = item.get("gripper_activity_index")
            if desc not in gripper_activity_desc_to_new_index:
                new_idx = len(all_gripper_activities)
                gripper_activity_desc_to_new_index[desc] = new_idx
                all_gripper_activities.append({"gripper_activity_index": new_idx, "gripper_activity": desc})
            folder_annotations_mapping[folder]["gripper_activity"][old_idx] = gripper_activity_desc_to_new_index[desc]
        ed_path = os.path.join(folder, "annotations", "eef_direction_annotation.jsonl")
        folder_ed = load_jsonl(ed_path) if os.path.exists(ed_path) else []
        for item in folder_ed:
            desc = item.get("eef_direction")
            old_idx = item.get("eef_direction_index")
            if desc not in eef_direction_desc_to_new_index:
                new_idx = len(all_eef_direction)
                eef_direction_desc_to_new_index[desc] = new_idx
                all_eef_direction.append({"eef_direction_index": new_idx, "eef_direction": desc})
            folder_annotations_mapping[folder]["eef_direction"][old_idx] = eef_direction_desc_to_new_index[desc]
        ev_path = os.path.join(folder, "annotations", "eef_velocity_annotation.jsonl")
        folder_ev = load_jsonl(ev_path) if os.path.exists(ev_path) else []
        for item in folder_ev:
            desc = item.get("eef_velocity")
            old_idx = item.get("eef_velocity_index")
            if desc not in eef_velocity_desc_to_new_index:
                new_idx = len(all_eef_velocity)
                eef_velocity_desc_to_new_index[desc] = new_idx
                all_eef_velocity.append({"eef_velocity_index": new_idx, "eef_velocity": desc})
            folder_annotations_mapping[folder]["eef_velocity"][old_idx] = eef_velocity_desc_to_new_index[desc]
        ea_path = os.path.join(folder, "annotations", "eef_acc_mag_annotation.jsonl")
        folder_ea = load_jsonl(ea_path) if os.path.exists(ea_path) else []
        for item in folder_ea:
            desc = item.get("eef_acc_mag")
            old_idx = item.get("eef_acc_mag_index")
            if desc not in eef_acc_mag_desc_to_new_index:
                new_idx = len(all_eef_acc_mag)
                eef_acc_mag_desc_to_new_index[desc] = new_idx
                all_eef_acc_mag.append({"eef_acc_mag_index": new_idx, "eef_acc_mag": desc})
            folder_annotations_mapping[folder]["eef_acc_mag"][old_idx] = eef_acc_mag_desc_to_new_index[desc]

        episodes = load_jsonl(os.path.join(folder, "meta", "episodes.jsonl"))
        stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
        episodes_stats = load_jsonl(stats_path) if os.path.exists(stats_path) else []
        stats_map = {s.get("episode_index"): s for s in episodes_stats if "episode_index" in s}

        for ep in episodes:
            # 跳过起始帧或起始episode
            if start_entries is not None and skipped_frames < start_entries:
                skipped_frames += ep.get("length", 0)
                skipped_episodes += 1
                continue
            if start_entries is None and start_episodes is not None and skipped_episodes < to_skip_episodes:
                skipped_frames += ep.get("length", 0)
                skipped_episodes += 1
                continue

            # 限制条件
            if max_entries is not None and total_frames >= max_entries:
                break
            if max_episodes is not None and selected_total_episodes >= max_episodes:
                break

            old_index = ep["episode_index"]
            new_index = selected_total_episodes
            ep["episode_index"] = new_index
            all_episodes.append(ep)

            if old_index in stats_map:
                stats = stats_map[old_index]
                stats["episode_index"] = new_index

                if "stats" in stats:
                    if "task_index" in stats["stats"]:
                        original_task_index = stats["stats"]["task_index"]
                        if isinstance(original_task_index, dict) and "min" in original_task_index:
                            old_task_idx = int(original_task_index["min"][0])
                        else:
                            old_task_idx = int(original_task_index)
                        if folder in folder_task_mapping and old_task_idx in folder_task_mapping[folder]:
                            new_task_idx = folder_task_mapping[folder][old_task_idx]
                            if isinstance(stats["stats"]["task_index"], dict):
                                stats["stats"]["task_index"]["min"] = [new_task_idx]
                                stats["stats"]["task_index"]["max"] = [new_task_idx]
                                stats["stats"]["task_index"]["mean"] = [float(new_task_idx)]
                            else:
                                stats["stats"]["task_index"] = new_task_idx

                    ann_map = folder_annotations_mapping.get(folder) if 'folder_annotations_mapping' in locals() else None
                    if ann_map:
                        def _map_vals(vals, key):
                            if isinstance(vals, list):
                                return [ann_map[key].get(int(v), int(v)) for v in vals]
                            return vals
                        for feat, key in [("subtask_annotation", "subtask_annotation"), ("scene_annotation", "scene_annotation"), ("gripper_mode_state", "gripper_mode"), ("gripper_mode_action", "gripper_mode"), ("gripper_activity_state", "gripper_activity"), ("gripper_activity_action", "gripper_activity"), ("eef_direction_state", "eef_direction"), ("eef_direction_action", "eef_direction"), ("eef_velocity_state", "eef_velocity"), ("eef_velocity_action", "eef_velocity"), ("eef_acc_mag_state", "eef_acc_mag"), ("eef_acc_mag_action", "eef_acc_mag")]:
                            if feat in stats["stats"] and isinstance(stats["stats"][feat], dict):
                                for k in ["min", "max", "mean"]:
                                    if k in stats["stats"][feat]:
                                        stats["stats"][feat][k] = _map_vals(stats["stats"][feat][k], key)

                all_episodes_stats.append(stats)
                if "stats" in stats:
                    all_stats_data.append(stats["stats"])

            episode_to_frame_index[new_index] = total_frames
            episode_length = ep.get("length", 0)
            total_frames += episode_length
            selected_total_episodes += 1
            episode_mapping.append((folder, old_index, new_index))

    all_annotations = {
        "subtask_annotations": all_subtasks,
        "scene_annotations": all_scenes,
        "gripper_mode_annotation": all_gripper_modes,
        "gripper_activity_annotation": all_gripper_activities,
        "eef_direction_annotation": all_eef_direction,
        "eef_velocity_annotation": all_eef_velocity,
        "eef_acc_mag_annotation": all_eef_acc_mag,
    }
    return (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        folder_annotations_mapping,
        all_unique_tasks,
        all_annotations,
        all_stats_data,
        total_frames,
    )


def write_meta_and_copy(
    source_folders,
    output_folder,
    episode_mapping,
    all_episodes,
    all_episodes_stats,
    folder_dimensions,
    folder_task_mapping,
    folder_annotations_mapping,
    episode_to_frame_index, # 注意：这是基于原始长度的旧映射，下面会重新计算
    all_stats_data,
    all_tasks,
    all_annotations,
    total_frames,
    max_dim_cli,
    fps,
    features_to_keep=None,
    video_size=None,
    remove_eef=False
):

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    episode_source_map = {new_idx: folder for folder, _, new_idx in episode_mapping}
    folder_indices_map = {} 
    global_max_dims_map = {} 
    max_dim_source_folder = {} 

    base_info = get_info(source_folders[0])
    
    for folder in source_folders:
        current_info = get_info(folder)
        folder_mapping_instructions[folder] = {} # 初始化
        
        if remove_eef:
            # [修改] 使用新的 get_standardized_mapping
            for feat in ["observation.state", "action"]:
                if feat in current_info.get("features", {}):
                    names = current_info["features"][feat].get("names", [])
                    mapping_instr, _ = get_standardized_mapping(names)
                    if mapping_instr:
                        folder_mapping_instructions[folder][feat] = mapping_instr

    # [修改] 强制最大维度为 16
    actual_max_dim = 16 
    if remove_eef:
        print(f"!!! 强制标准化维度: 16 (Left:7+1, Right:7+1) !!!")
        # 更新 info.json 的 names
        for feat in ["observation.state", "action"]:
            if feat in base_info.get("features", {}):
                base_info["features"][feat]["shape"] = [actual_max_dim]
                base_info["features"][feat]["names"] = TARGET_NAMES
    else:
        actual_max_dim = max_dim_cli or max(folder_dimensions.values())


    if remove_eef:
        for feat in ["observation.state", "action"]:
            if feat in global_max_dims_map:
                if feat not in base_info["features"]:
                    continue 

                base_info["features"][feat]["shape"] = [actual_max_dim]
                
                src_folder = max_dim_source_folder.get(feat)
                if src_folder:
                    src_info = get_info(src_folder)
                    src_names = src_info["features"][feat].get("names", [])
                    if src_names:
                        _, kept_names = get_safe_indices(src_names)
                        if len(kept_names) < actual_max_dim:
                            kept_names += [f"pad_{i}" for i in range(len(kept_names), actual_max_dim)]
                        base_info["features"][feat]["names"] = kept_names

    RESERVED_FEATURES = SYSTEM_RESERVED_FIELDS
    if features_to_keep:
        print(f"筛选特定字段: {features_to_keep}")
        original_features = base_info.get("features", {})
        filtered_features = {}
        all_wanted = set(features_to_keep) | set(RESERVED_FEATURES)
        for feat in all_wanted:
            if feat in original_features:
                feat_def = original_features[feat].copy()
                if feat_def.get("dtype") == "video" and video_size is not None:
                    tw, th = video_size
                    feat_def["shape"] = [th, tw, 3]
                    if "info" in feat_def:
                        feat_def["info"] = feat_def["info"].copy()
                        feat_def["info"]["video.width"] = tw
                        feat_def["info"]["video.height"] = th
                filtered_features[feat] = feat_def
        base_info["features"] = filtered_features
        video_keys = [k for k in get_video_keys({"features": filtered_features})]
    else:
        video_keys = get_video_keys(base_info)
    
    chunks_size = get_chunks_size(base_info)
    total_episodes = len(all_episodes)
    total_videos = len(video_keys) * total_episodes

    aligned_episode_stats = []
    features_set = set(features_to_keep) if features_to_keep else None

    for stats in all_episodes_stats:
        if "stats" in stats:
            current_stats = stats["stats"]
            ep_idx = stats.get("episode_index")
            source_folder = episode_source_map.get(ep_idx)
            current_folder_indices = folder_indices_map.get(source_folder, {})
            current_mappings = folder_mapping_instructions.get(source_folder, {})
            new_stats_content = {}
            for k, v in current_stats.items():
                keep = False
                if features_set is None:
                    keep = True
                elif k in features_set or k in RESERVED_FEATURES:
                    keep = True
                else:
                    pass 
                
                if keep:
                    # [修改] 如果是 state/action 且有映射指令，应用新函数
                    if remove_eef and k in ["observation.state", "action"] and k in current_mappings:
                        new_stats_content[k] = apply_mapping_to_stats(v, current_mappings[k])
                    else:
                        # 旧的切片逻辑或原样保留
                        new_stats_content[k] = v
            stats["stats"] = new_stats_content

        pad_episode_stats(stats, from_dim=actual_max_dim, to_dim=actual_max_dim)
        aligned_episode_stats.append(stats)

    # === [FIX] 核心修复点 1: 重建 episode_to_frame_index 映射 ===
    # 因为重采样后，episode长度变了，旧的映射表（基于源数据长度）已经失效
    # 必须基于新的长度重新计算，否则 data parquet 的 index 列会错乱
    total_frames_resampled = 0
    folder_fps_cache = {} 
    
    # 新的映射表
    new_episode_to_frame_index = {}

    for ep in all_episodes:
        src_folder = episode_source_map[ep["episode_index"]]
        if src_folder not in folder_fps_cache:
            # [FIX] 统一使用 30 作为默认值，而不是 fallback 到 target fps，确保缺少 FPS 时能触发重采样
            folder_fps_cache[src_folder] = get_info(src_folder).get("fps", 30)
        
        src_fps = folder_fps_cache[src_folder]
        
        # 记录重采样后的起始帧
        new_episode_to_frame_index[ep["episode_index"]] = total_frames_resampled

        if fps is not None and src_fps != fps:
            original_len = ep["length"]
            duration = original_len / src_fps
            new_len = int(np.ceil(duration * fps))
            ep["length"] = new_len
        
        total_frames_resampled += ep["length"]

    # [FIX] 覆盖旧的映射表，传给 copy_data_files
    episode_to_frame_index = new_episode_to_frame_index

    # 设置总帧数为重采样后的值
    base_info["total_frames"] = total_frames_resampled

    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(aligned_episode_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))

    used_task_indices = set()
    for episode in all_episodes:
        if "task_index" in episode:
            used_task_indices.add(episode["task_index"])
    if not used_task_indices:
        for stats in aligned_episode_stats:
            if "stats" in stats and "task_index" in stats["stats"]:
                task_idx_info = stats["stats"]["task_index"]
                if isinstance(task_idx_info, dict) and "min" in task_idx_info:
                    used_task_indices.add(int(task_idx_info["min"][0]))
                else:
                    used_task_indices.add(int(task_idx_info))
    filtered_tasks = [t for t in all_tasks if t["task_index"] in used_task_indices]
    save_jsonl(filtered_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)
    if all_annotations:
        for key, val in all_annotations.items():
            if val:
                save_jsonl(val, os.path.join(output_folder, "annotations", f"{key}.jsonl"))

    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats_list.append(json.load(f))
    if stats_list:
        merged_stats = merge_stats(stats_list)
        recalc_merged_stats_with_episode_stats(merged_stats, all_stats_data, target_dim=actual_max_dim)
        if features_to_keep:
            merged_stats = {k: v for k, v in merged_stats.items() if k in features_set or k in RESERVED_FEATURES}
        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    base_info["total_episodes"] = total_episodes
    # [FIX] 核心修复点 2: 删除了这里覆盖 total_frames 的代码
    # base_info["total_frames"] = total_frames <-- 删除这行，保留上面的 total_frames_resampled
    
    base_info["total_tasks"] = len(filtered_tasks)
    base_info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
    base_info["splits"] = {"train": f"0:{total_episodes}"}
    base_info["fps"] = fps
    base_info["total_videos"] = total_videos
    for feature_name in ["observation.state", "action"]:
        if feature_name in base_info.get("features", {}) and "shape" in base_info["features"][feature_name]:
            base_info["features"][feature_name]["shape"] = [actual_max_dim]

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(base_info, f, indent=4)

    if video_keys:
        # [FIX] 传递 fps 参数给 copy_videos
        copy_videos(source_folders, output_folder, episode_mapping, video_size=video_size, filtered_features=base_info["features"], fps=fps)
    
    copy_data_files(
        source_folders=source_folders,
        output_folder=output_folder,
        episode_mapping=episode_mapping,
        max_dim=actual_max_dim, 
        fps=fps,
        episode_to_frame_index=episode_to_frame_index, # [FIX] 传入更新后的映射表
        folder_task_mapping=folder_task_mapping,
        folder_annotations_mapping=folder_annotations_mapping,
        chunks_size=chunks_size,
        features_to_keep=features_to_keep,
        remove_eef_flag=remove_eef,
        features_def=base_info["features"],
    )
    print(f"Done: {total_episodes} episodes, {total_frames_resampled} frames (resampled), output={output_folder}")

# ========= 原有函数 =========

def merge_stats(stats_list):
    """
    合并多个数据集的统计信息，确保维度一致性
    (Merge statistics from multiple datasets, ensuring dimensional consistency)

    Args:
        stats_list (list): 包含每个数据集统计信息的字典列表
                          (List of dictionaries containing statistics for each dataset)

    Returns:
        dict: 合并后的统计信息 (Merged statistics)
    """
    # Initialize merged stats with the structure of the first stats
    merged_stats = {}

    # Find common features across all stats
    common_features = set(stats_list[0].keys())
    for stats in stats_list[1:]:
        common_features = common_features.intersection(set(stats.keys()))

    # Process features in the order they appear in the first stats file
    for feature in stats_list[0]:
        if feature not in common_features:
            continue

        merged_stats[feature] = {}

        # Find common stat types for this feature
        common_stat_types = []
        for stat_type in ["mean", "std", "max", "min"]:
            if all(stat_type in stats[feature] for stats in stats_list):
                common_stat_types.append(stat_type)

        # Determine the original shape of each value
        original_shapes = []
        for stats in stats_list:
            if "mean" in stats[feature]:
                shape = np.array(stats[feature]["mean"]).shape
                original_shapes.append(shape)

        # Special handling for image features to preserve nested structure
        if feature.startswith("observation.images."):
            for stat_type in common_stat_types:
                try:
                    # Get all values
                    values = [stats[feature][stat_type] for stats in stats_list]

                    # For image features, we need to preserve the nested structure
                    # Initialize with the first value's structure
                    result = []

                    # For RGB channels
                    for channel_idx in range(len(values[0])):
                        channel_result = []

                        # For each pixel row
                        for pixel_idx in range(len(values[0][channel_idx])):
                            pixel_result = []

                            # For each pixel value
                            for value_idx in range(len(values[0][channel_idx][pixel_idx])):
                                # Calculate statistic based on type
                                if stat_type == "mean":
                                    # Simple average
                                    avg = sum(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    ) / len(values)
                                    pixel_result.append(avg)
                                elif stat_type == "std":
                                    # Simple average of std
                                    avg = sum(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    ) / len(values)
                                    pixel_result.append(avg)
                                elif stat_type == "max":
                                    # Maximum
                                    max_val = max(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    )
                                    pixel_result.append(max_val)
                                elif stat_type == "min":
                                    # Minimum
                                    min_val = min(
                                        values[i][channel_idx][pixel_idx][value_idx]
                                        for i in range(len(values))
                                    )
                                    pixel_result.append(min_val)

                            channel_result.append(pixel_result)

                        result.append(channel_result)

                    merged_stats[feature][stat_type] = result
                except Exception as e:
                    print(f"Warning: Error processing image feature {feature}.{stat_type}: {e}")
                    # Fallback to first value
                    merged_stats[feature][stat_type] = values[0]
        # If all shapes are the same, no need for special handling
        elif len({str(shape) for shape in original_shapes}) == 1:
            # All shapes are the same, use standard merging
            for stat_type in common_stat_types:
                values = [stats[feature][stat_type] for stats in stats_list]

                try:
                    # Calculate the new statistic based on the type
                    if stat_type == "mean":
                        if all("count" in stats[feature] for stats in stats_list):
                            counts = [stats[feature]["count"][0] for stats in stats_list]
                            total_count = sum(counts)
                            weighted_values = [
                                np.array(val) * count / total_count
                                for val, count in zip(values, counts, strict=False)
                            ]
                            merged_stats[feature][stat_type] = np.sum(weighted_values, axis=0).tolist()
                        else:
                            merged_stats[feature][stat_type] = np.mean(np.array(values), axis=0).tolist()

                    elif stat_type == "std":
                        if all("count" in stats[feature] for stats in stats_list):
                            counts = [stats[feature]["count"][0] for stats in stats_list]
                            total_count = sum(counts)
                            variances = [np.array(std) ** 2 for std in values]
                            weighted_variances = [
                                var * count / total_count
                                for var, count in zip(variances, counts, strict=False)
                            ]
                            merged_stats[feature][stat_type] = np.sqrt(
                                np.sum(weighted_variances, axis=0)
                            ).tolist()
                        else:
                            merged_stats[feature][stat_type] = np.mean(np.array(values), axis=0).tolist()

                    elif stat_type == "max":
                        merged_stats[feature][stat_type] = np.maximum.reduce(np.array(values)).tolist()

                    elif stat_type == "min":
                        merged_stats[feature][stat_type] = np.minimum.reduce(np.array(values)).tolist()
                except Exception as e:
                    print(f"Warning: Error processing {feature}.{stat_type}: {e}")
                    continue
        else:
            # Shapes are different, need special handling for state vectors
            if feature in ["observation.state", "action"]:
                # For state vectors, we need to handle different dimensions
                max_dim = max(len(np.array(stats[feature]["mean"]).flatten()) for stats in stats_list)

                for stat_type in common_stat_types:
                    try:
                        # Get values and their original dimensions
                        values_with_dims = []
                        for stats in stats_list:
                            val = np.array(stats[feature][stat_type]).flatten()
                            dim = len(val)
                            values_with_dims.append((val, dim))

                        # Initialize result array with zeros
                        result = np.zeros(max_dim)

                        # Calculate statistics for each dimension separately
                        if stat_type == "mean":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [stats[feature]["count"][0] for stats in stats_list]
                                total_count = sum(counts)

                                # For each dimension, calculate weighted mean of available values
                                for d in range(max_dim):
                                    dim_values = []
                                    dim_weights = []
                                    for (val, dim), count in zip(values_with_dims, counts, strict=False):
                                        if d < dim:  # Only use values that have this dimension
                                            dim_values.append(val[d])
                                            dim_weights.append(count)

                                    if dim_values:  # If we have values for this dimension
                                        weighted_sum = sum(
                                            v * w for v, w in zip(dim_values, dim_weights, strict=False)
                                        )
                                        result[d] = weighted_sum / sum(dim_weights)
                            else:
                                # Simple average for each dimension
                                for d in range(max_dim):
                                    dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                    if dim_values:
                                        result[d] = sum(dim_values) / len(dim_values)

                        elif stat_type == "std":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [stats[feature]["count"][0] for stats in stats_list]
                                total_count = sum(counts)

                                # For each dimension, calculate weighted variance
                                for d in range(max_dim):
                                    dim_variances = []
                                    dim_weights = []
                                    for (val, dim), count in zip(values_with_dims, counts, strict=False):
                                        if d < dim:  # Only use values that have this dimension
                                            dim_variances.append(val[d] ** 2)  # Square for variance
                                            dim_weights.append(count)

                                    if dim_variances:  # If we have values for this dimension
                                        weighted_var = sum(
                                            v * w for v, w in zip(dim_variances, dim_weights, strict=False)
                                        ) / sum(dim_weights)
                                        result[d] = np.sqrt(weighted_var)  # Take sqrt for std
                            else:
                                # Simple average of std for each dimension
                                for d in range(max_dim):
                                    dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                    if dim_values:
                                        result[d] = sum(dim_values) / len(dim_values)

                        elif stat_type == "max":
                            # For each dimension, take the maximum of available values
                            for d in range(max_dim):
                                dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                if dim_values:
                                    result[d] = max(dim_values)

                        elif stat_type == "min":
                            # For each dimension, take the minimum of available values
                            for d in range(max_dim):
                                dim_values = [val[d] for val, dim in values_with_dims if d < dim]
                                if dim_values:
                                    result[d] = min(dim_values)

                        # Convert result to list and store
                        merged_stats[feature][stat_type] = result.tolist()

                    except Exception as e:
                        print(
                            f"Warning: Error processing {feature}.{stat_type} with different dimensions: {e}"
                        )
                        continue
            else:
                # For other features with different shapes, use the first shape as template
                template_shape = original_shapes[0]
                print(f"Using shape {template_shape} as template for {feature}")

                for stat_type in common_stat_types:
                    try:
                        # Use the first stats as template
                        merged_stats[feature][stat_type] = stats_list[0][feature][stat_type]
                    except Exception as e:
                        print(
                            f"Warning: Error processing {feature}.{stat_type} with shape {template_shape}: {e}"
                        )
                        continue

        # Add count if available in all stats
        if all("count" in stats[feature] for stats in stats_list):
            try:
                merged_stats[feature]["count"] = [sum(stats[feature]["count"][0] for stats in stats_list)]
            except Exception as e:
                print(f"Warning: Error processing {feature}.count: {e}")

    return merged_stats


def copy_videos(source_folders, output_folder, episode_mapping, video_size=None, filtered_features=None, fps=None):
    """
    从源文件夹复制视频文件到输出文件夹，保持正确的索引和结构
    [FIX] 增加了 fps 参数，用于在分辨率匹配时检查帧率是否需要转换
    """
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]
    
    features_to_scan = filtered_features if filtered_features is not None else info.get("features", {})
    video_keys = []
    for k, v in features_to_scan.items():
        if v.get("dtype") == "video":
            video_keys.append(k)
    
    print(f"即将复制的视频流: {video_keys}")

    # [FIX] 缓存文件夹 FPS，避免重复IO
    folder_fps_cache = {}

    for old_folder, old_index, new_index in episode_mapping:
        episode_chunk = old_index // info["chunks_size"]
        new_episode_chunk = new_index // info["chunks_size"]

        # [FIX] 获取源 FPS
        if old_folder not in folder_fps_cache:
            try:
                with open(os.path.join(old_folder, "meta", "info.json")) as f:
                    folder_fps_cache[old_folder] = json.load(f).get("fps", 30)
            except:
                folder_fps_cache[old_folder] = 30
        src_fps = folder_fps_cache[old_folder]

        for video_key in video_keys:
            source_patterns = [
                os.path.join(old_folder, video_path_template.format(episode_chunk=episode_chunk, video_key=video_key, episode_index=old_index)),
                os.path.join(old_folder, video_path_template.format(episode_chunk=0, video_key=video_key, episode_index=0)),
                os.path.join(old_folder, f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4"),
                os.path.join(old_folder, f"videos/chunk-000/{video_key}/episode_000000.mp4"),
            ]

            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break

            if source_video_path:
                dest_video_path = os.path.join(output_folder, video_path_template.format(episode_chunk=new_episode_chunk, video_key=video_key, episode_index=new_index))
                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                if video_size is not None:
                    target_w, target_h = video_size
                    need_convert = True
                    try:
                        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", source_video_path]
                        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
                        if output:
                            parts = output.split('x')
                            if len(parts) == 2:
                                w, h = int(parts[0]), int(parts[1])
                                # [FIX] 核心修复：只有当分辨率一致 且 FPS 不需要转换时，才跳过
                                fps_match = (fps is None) or (abs(src_fps - fps) < 1e-5)
                                if w == target_w and h == target_h and fps_match:
                                    need_convert = False
                                    print(f"Skipping convert (size match {w}x{h}, fps match): {source_video_path} -> Copying...")
                                elif w == target_w and h == target_h and not fps_match:
                                    print(f"Force convert (FPS change {src_fps}->{fps}): {source_video_path}")
                                    
                    except Exception as e:
                        print(f"Check video size failed: {e}, will force convert.")
                        need_convert = True
                    
                    if need_convert:
                        try:
                            # [FIX] 确保传入 target_fps
                            process_video_pad(source_video_path, dest_video_path, target_w, target_h, target_fps=fps)
                        except Exception as e:
                            print(f"Error processing video {source_video_path}: {e}")
                            shutil.copy2(source_video_path, dest_video_path)
                    else:
                        shutil.copy2(source_video_path, dest_video_path)
                else:
                    shutil.copy2(source_video_path, dest_video_path)
            else:
                print(f"Warning: Video file not found for {video_key}, episode {old_index} in {old_folder}")

def validate_timestamps(source_folders, tolerance_s=1e-4):
    """
    验证源数据集的时间戳结构，识别潜在问题
    (Validate timestamp structure of source datasets, identify potential issues)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        tolerance_s (float): 时间戳不连续性的容差值，以秒为单位 (Tolerance for timestamp discontinuities in seconds)

    Returns:
        tuple: (issues, fps_values) - 问题列表和检测到的FPS值列表
               (List of issues and list of detected FPS values)
    """
    issues = []
    fps_values = []

    for folder in source_folders:
        try:
            # 尝试从 info.json 获取 FPS (Try to get FPS from info.json)
            info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                    if "fps" in info:
                        fps = info["fps"]
                        fps_values.append(fps)
                        print(f"数据集 {folder} FPS={fps} (Dataset {folder} FPS={fps})")

            # 检查是否有parquet文件包含时间戳 (Check if any parquet files contain timestamps)
            parquet_path = None
            for root, _, files in os.walk(os.path.join(folder, "parquet")):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_path = os.path.join(root, file)
                        break
                if parquet_path:
                    break

            if not parquet_path:
                for root, _, files in os.walk(os.path.join(folder, "data")):
                    for file in files:
                        if file.endswith(".parquet"):
                            parquet_path = os.path.join(root, file)
                            break
                    if parquet_path:
                        break

            if parquet_path:
                df = pd.read_parquet(parquet_path)
                timestamp_cols = [col for col in df.columns if "timestamp" in col or "time" in col]
                if timestamp_cols:
                    print(
                        f"数据集 {folder} 包含时间戳列: {timestamp_cols} (Dataset {folder} contains timestamp columns: {timestamp_cols})"
                    )
                else:
                    issues.append(
                        f"警告: 数据集 {folder} 没有时间戳列 (Warning: Dataset {folder} has no timestamp columns)"
                    )
            else:
                issues.append(
                    f"警告: 数据集 {folder} 未找到parquet文件 (Warning: No parquet files found in dataset {folder})"
                )

        except Exception as e:
            issues.append(
                f"错误: 验证数据集 {folder} 失败: {e} (Error: Failed to validate dataset {folder}: {e})"
            )
            print(f"验证错误: {e} (Validation error: {e})")
            traceback.print_exc()

    # 检查FPS是否一致 (Check if FPS values are consistent)
    if len(set(fps_values)) > 1:
        issues.append(
            f"警告: 数据集FPS不一致: {fps_values} (Warning: Inconsistent FPS across datasets: {fps_values})"
        )

    return issues, fps_values


import pyarrow as pa
import pyarrow.parquet as pq

def copy_data_files(
    source_folders,
    output_folder,
    episode_mapping,
    max_dim=18,
    fps=None,
    episode_to_frame_index=None,
    folder_task_mapping=None,
    folder_annotations_mapping=None,
    chunks_size=1000,
    default_fps=30,
    features_to_keep=None,
    remove_eef_flag=False,
    features_def=None,
    folder_mapping_instructions=None
):
    """
    复制并处理parquet数据文件，使用 Pyarrow 强制执行严格的 Schema 类型转换
    [FIX] 增加了强制重写 timestamp 的逻辑，解决训练时的 diff 报错
    """
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get("fps", default_fps)
        else:
            fps = default_fps

    print(f"使用FPS={fps}")
    total_copied = 0
    total_failed = 0
    failed_files = []
    
    RESERVED_COLUMNS = SYSTEM_RESERVED_FIELDS

    def _build_pyarrow_schema(df_columns, features_def):
        fields = []
        for col in df_columns:
            pa_type = pa.string()
            if col in ["frame_index", "episode_index", "index", "task_index"]:
                pa_type = pa.int64()
            elif col == "timestamp":
                pa_type = pa.float32()
            elif features_def and col in features_def:
                feat = features_def[col]
                dtype = feat.get("dtype")
                shape = feat.get("shape")
                if dtype == "float32":
                    if shape and (len(shape) > 0 and shape != [1]):
                        pa_type = pa.list_(pa.float32())
                    else:
                        pa_type = pa.float32()
                elif dtype == "int32":
                    if shape and (len(shape) > 0 and shape != [1]):
                        pa_type = pa.list_(pa.int32())
                    else:
                        pa_type = pa.int32()
                elif dtype == "video" or dtype == "string":
                    pa_type = pa.string()
            else:
                if "annotation" in col or "gripper" in col or "eef" in col:
                     pa_type = pa.list_(pa.int32())
            fields.append((col, pa_type))
        return pa.schema(fields)

    def _filter_columns(dataframe):
        if features_to_keep:
            cols_to_retain = set(features_to_keep)
            final_columns = []
            for col in dataframe.columns:
                if (col in cols_to_retain or 
                    col in RESERVED_COLUMNS or 
                    any(col.startswith(f"{k}.") for k in cols_to_retain)):
                    final_columns.append(col)
            if final_columns:
                return dataframe[final_columns]
        return dataframe

    folder_meta_cache = {}

    for i, (old_folder, old_index, new_index) in enumerate(episode_mapping):
        episode_str = f"episode_{old_index:06d}.parquet"
        source_paths = [
            os.path.join(old_folder, "parquet", episode_str),
            os.path.join(old_folder, "data", episode_str),
        ]
        source_path = None
        for path in source_paths:
            if os.path.exists(path):
                source_path = path
                break
        
        if not source_path:
             for root, _, files in os.walk(old_folder):
                for file in files:
                    if file.endswith(".parquet") and f"episode_{old_index:06d}" in file:
                         source_path = os.path.join(root, file)
                         break
                if source_path: break

        if source_path:
            try:
                df = pd.read_parquet(source_path)
                src_info = get_info(old_folder)
                src_fps = src_info.get("fps", default_fps)
                
                # 1. 重采样
                if fps is not None and src_fps != fps:
                    indices = get_resample_indices(len(df), src_fps, fps)
                    if indices is not None and len(indices) > 0:
                        df = df.iloc[indices]
                        df.reset_index(drop=True, inplace=True)
                
                # [FIX] 2. 强制重写时间戳
                # 消除源数据的采集抖动和重采样带来的浮点误差，确保 timestamp 严格对齐
                if "timestamp" in df.columns and fps is not None and fps > 0:
                    df["timestamp"] = (np.arange(len(df), dtype=np.float64) / fps).astype(np.float32)

                df = _filter_columns(df)
                
                # 3. 移# ================= [新增代码 Start] =================
                # 应用标准化映射 (如果有指令)
                mapping_applied = False # [新增标记]
                if folder_mapping_instructions:
                    mappings = folder_mapping_instructions.get(old_folder, {})
                    for feat in ["observation.state", "action"]:
                        if feat in df.columns and feat in mappings:
                            instr = mappings[feat]
                            # 逐行应用转换
                            df[feat] = df[feat].apply(lambda x: apply_mapping_to_vector(x, instr))
                            mapping_applied = True
                # ================= [新增代码 End] =================除 EEF
                if remove_eef_flag:
                    if old_folder not in folder_meta_cache:
                        src_info = get_info(old_folder)
                        folder_meta_cache[old_folder] = src_info["features"]
                    src_features = folder_meta_cache[old_folder]
                    for feat in ["observation.state", "action"]:
                        if feat in df.columns and feat in src_features:
                            original_names = src_features[feat].get("names", [])
                            if original_names:
                                keep_idxs, _ = get_safe_indices(original_names)
                                df[feat] = df[feat].apply(
                                    lambda x: np.array(x)[keep_idxs].tolist() 
                                    if isinstance(x, (list, np.ndarray)) and len(x) >= len(original_names) else x
                                )

                # 4. 维度 Padding
                for feature in ["observation.state", "action"]:
                    if feature in df.columns:
                        first_val = df[feature].dropna().iloc[0] if not df[feature].dropna().empty else None
                        if isinstance(first_val, (list, np.ndarray)):
                             df[feature] = df[feature].apply(
                                lambda x: np.pad(x, (0, max_dim - len(x)), "constant").tolist()
                                if x is not None and isinstance(x, (list, np.ndarray)) and len(x) < max_dim else x
                            )

                if "episode_index" in df.columns:
                    df["episode_index"] = new_index

                if "index" in df.columns:
                    if episode_to_frame_index and new_index in episode_to_frame_index:
                        first_index = episode_to_frame_index[new_index]
                    else:
                        first_index = new_index * len(df)
                    df["index"] = [first_index + i for i in range(len(df))]

                if "task_index" in df.columns and folder_task_mapping and old_folder in folder_task_mapping:
                    current_task_index = df["task_index"].iloc[0]
                    if current_task_index in folder_task_mapping[old_folder]:
                        df["task_index"] = folder_task_mapping[old_folder][current_task_index]

                if folder_annotations_mapping and old_folder in folder_annotations_mapping:
                    ann_map = folder_annotations_mapping[old_folder]
                    
                    def apply_map(col_name, mapping_key):
                        if col_name in df.columns and mapping_key in ann_map:
                            first_val = df[col_name].dropna().iloc[0] if not df[col_name].dropna().empty else None
                            is_list = isinstance(first_val, (list, np.ndarray))
                            
                            if is_list:
                                df[col_name] = df[col_name].apply(
                                    lambda x: [ann_map[mapping_key].get(int(v), int(v)) for v in x] if isinstance(x, list) else x
                                )
                            else:
                                df[col_name] = df[col_name].apply(
                                    lambda x: ann_map[mapping_key].get(int(x), int(x)) if pd.notnull(x) else x
                                )

                    mapping_configs = [
                        ("subtask_annotation", "subtask_annotation"),
                        ("scene_annotation", "scene_annotation"),
                        ("gripper_mode_state", "gripper_mode"), ("gripper_mode_action", "gripper_mode"),
                        ("gripper_activity_state", "gripper_activity"), ("gripper_activity_action", "gripper_activity"),
                        ("eef_direction_state", "eef_direction"), ("eef_direction_action", "eef_direction"),
                        ("eef_velocity_state", "eef_velocity"), ("eef_velocity_action", "eef_velocity"),
                        ("eef_acc_mag_state", "eef_acc_mag"), ("eef_acc_mag_action", "eef_acc_mag")
                    ]
                    for col, key in mapping_configs:
                        apply_map(col, key)
                
                scalar_candidates = ["scene_annotation", "task_index", "frame_index", "episode_index", "index"]
                for col in df.columns:
                    should_flatten = False
                    if features_def and col in features_def:
                        target_shape = features_def[col].get("shape")
                        if target_shape is None or target_shape == 1 or target_shape == [1]:
                             should_flatten = True
                    elif col in scalar_candidates:
                        should_flatten = True
                    
                    if should_flatten:
                        first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                        if isinstance(first_val, (list, np.ndarray)):
                             df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else (x if x is not None else 0))

                target_schema = _build_pyarrow_schema(df.columns, features_def)
                
                try:
                    table = pa.Table.from_pandas(df, schema=target_schema, preserve_index=False)
                except Exception as cast_err:
                    print(f"Pyarrow casting error in {source_path}: {cast_err}")
                    for col in df.columns:
                        if target_schema.field(col).type == pa.float32():
                            df[col] = df[col].astype("float32")
                        elif target_schema.field(col).type == pa.int64():
                            df[col] = df[col].astype("int64")
                        elif target_schema.field(col).type == pa.int32():
                            df[col] = df[col].astype("int32")
                    table = pa.Table.from_pandas(df, schema=target_schema, preserve_index=False)

                chunk_index = new_index // chunks_size
                chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
                os.makedirs(chunk_dir, exist_ok=True)
                dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")
                
                pq.write_table(table, dest_path)
                
                total_copied += 1
                if total_copied % 10 == 0:
                    print(f"Processed {total_copied} files...", end='\r')

            except Exception as e:
                print(f"Error processing {source_path}: {e}")
                traceback.print_exc()
                failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                total_failed += 1
        else:
            print(f"Warning: File not found for episode {old_index} in {old_folder}")
            total_failed += 1

    print(f"\n共复制 {total_copied} 个数据文件，{total_failed} 个失败")
    return total_copied > 0


def pad_parquet_data(source_path, target_path, original_dim=14, target_dim=18):
    """
    通过零填充将parquet数据从原始维度扩展到目标维度
    (Extend parquet data from original dimension to target dimension by zero-padding)

    Args:
        source_path (str): 源parquet文件路径 (Source parquet file path)
        target_path (str): 目标parquet文件路径 (Target parquet file path)
        original_dim (int): 原始向量维度 (Original vector dimension)
        target_dim (int): 目标向量维度 (Target vector dimension)
    """
    # 读取parquet文件
    df = pd.read_parquet(source_path)

    # 打印列名以便调试
    print(f"Columns in {source_path}: {df.columns.tolist()}")

    # 创建新的DataFrame来存储填充后的数据
    new_df = df.copy()

    # 检查observation.state和action列是否存在
    if "observation.state" in df.columns:
        # 检查第一行数据，确认是否为向量
        first_state = df["observation.state"].iloc[0]
        print(f"First observation.state type: {type(first_state)}, value: {first_state}")

        # 如果是向量（列表或numpy数组）
        if isinstance(first_state, (list, np.ndarray)):
            # 检查维度
            state_dim = len(first_state)
            print(f"observation.state dimension: {state_dim}")

            if state_dim < target_dim:
                # 填充向量
                print(f"Padding observation.state from {state_dim} to {target_dim} dimensions")
                new_df["observation.state"] = df["observation.state"].apply(
                    lambda x: np.pad(x, (0, target_dim - len(x)), "constant").tolist()
                )

    # 同样处理action列
    if "action" in df.columns:
        # 检查第一行数据
        first_action = df["action"].iloc[0]
        print(f"First action type: {type(first_action)}, value: {first_action}")

        # 如果是向量
        if isinstance(first_action, (list, np.ndarray)):
            # 检查维度
            action_dim = len(first_action)
            print(f"action dimension: {action_dim}")

            if action_dim < target_dim:
                # 填充向量
                print(f"Padding action from {action_dim} to {target_dim} dimensions")
                new_df["action"] = df["action"].apply(
                    lambda x: np.pad(x, (0, target_dim - len(x)), "constant").tolist()
                )

    # 确保目标目录存在
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # 保存到新的parquet文件
    new_df.to_parquet(target_path, index=False)

    print(f"已将{source_path}处理并保存到{target_path}")

    return new_df


def write_meta_and_copy(
    source_folders,
    output_folder,
    episode_mapping,
    all_episodes,
    all_episodes_stats,
    folder_dimensions,
    folder_task_mapping,
    folder_annotations_mapping,
    episode_to_frame_index, # 注意：这是基于原始长度的旧映射，下面会重新计算
    all_stats_data,
    all_tasks,
    all_annotations,
    total_frames,
    max_dim_cli,
    fps,
    features_to_keep=None,
    video_size=None,
    remove_eef=False
):

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    episode_source_map = {new_idx: folder for folder, _, new_idx in episode_mapping}
    folder_indices_map = {} 
    global_max_dims_map = {} 
    max_dim_source_folder = {} 
    
    # [修复] 必须在这里初始化，否则下面循环会报错
    folder_mapping_instructions = {} 

    base_info = get_info(source_folders[0])
    
    for folder in source_folders:
        current_info = get_info(folder)
        folder_mapping_instructions[folder] = {} # 初始化
        
        if remove_eef:
            # [修改] 使用新的 get_standardized_mapping
            for feat in ["observation.state", "action"]:
                if feat in current_info.get("features", {}):
                    names = current_info["features"][feat].get("names", [])
                    mapping_instr, _ = get_standardized_mapping(names)
                    if mapping_instr:
                        folder_mapping_instructions[folder][feat] = mapping_instr

    # [修改] 强制最大维度为 16
    actual_max_dim = 16 
    if remove_eef:
        print(f"!!! 强制标准化维度: 16 (Left:7+1, Right:7+1) !!!")
        # 更新 info.json 的 names
        for feat in ["observation.state", "action"]:
            if feat in base_info.get("features", {}):
                base_info["features"][feat]["shape"] = [actual_max_dim]
                base_info["features"][feat]["names"] = TARGET_NAMES
    else:
        actual_max_dim = max_dim_cli or max(folder_dimensions.values())

    if remove_eef:
        for feat in ["observation.state", "action"]:
            if feat in global_max_dims_map:
                if feat not in base_info["features"]:
                    continue 

                base_info["features"][feat]["shape"] = [actual_max_dim]
                
                src_folder = max_dim_source_folder.get(feat)
                if src_folder:
                    src_info = get_info(src_folder)
                    src_names = src_info["features"][feat].get("names", [])
                    if src_names:
                        _, kept_names = get_safe_indices(src_names)
                        if len(kept_names) < actual_max_dim:
                            kept_names += [f"pad_{i}" for i in range(len(kept_names), actual_max_dim)]
                        base_info["features"][feat]["names"] = kept_names

    RESERVED_FEATURES = SYSTEM_RESERVED_FIELDS
    if features_to_keep:
        print(f"筛选特定字段: {features_to_keep}")
        original_features = base_info.get("features", {})
        filtered_features = {}
        all_wanted = set(features_to_keep) | set(RESERVED_FEATURES)
        for feat in all_wanted:
            if feat in original_features:
                feat_def = original_features[feat].copy()
                if feat_def.get("dtype") == "video" and video_size is not None:
                    tw, th = video_size
                    feat_def["shape"] = [th, tw, 3]
                    if "info" in feat_def:
                        feat_def["info"] = feat_def["info"].copy()
                        feat_def["info"]["video.width"] = tw
                        feat_def["info"]["video.height"] = th
                filtered_features[feat] = feat_def
        base_info["features"] = filtered_features
        video_keys = [k for k in get_video_keys({"features": filtered_features})]
    else:
        video_keys = get_video_keys(base_info)
    
    chunks_size = get_chunks_size(base_info)
    total_episodes = len(all_episodes)
    total_videos = len(video_keys) * total_episodes

    aligned_episode_stats = []
    features_set = set(features_to_keep) if features_to_keep else None

    for stats in all_episodes_stats:
        if "stats" in stats:
            current_stats = stats["stats"]
            ep_idx = stats.get("episode_index")
            source_folder = episode_source_map.get(ep_idx)
            current_folder_indices = folder_indices_map.get(source_folder, {})
            current_mappings = folder_mapping_instructions.get(source_folder, {})
            new_stats_content = {}
            for k, v in current_stats.items():
                keep = False
                if features_set is None:
                    keep = True
                elif k in features_set or k in RESERVED_FEATURES:
                    keep = True
                else:
                    pass 
                
                if keep:
                    # [修改] 如果是 state/action 且有映射指令，应用新函数
                    if remove_eef and k in ["observation.state", "action"] and k in current_mappings:
                        new_stats_content[k] = apply_mapping_to_stats(v, current_mappings[k])
                    else:
                        # 旧的切片逻辑或原样保留
                        new_stats_content[k] = v
            stats["stats"] = new_stats_content

        pad_episode_stats(stats, from_dim=actual_max_dim, to_dim=actual_max_dim)
        aligned_episode_stats.append(stats)

    # === [FIX] 核心修复点 1: 重建 episode_to_frame_index 映射 ===
    # 因为重采样后，episode长度变了，旧的映射表（基于源数据长度）已经失效
    # 必须基于新的长度重新计算，否则 data parquet 的 index 列会错乱
    total_frames_resampled = 0
    folder_fps_cache = {} 
    
    # 新的映射表
    new_episode_to_frame_index = {}

    for ep in all_episodes:
        src_folder = episode_source_map[ep["episode_index"]]
        if src_folder not in folder_fps_cache:
            # [FIX] 统一使用 30 作为默认值，而不是 fallback 到 target fps，确保缺少 FPS 时能触发重采样
            folder_fps_cache[src_folder] = get_info(src_folder).get("fps", 30)
        
        src_fps = folder_fps_cache[src_folder]
        
        # 记录重采样后的起始帧
        new_episode_to_frame_index[ep["episode_index"]] = total_frames_resampled

        if fps is not None and src_fps != fps:
            original_len = ep["length"]
            duration = original_len / src_fps
            new_len = int(np.ceil(duration * fps))
            ep["length"] = new_len
        
        total_frames_resampled += ep["length"]

    # [FIX] 覆盖旧的映射表，传给 copy_data_files
    episode_to_frame_index = new_episode_to_frame_index

    # 设置总帧数为重采样后的值
    base_info["total_frames"] = total_frames_resampled

    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(aligned_episode_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))

    used_task_indices = set()
    for episode in all_episodes:
        if "task_index" in episode:
            used_task_indices.add(episode["task_index"])
    if not used_task_indices:
        for stats in aligned_episode_stats:
            if "stats" in stats and "task_index" in stats["stats"]:
                task_idx_info = stats["stats"]["task_index"]
                if isinstance(task_idx_info, dict) and "min" in task_idx_info:
                    used_task_indices.add(int(task_idx_info["min"][0]))
                else:
                    used_task_indices.add(int(task_idx_info))
    filtered_tasks = [t for t in all_tasks if t["task_index"] in used_task_indices]
    save_jsonl(filtered_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)
    if all_annotations:
        for key, val in all_annotations.items():
            if val:
                save_jsonl(val, os.path.join(output_folder, "annotations", f"{key}.jsonl"))

    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats_list.append(json.load(f))
    if stats_list:
        merged_stats = merge_stats(stats_list)
        recalc_merged_stats_with_episode_stats(merged_stats, all_stats_data, target_dim=actual_max_dim)
        if features_to_keep:
            merged_stats = {k: v for k, v in merged_stats.items() if k in features_set or k in RESERVED_FEATURES}
        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    base_info["total_episodes"] = total_episodes
    # [FIX] 核心修复点 2: 删除了这里覆盖 total_frames 的代码
    # base_info["total_frames"] = total_frames <-- 删除这行，保留上面的 total_frames_resampled
    
    base_info["total_tasks"] = len(filtered_tasks)
    base_info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
    base_info["splits"] = {"train": f"0:{total_episodes}"}
    base_info["fps"] = fps
    base_info["total_videos"] = total_videos
    for feature_name in ["observation.state", "action"]:
        if feature_name in base_info.get("features", {}) and "shape" in base_info["features"][feature_name]:
            base_info["features"][feature_name]["shape"] = [actual_max_dim]

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(base_info, f, indent=4)

    if video_keys:
        # [FIX] 传递 fps 参数给 copy_videos
        copy_videos(source_folders, output_folder, episode_mapping, video_size=video_size, filtered_features=base_info["features"], fps=fps)
    
    copy_data_files(
        source_folders=source_folders,
        output_folder=output_folder,
        episode_mapping=episode_mapping,
        max_dim=actual_max_dim, 
        fps=fps,
        episode_to_frame_index=episode_to_frame_index, # [FIX] 传入更新后的映射表
        folder_task_mapping=folder_task_mapping,
        folder_annotations_mapping=folder_annotations_mapping,
        chunks_size=chunks_size,
        features_to_keep=features_to_keep,
        remove_eef_flag=remove_eef,
        features_def=base_info["features"],
        folder_mapping_instructions=folder_mapping_instructions # [Fix] 传入映射指令
    )
    print(f"Done: {total_episodes} episodes, {total_frames_resampled} frames (resampled), output={output_folder}")