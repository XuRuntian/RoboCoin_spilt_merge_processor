import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pandas as pd
import cv2
# 全局保留字段列表
SYSTEM_RESERVED_FIELDS = [
    "episode_index", "frame_index", "timestamp", "index", "task_index", 
    "subtask_annotation", "scene_annotation", 
    "eef_direction_state", "eef_direction_action",
    "eef_velocity_state", "eef_velocity_action",
    "gripper_mode_state", "gripper_mode_action",
    "gripper_activity_state", "gripper_activity_action"
]

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

def process_video_pad(src_path, dst_path, target_w, target_h):
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
        "-c:v", "libx264", 
        "-pix_fmt", "yuv420p",
        dst_path
    ]

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
    episode_to_frame_index,
    all_stats_data,
    all_tasks,
    all_annotations,
    total_frames,
    max_dim_cli,
    fps,
    features_to_keep=None,  # 新增参数
    video_size=None, # 新增参数
    remove_eef=False
):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    base_info = get_info(source_folders[0])
    original_max_dim = max_dim_cli or max(folder_dimensions.values())
    new_shapes = {}
    
    if remove_eef:
        for feat in ["observation.state", "action"]:
            if feat in base_info["features"]:
                names = base_info["features"][feat].get("names", [])
                if names:
                    _, kept_names = get_safe_indices(names)
                    base_info["features"][feat]["names"] = kept_names
                    # 更新 shape，例如从 [41] 变为 [16]
                    new_dim = len(kept_names)
                    base_info["features"][feat]["shape"] = [new_dim]
                    new_shapes[feat] = new_dim
                    print(f"[{feat}] 剔除EEF信息: 原维度 {len(names)} -> 新维度 {new_dim}")
    if "observation.state" in new_shapes:
        actual_max_dim = new_shapes["observation.state"]
        print(f"!!! 全局维度已更新: {original_max_dim} -> {actual_max_dim} !!!")
    elif max_dim_cli:
        actual_max_dim = max_dim_cli
    # === 修改逻辑开始：过滤 info.json 中的 features ===
    RESERVED_FEATURES = SYSTEM_RESERVED_FIELDS
    if features_to_keep:
        print(f"筛选特定字段: {features_to_keep}")
        # 保留必要的元数据键，但过滤 features
        original_features = base_info.get("features", {})
        filtered_features = {}
        all_wanted = set(features_to_keep) | set(RESERVED_FEATURES)
        for feat in all_wanted:
            if feat in original_features:
                feat_def = original_features[feat].copy() # 必须用 copy
                
                # === 关键修复：更新 info.json 里的分辨率数值 ===
                if feat_def.get("dtype") == "video" and video_size is not None:
                    tw, th = video_size
                    feat_def["shape"] = [th, tw, 3] # 设置为新分辨率
                    if "info" in feat_def:
                        feat_def["info"] = feat_def["info"].copy()
                        feat_def["info"]["video.width"] = tw
                        feat_def["info"]["video.height"] = th

                filtered_features[feat] = feat_def
        base_info["features"] = filtered_features
        
        # 同时更新 video_keys，如果用户没选视频字段，就不应该复制视频
        video_keys = [k for k in get_video_keys({"features": filtered_features})]
    else:
        video_keys = get_video_keys(base_info)
    # === 修改逻辑结束 ===
    chunks_size = get_chunks_size(base_info)

    total_episodes = len(all_episodes)
    total_videos = len(video_keys) * total_episodes


    # 对齐并保存 episodes_stats
    aligned_episode_stats = []
    for stats in all_episodes_stats:
        pad_episode_stats(stats, from_dim=actual_max_dim, to_dim=actual_max_dim)
        aligned_episode_stats.append(stats)

    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(aligned_episode_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))

    # 过滤并保存 tasks（仅保留使用到的任务索引）
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
        if all_annotations.get("subtask_annotations"):
            save_jsonl(all_annotations["subtask_annotations"], os.path.join(output_folder, "annotations", "subtask_annotations.jsonl"))
        if all_annotations.get("scene_annotations"):
            save_jsonl(all_annotations["scene_annotations"], os.path.join(output_folder, "annotations", "scene_annotations.jsonl"))
        if all_annotations.get("gripper_mode_annotation"):
            save_jsonl(all_annotations["gripper_mode_annotation"], os.path.join(output_folder, "annotations", "gripper_mode_annotation.jsonl"))
        if all_annotations.get("gripper_activity_annotation"):
            save_jsonl(all_annotations["gripper_activity_annotation"], os.path.join(output_folder, "annotations", "gripper_activity_annotation.jsonl"))
        if all_annotations.get("eef_direction_annotation"):
            save_jsonl(all_annotations["eef_direction_annotation"], os.path.join(output_folder, "annotations", "eef_direction_annotation.jsonl"))
        if all_annotations.get("eef_velocity_annotation"):
            save_jsonl(all_annotations["eef_velocity_annotation"], os.path.join(output_folder, "annotations", "eef_velocity_annotation.jsonl"))
        if all_annotations.get("eef_acc_mag_annotation"):
            save_jsonl(all_annotations["eef_acc_mag_annotation"], os.path.join(output_folder, "annotations", "eef_acc_mag_annotation.jsonl"))

    # 合并并保存 stats.json
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
            merged_stats = {k: v for k, v in merged_stats.items() if k in features_to_keep}
        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    # 更新 info.json
    base_info["total_episodes"] = total_episodes
    base_info["total_frames"] = total_frames
    base_info["total_tasks"] = len(filtered_tasks)
    base_info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
    base_info["splits"] = {"train": f"0:{total_episodes}"}
    base_info["fps"] = fps

    for feature_name in ["observation.state", "action"]:
        if feature_name in base_info.get("features", {}) and "shape" in base_info["features"][feature_name]:
            base_info["features"][feature_name]["shape"] = [actual_max_dim]
    base_info["total_videos"] = total_videos

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(base_info, f, indent=4)

    # 复制视频和数据文件
    if video_keys:
        copy_videos(source_folders, output_folder, episode_mapping, video_size=video_size, filtered_features=base_info["features"])
    copy_data_files(
        source_folders=source_folders,
        output_folder=output_folder,
        episode_mapping=episode_mapping,
        max_dim=actual_max_dim,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_task_mapping=folder_task_mapping,
        folder_annotations_mapping=folder_annotations_mapping,
        chunks_size=chunks_size,
        features_to_keep=features_to_keep,
        remove_eef_flag=remove_eef,
    )
    print(f"Done: {total_episodes} episodes, {total_frames} frames, output={output_folder}")


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


def copy_videos(source_folders, output_folder, episode_mapping, video_size=None, filtered_features=None):
    """
    从源文件夹复制视频文件到输出文件夹，保持正确的索引和结构
    (Copy video files from source folders to output folder, maintaining correct indices and structure)
    """
    # Get info.json to determine video structure
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]
    
    # === 决定扫描哪些 Video Key ===
    # 如果传入了 filtered_features，只处理其中的 video 类型字段
    features_to_scan = filtered_features if filtered_features is not None else info.get("features", {})
    
    video_keys = []
    for k, v in features_to_scan.items():
        if v.get("dtype") == "video":
            video_keys.append(k)
    
    print(f"即将复制的视频流: {video_keys}")

    # Copy videos for each episode
    for old_folder, old_index, new_index in episode_mapping:
        # Determine episode chunk (usually 0 for small datasets)
        episode_chunk = old_index // info["chunks_size"]
        new_episode_chunk = new_index // info["chunks_size"]

        for video_key in video_keys:
            # Try different possible source paths
            source_patterns = [
                # Standard path with the episode index from metadata
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=episode_chunk, video_key=video_key, episode_index=old_index
                    ),
                ),
                # Try with 0-based indexing
                os.path.join(
                    old_folder,
                    video_path_template.format(episode_chunk=0, video_key=video_key, episode_index=0),
                ),
                # Try with different formatting
                os.path.join(
                    old_folder, f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4"
                ),
                # Fallback for some datasets
                os.path.join(old_folder, f"videos/chunk-000/{video_key}/episode_000000.mp4"),
            ]

            # Find the first existing source path
            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break

            if source_video_path:
                # Construct destination path
                dest_video_path = os.path.join(
                    output_folder,
                    video_path_template.format(
                        episode_chunk=new_episode_chunk, video_key=video_key, episode_index=new_index
                    ),
                )

                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                # === 处理逻辑 ===
                if video_size is not None:
                    target_w, target_h = video_size
                    print(f"Processing video: {source_video_path} -> {dest_video_path}") 
                    try:
                        process_video_pad(source_video_path, dest_video_path, target_w, target_h)
                    except Exception as e:
                        print(f"Error processing video {source_video_path}: {e}")
                        # 失败时回退到直接复制
                        shutil.copy2(source_video_path, dest_video_path)
                else:
                    # 原有逻辑：直接复制
                    # print(f"Copying video: {source_video_path} -> {dest_video_path}")
                    shutil.copy2(source_video_path, dest_video_path)
            else:
                # === 修改点：删除了原来的递归搜索 (os.walk) ===
                # 如果标准路径找不到，直接报警告，不进行模糊搜索
                print(
                    f"Warning: Video file not found for {video_key}, episode {old_index} in {old_folder}"
                )

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
    default_fps=20,
    features_to_keep=None, # 新增参数
    remove_eef_flag=False
):
    """
    复制并处理parquet数据文件，包括维度填充和索引更新
    (Copy and process parquet data files, including dimension padding and index updates)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        output_folder (str): 输出文件夹路径 (Output folder path)
        episode_mapping (list): 包含(旧文件夹,旧索引,新索引)元组的列表
                               (List of tuples containing (old_folder, old_index, new_index))
        max_dim (int): 向量的最大维度 (Maximum dimension for vectors)
        fps (float, optional): 帧率，如果未提供则从第一个数据集获取 (Frame rate, if not provided will be obtained from the first dataset)
        episode_to_frame_index (dict, optional): 每个新episode索引对应的起始帧索引映射
                                               (Mapping of each new episode index to its starting frame index)
        folder_task_mapping (dict, optional): 每个文件夹中task_index的映射关系
                                            (Mapping of task_index for each folder)
        chunks_size (int): 每个chunk包含的episode数量 (Number of episodes per chunk)
        default_fps (float): 默认帧率，当无法从数据集获取时使用 (Default frame rate when unable to obtain from dataset)

    Returns:
        bool: 是否成功复制了至少一个文件 (Whether at least one file was successfully copied)
    """
    # 获取第一个数据集的FPS（如果未提供）(Get FPS from first dataset if not provided)
    if fps is None:
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                fps = info.get(
                    "fps", default_fps
                )  # 使用变量替代硬编码的20 (Use variable instead of hardcoded 20)
        else:
            fps = default_fps  # 使用变量替代硬编码的20 (Use variable instead of hardcoded 20)

    print(f"使用FPS={fps} (Using FPS={fps})")

    # 为每个episode复制和处理数据文件 (Copy and process data files for each episode)
    total_copied = 0
    total_failed = 0

    # 添加一个列表来记录失败的文件及原因
    # (Add a list to record failed files and reasons)
    failed_files = []
    RESERVED_COLUMNS = SYSTEM_RESERVED_FIELDS
    # === 辅助函数：统一的列过滤逻辑 ===
    def _filter_columns(dataframe):
        if features_to_keep:
            cols_to_retain = set(features_to_keep)
            final_columns = []
            for col in dataframe.columns:
                # 保留条件：用户指定 或 系统保留 或 指定字段的子属性
                if (col in cols_to_retain or 
                    col in RESERVED_COLUMNS or 
                    any(col.startswith(f"{k}.") for k in cols_to_retain)):
                    final_columns.append(col)
            
            if final_columns:
                return dataframe[final_columns]
        return dataframe
    # ===================================
    folder_meta_cache = {}

    for i, (old_folder, old_index, new_index) in enumerate(episode_mapping):
        # 尝试找到源parquet文件 (Try to find source parquet file)
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

        if source_path:
            try:
                # 读取parquet文件 (Read parquet file)
                # 读取
                df = pd.read_parquet(source_path) 
                # 过滤列
                df = _filter_columns(df)
                # === 修改点2：更新过滤逻辑 ===
                # 剔除
                if remove_eef_flag:
                    if old_folder not in folder_meta_cache:
                        src_info = get_info(old_folder)
                        folder_meta_cache[old_folder] = src_info["features"]
                    src_features = folder_meta_cache[old_folder]
                    for feat in ["observation.state", "action"]:
                        if feat in df.columns and feat in src_features:
                            # 获取该数据集原始的 names
                            original_names = src_features[feat].get("names", [])
                            if original_names:
                                # 计算需要保留的索引
                                keep_idxs, _ = get_safe_indices(original_names)
                                
                                # 执行 numpy 切片：只保留 indices 对应的数据
                                # 假设数据是 list 或 np.ndarray
                                df[feat] = df[feat].apply(
                                    lambda x: np.array(x)[keep_idxs].tolist() 
                                    if isinstance(x, (list, np.ndarray)) and len(x) >= len(original_names) # 暂时大于等于，因为数据有问题
                                    else x
                                )
                if features_to_keep:
                    cols_to_retain = set(features_to_keep)
                    final_columns = []
                    
                    for col in df.columns:
                        # 保留条件：
                        # 1. 是用户指定的 (如 observation.state)
                        # 2. 是强制保留字段 (如 timestamp)
                        # 3. 是用户指定字段的子属性 (如 observation.images.cam_high.path)
                        if (col in cols_to_retain or 
                            col in RESERVED_COLUMNS or 
                            any(col.startswith(f"{k}.") for k in cols_to_retain)):
                            final_columns.append(col)
                    
                    # 只有当筛选结果不为空时才覆盖，防止误删所有
                    if final_columns:
                        df = df[final_columns]
                # 检查是否需要填充维度 (Check if dimensions need padding)
                for feature in ["observation.state", "action"]:
                    if feature in df.columns:
                        # 检查第一个非空值 (Check first non-null value)
                        for _idx, value in enumerate(df[feature]):
                            if value is not None and isinstance(value, (list, np.ndarray)):
                                current_dim = len(value)
                                if current_dim < max_dim:
                                    print(
                                        f"填充 {feature} 从 {current_dim} 维到 {max_dim} 维 (Padding {feature} from {current_dim} to {max_dim} dimensions)"
                                    )
                                    # 使用零填充到目标维度 (Pad with zeros to target dimension)
                                    df[feature] = df[feature].apply(
                                        lambda x: np.pad(x, (0, max_dim - len(x)), "constant").tolist()
                                        if x is not None
                                        and isinstance(x, (list, np.ndarray))
                                        and len(x) < max_dim
                                        else x
                                    )
                                break

                # 更新episode_index列 (Update episode_index column)
                if "episode_index" in df.columns:
                    print(
                        f"更新episode_index从 {df['episode_index'].iloc[0]} 到 {new_index} (Update episode_index from {df['episode_index'].iloc[0]} to {new_index})"
                    )
                    df["episode_index"] = new_index

                # 更新index列 (Update index column)
                if "index" in df.columns:
                    if episode_to_frame_index and new_index in episode_to_frame_index:
                        # 使用预先计算的帧索引起始值 (Use pre-calculated frame index start value)
                        first_index = episode_to_frame_index[new_index]
                        print(
                            f"更新index列，起始值: {first_index}（使用全局累积帧计数）(Update index column, start value: {first_index} (using global cumulative frame count))"
                        )
                    else:
                        # 如果没有提供映射，使用当前的计算方式作为回退
                        # (If no mapping provided, use current calculation as fallback)
                        first_index = new_index * len(df)
                        print(
                            f"更新index列，起始值: {first_index}（使用episode索引乘以长度）(Update index column, start value: {first_index} (using episode index multiplied by length))"
                        )

                    # 更新所有帧的索引 (Update indices for all frames)
                    df["index"] = [first_index + i for i in range(len(df))]

                # 更新task_index列 (Update task_index column)
                if "task_index" in df.columns and folder_task_mapping and old_folder in folder_task_mapping:
                    current_task_index = df["task_index"].iloc[0]
                    if current_task_index in folder_task_mapping[old_folder]:
                        new_task_index = folder_task_mapping[old_folder][current_task_index]
                        print(
                            f"更新task_index从 {current_task_index} 到 {new_task_index} (Update task_index from {current_task_index} to {new_task_index})"
                        )
                        df["task_index"] = new_task_index
                    else:
                        print(
                            f"警告: 找不到task_index {current_task_index}的映射关系 (Warning: No mapping found for task_index {current_task_index})"
                        )

                if folder_annotations_mapping and old_folder in folder_annotations_mapping:
                    ann_map = folder_annotations_mapping[old_folder]
                    if "subtask_annotation" in df.columns and "subtask_annotation" in ann_map:
                        df["subtask_annotation"] = df["subtask_annotation"].apply(
                            lambda x: [ann_map["subtask_annotation"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                        )
                    if "scene_annotation" in df.columns and "scene_annotation" in ann_map:
                        df["scene_annotation"] = df["scene_annotation"].apply(
                            lambda x: [ann_map["scene_annotation"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                        )
                    for col in ["gripper_mode_state", "gripper_mode_action"]:
                        if col in df.columns and "gripper_mode" in ann_map:
                            df[col] = df[col].apply(
                                lambda x: [ann_map["gripper_mode"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                            )
                    for col in ["gripper_activity_state", "gripper_activity_action"]:
                        if col in df.columns and "gripper_activity" in ann_map:
                            df[col] = df[col].apply(
                                lambda x: [ann_map["gripper_activity"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                            )
                    for col in ["eef_direction_state", "eef_direction_action"]:
                        if col in df.columns and "eef_direction" in ann_map:
                            df[col] = df[col].apply(
                                lambda x: [ann_map["eef_direction"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                            )
                    for col in ["eef_velocity_state", "eef_velocity_action"]:
                        if col in df.columns and "eef_velocity" in ann_map:
                            df[col] = df[col].apply(
                                lambda x: [ann_map["eef_velocity"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                            )
                    for col in ["eef_acc_mag_state", "eef_acc_mag_action"]:
                        if col in df.columns and "eef_acc_mag" in ann_map:
                            df[col] = df[col].apply(
                                lambda x: [ann_map["eef_acc_mag"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                            )

                # 计算chunk编号 (Calculate chunk number)
                chunk_index = new_index // chunks_size

                # 创建正确的目标目录 (Create correct target directory)
                chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
                os.makedirs(chunk_dir, exist_ok=True)

                # 构建正确的目标路径 (Build correct target path)
                dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")

                # 保存到正确位置 (Save to correct location)
                df.to_parquet(dest_path, index=False)

                total_copied += 1
                print(f"已处理并保存: {dest_path} (Processed and saved: {dest_path})")

            except Exception as e:
                error_msg = f"处理 {source_path} 失败: {e} (Processing {source_path} failed: {e})"
                print(error_msg)
                traceback.print_exc()
                failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                total_failed += 1
        else:
            # 文件不在标准位置，尝试递归搜索
            found = False
            for root, _, files in os.walk(old_folder):
                for file in files:
                    if file.endswith(".parquet") and f"episode_{old_index:06d}" in file:
                        try:
                            source_path = os.path.join(root, file)

                            # 读取parquet文件 (Read parquet file)
                            df = pd.read_parquet(source_path)
                            df = _filter_columns(df)

                            if remove_eef_flag:
                                if old_folder not in folder_meta_cache:
                                    src_info = get_info(old_folder)
                                    folder_meta_cache[old_folder] = src_info["features"]
                                src_features = folder_meta_cache[old_folder]
                                for feat in ["observation.state", "action"]:
                                    if feat in df.columns and feat in src_features:
                                        # 获取该数据集原始的 names
                                        original_names = src_features[feat].get("names", [])
                                        if original_names:
                                            # 计算需要保留的索引
                                            keep_idxs, _ = get_safe_indices(original_names)
                                            
                                            # 执行 numpy 切片：只保留 indices 对应的数据
                                            # 假设数据是 list 或 np.ndarray
                                            df[feat] = df[feat].apply(
                                                lambda x: np.array(x)[keep_idxs].tolist() 
                                                if isinstance(x, (list, np.ndarray)) and len(x) >= len(original_names)  # 暂时大于等于，因为数据有问题 shape 与 names 数量没对上
                                                else x
                                            )

                            if features_to_keep:
                                cols_to_retain = set(features_to_keep)
                                final_columns = []
                                
                                for col in df.columns:
                                    # 保留条件：
                                    # 1. 是用户指定的 (如 observation.state)
                                    # 2. 是强制保留字段 (如 timestamp)
                                    # 3. 是用户指定字段的子属性 (如 observation.images.cam_high.path)
                                    if (col in cols_to_retain or 
                                        col in RESERVED_COLUMNS or 
                                        any(col.startswith(f"{k}.") for k in cols_to_retain)):
                                        final_columns.append(col)
                                
                                # 只有当筛选结果不为空时才覆盖，防止误删所有
                                if final_columns:
                                    df = df[final_columns]                            
                            # === 修改点2：更新过滤逻辑 ===
                            # 检查是否需要填充维度 (Check if dimensions need padding)
 
                            for feature in ["observation.state", "action"]:
                                if feature in df.columns:
                                    # 检查第一个非空值 (Check first non-null value)
                                    for _idx, value in enumerate(df[feature]):
                                        if value is not None and isinstance(value, (list, np.ndarray)):
                                            current_dim = len(value)
                                            if current_dim < max_dim:
                                                print(
                                                    f"填充 {feature} 从 {current_dim} 维到 {max_dim} 维 (Padding {feature} from {current_dim} to {max_dim} dimensions)"
                                                )
                                                # 使用零填充到目标维度 (Pad with zeros to target dimension)
                                                df[feature] = df[feature].apply(
                                                    lambda x: np.pad(
                                                        x, (0, max_dim - len(x)), "constant"
                                                    ).tolist()
                                                    if x is not None
                                                    and isinstance(x, (list, np.ndarray))
                                                    and len(x) < max_dim
                                                    else x
                                                )
                                            break

                            # 更新episode_index列 (Update episode_index column)
                            if "episode_index" in df.columns:
                                print(
                                    f"更新episode_index从 {df['episode_index'].iloc[0]} 到 {new_index} (Update episode_index from {df['episode_index'].iloc[0]} to {new_index})"
                                )
                                df["episode_index"] = new_index

                            # 更新index列 (Update index column)
                            if "index" in df.columns:
                                if episode_to_frame_index and new_index in episode_to_frame_index:
                                    # 使用预先计算的帧索引起始值 (Use pre-calculated frame index start value)
                                    first_index = episode_to_frame_index[new_index]
                                    print(
                                        f"更新index列，起始值: {first_index}（使用全局累积帧计数）(Update index column, start value: {first_index} (using global cumulative frame count))"
                                    )
                                else:
                                    # 如果没有提供映射，使用当前的计算方式作为回退
                                    # (If no mapping provided, use current calculation as fallback)
                                    first_index = new_index * len(df)
                                    print(
                                        f"更新index列，起始值: {first_index}（使用episode索引乘以长度）(Update index column, start value: {first_index} (using episode index multiplied by length))"
                                    )

                                # 更新所有帧的索引 (Update indices for all frames)
                                df["index"] = [first_index + i for i in range(len(df))]

                            # 更新task_index列 (Update task_index column)
                            if (
                                "task_index" in df.columns
                                and folder_task_mapping
                                and old_folder in folder_task_mapping
                            ):
                                current_task_index = df["task_index"].iloc[0]
                                if current_task_index in folder_task_mapping[old_folder]:
                                    new_task_index = folder_task_mapping[old_folder][current_task_index]
                                    print(
                                        f"更新task_index从 {current_task_index} 到 {new_task_index} (Update task_index from {current_task_index} to {new_task_index})"
                                    )
                                    df["task_index"] = new_task_index
                                else:
                                    print(
                                        f"警告: 找不到task_index {current_task_index}的映射关系 (Warning: No mapping found for task_index {current_task_index})"
                                    )

                            if folder_annotations_mapping and old_folder in folder_annotations_mapping:
                                ann_map = folder_annotations_mapping[old_folder]
                                if "subtask_annotation" in df.columns and "subtask_annotation" in ann_map:
                                    df["subtask_annotation"] = df["subtask_annotation"].apply(
                                        lambda x: [ann_map["subtask_annotation"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                    )
                                if "scene_annotation" in df.columns and "scene_annotation" in ann_map:
                                    df["scene_annotation"] = df["scene_annotation"].apply(
                                        lambda x: [ann_map["scene_annotation"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                    )
                                for col in ["gripper_mode_state", "gripper_mode_action"]:
                                    if col in df.columns and "gripper_mode" in ann_map:
                                        df[col] = df[col].apply(
                                            lambda x: [ann_map["gripper_mode"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                        )
                                for col in ["gripper_activity_state", "gripper_activity_action"]:
                                    if col in df.columns and "gripper_activity" in ann_map:
                                        df[col] = df[col].apply(
                                            lambda x: [ann_map["gripper_activity"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                        )
                                for col in ["eef_direction_state", "eef_direction_action"]:
                                    if col in df.columns and "eef_direction" in ann_map:
                                        df[col] = df[col].apply(
                                            lambda x: [ann_map["eef_direction"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                        )
                                for col in ["eef_velocity_state", "eef_velocity_action"]:
                                    if col in df.columns and "eef_velocity" in ann_map:
                                        df[col] = df[col].apply(
                                            lambda x: [ann_map["eef_velocity"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                        )
                                for col in ["eef_acc_mag_state", "eef_acc_mag_action"]:
                                    if col in df.columns and "eef_acc_mag" in ann_map:
                                        df[col] = df[col].apply(
                                            lambda x: [ann_map["eef_acc_mag"].get(int(v), int(v)) for v in (x if isinstance(x, (list, np.ndarray)) else [x])] if x is not None else x
                                        )

                            # 计算chunk编号 (Calculate chunk number)
                            chunk_index = new_index // chunks_size

                            # 创建正确的目标目录 (Create correct target directory)
                            chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
                            os.makedirs(chunk_dir, exist_ok=True)

                            # 构建正确的目标路径 (Build correct target path)
                            dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")

                            # 保存到正确位置 (Save to correct location)
                            df.to_parquet(dest_path, index=False)

                            total_copied += 1
                            found = True
                            print(f"已处理并保存: {dest_path} (Processed and saved: {dest_path})")
                            break
                        except Exception as e:
                            error_msg = f"处理 {source_path} 失败: {e} (Processing {source_path} failed: {e})"
                            print(error_msg)
                            traceback.print_exc()
                            failed_files.append({"file": source_path, "reason": str(e), "episode": old_index})
                            total_failed += 1
                if found:
                    break

            if not found:
                error_msg = f"找不到episode {old_index}的parquet文件，源文件夹: {old_folder}"
                print(error_msg)
                failed_files.append(
                    {"file": f"episode_{old_index:06d}.parquet", "reason": "文件未找到", "folder": old_folder}
                )
                total_failed += 1

    print(f"共复制 {total_copied} 个数据文件，{total_failed} 个失败")

    # 打印所有失败的文件详情 (Print details of all failed files)
    if failed_files:
        print("\n失败的文件详情 (Details of failed files):")
        for i, failed in enumerate(failed_files):
            print(f"{i + 1}. 文件 (File): {failed['file']}")
            if "folder" in failed:
                print(f"   文件夹 (Folder): {failed['folder']}")
            if "episode" in failed:
                print(f"   Episode索引 (Episode index): {failed['episode']}")
            print(f"   原因 (Reason): {failed['reason']}")
            print("---")

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


def merge_datasets(
    source_folders, output_folder, validate_ts=False, tolerance_s=1e-4, max_dim=18, default_fps=20
):
    """
    将多个数据集文件夹合并为一个，处理索引、维度和元数据
    (Merge multiple dataset folders into one, handling indices, dimensions, and metadata)

    Args:
        source_folders (list): 源数据集文件夹路径列表 (List of source dataset folder paths)
        output_folder (str): 输出文件夹路径 (Output folder path)
        validate_ts (bool): 是否验证时间戳 (Whether to validate timestamps)
        tolerance_s (float): 时间戳不连续性的容差值，以秒为单位 (Tolerance for timestamp discontinuities in seconds)
        max_dim (int): 向量的最大维度 (Maximum dimension for vectors)
        default_fps (float): 默认帧率 (Default frame rate)

    这个函数执行以下操作:
    (This function performs the following operations:)
    1. 合并所有的episodes、tasks和stats (Merges all episodes, tasks and stats)
    2. 重新编号所有的索引以保持连续性 (Renumbers all indices to maintain continuity)
    3. 填充向量维度使其一致 (Pads vector dimensions for consistency)
    4. 更新元数据文件 (Updates metadata files)
    5. 复制并处理数据和视频文件 (Copies and processes data and video files)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    # 使用默认FPS
    fps = default_fps
    print(f"使用默认FPS值: {fps}")

    # Load episodes from all source folders
    all_episodes = []
    all_episodes_stats = []
    all_tasks = []

    total_frames = 0
    total_episodes = 0

    # Keep track of episode mapping (old_folder, old_index, new_index)
    episode_mapping = []

    # Collect all stats for proper merging
    all_stats_data = []

    # Track dimensions for each folder
    folder_dimensions = {}

    # 累积帧数与帧索引映射
    cumulative_frame_count = 0
    episode_to_frame_index = {}

    # 任务映射相关
    task_desc_to_new_index = {}
    folder_task_mapping = {}
    all_unique_tasks = []

    # 从info.json获取chunks_size
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    chunks_size = 1000  # 默认值
    if os.path.exists(info_path):
        with open(info_path) as f:
            info_first = json.load(f)
            chunks_size = info_first.get("chunks_size", 1000)

    # 使用更简单的方法计算视频总数 (Use simpler method to calculate total videos)
    total_videos = 0

    for folder in source_folders:
        try:
            # 从每个数据集的info.json直接获取total_videos
            folder_info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(folder_info_path):
                with open(folder_info_path) as f:
                    folder_info = json.load(f)
                    if "total_videos" in folder_info:
                        folder_videos = folder_info["total_videos"]
                        total_videos += folder_videos
                        print(
                            f"从{folder}的info.json中读取到视频数量: {folder_videos} (Read video count from {folder}'s info.json: {folder_videos})"
                        )

            # 检测维度
            folder_dim = max_dim
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".parquet"):
                        try:
                            df = pd.read_parquet(os.path.join(root, file))
                            if "observation.state" in df.columns:
                                first_state = df["observation.state"].iloc[0]
                                if isinstance(first_state, (list, np.ndarray)):
                                    folder_dim = len(first_state)
                                    print(f"Detected {folder_dim} dimensions in {folder}")
                                    break
                        except Exception as e:
                            print(f"Error checking dimensions in {folder}: {e}")
                        break
                if folder_dim != max_dim:
                    break
            folder_dimensions[folder] = folder_dim

            # 加载episodes
            episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
            if not os.path.exists(episodes_path):
                print(f"Warning: Episodes file not found in {folder}, skipping")
                continue
            episodes = load_jsonl(episodes_path)

            # 加载episode stats
            episodes_stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
            episodes_stats = []
            if os.path.exists(episodes_stats_path):
                episodes_stats = load_jsonl(episodes_stats_path)

            # 创建映射 episode_index -> stats
            stats_map = {}
            for stat in episodes_stats:
                if "episode_index" in stat:
                    stats_map[stat["episode_index"]] = stat

            # 加载tasks并构建映射
            tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
            folder_tasks = []
            if os.path.exists(tasks_path):
                folder_tasks = load_jsonl(tasks_path)

            folder_task_mapping[folder] = {}
            for task in folder_tasks:
                task_desc = task["task"]
                old_index = task["task_index"]
                if task_desc not in task_desc_to_new_index:
                    new_index = len(all_unique_tasks)
                    task_desc_to_new_index[task_desc] = new_index
                    all_unique_tasks.append({"task_index": new_index, "task": task_desc})
                folder_task_mapping[folder][old_index] = task_desc_to_new_index[task_desc]

            # 处理episodes
            for episode in episodes:
                old_index = episode["episode_index"]
                # 新增：完整的episode处理与统计累积
                new_index = total_episodes
                episode["episode_index"] = new_index
                all_episodes.append(episode)

                # 更新对应的episode统计（如存在）
                if old_index in stats_map:
                    stats = stats_map[old_index]
                    stats["episode_index"] = new_index

                    # Pad stats data if needed
                    if "stats" in stats and folder_dimensions[folder] < max_dim:  # 使用变量替代硬编码的18
                        # Pad observation.state and action stats
                        for feature in ["observation.state", "action"]:
                            if feature in stats["stats"]:
                                for stat_type in ["mean", "std", "max", "min"]:
                                    if stat_type in stats["stats"][feature]:
                                        # Get current values
                                        values = stats["stats"][feature][stat_type]

                                        # Check if it's a list/array that needs padding
                                        if (
                                            isinstance(values, list) and len(values) < max_dim
                                        ):  # 使用变量替代硬编码的18
                                            # Pad with zeros
                                            padded = values + [0.0] * (
                                                max_dim - len(values)
                                            )  # 使用变量替代硬编码的18
                                            stats["stats"][feature][stat_type] = padded

                    all_episodes_stats.append(stats)

                    # Add to all_stats_data for proper merging
                    if "stats" in stats:
                        all_stats_data.append(stats["stats"])

                # Add to mapping
                episode_mapping.append((folder, old_index, new_index))

                # Update counters
                total_episodes += 1
                total_frames += episode["length"]

                # 处理每个episode时收集此信息
                episode_to_frame_index[new_index] = cumulative_frame_count
                cumulative_frame_count += episode["length"]

            # 使用收集的唯一任务列表替换之前的任务处理逻辑
            all_tasks = all_unique_tasks

        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")

    # Save combined episodes and stats
    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(all_episodes_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))
    save_jsonl(all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    # Merge and save stats
    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                stats_list.append(stats)

    if stats_list:
        # Merge global stats
        merged_stats = merge_stats(stats_list)

        # Update merged stats with episode-specific stats if available
        if all_stats_data:
            # For each feature in the stats
            for feature in merged_stats:
                if feature in all_stats_data[0]:
                    # Recalculate statistics based on all episodes
                    values = [stat[feature] for stat in all_stats_data if feature in stat]

                    # Find the maximum dimension for this feature
                    max_dim = max(
                        len(np.array(val.get("mean", [0])).flatten()) for val in values if "mean" in val
                    )

                    # Update count
                    if "count" in merged_stats[feature]:
                        merged_stats[feature]["count"] = [
                            sum(stat.get("count", [0])[0] for stat in values if "count" in stat)
                        ]

                    # Update min/max with padding
                    if "min" in merged_stats[feature] and all("min" in stat for stat in values):
                        # Pad min values
                        padded_mins = []
                        for val in values:
                            val_array = np.array(val["min"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_mins.append(padded)
                            else:
                                padded_mins.append(val_flat)
                        merged_stats[feature]["min"] = np.minimum.reduce(padded_mins).tolist()

                    if "max" in merged_stats[feature] and all("max" in stat for stat in values):
                        # Pad max values
                        padded_maxs = []
                        for val in values:
                            val_array = np.array(val["max"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_maxs.append(padded)
                            else:
                                padded_maxs.append(val_flat)
                        merged_stats[feature]["max"] = np.maximum.reduce(padded_maxs).tolist()

                    # Update mean and std (weighted by count if available)
                    if "mean" in merged_stats[feature] and all("mean" in stat for stat in values):
                        # Pad mean values
                        padded_means = []
                        for val in values:
                            val_array = np.array(val["mean"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_means.append(padded)
                            else:
                                padded_means.append(val_flat)

                        if all("count" in stat for stat in values):
                            counts = [stat["count"][0] for stat in values]
                            total_count = sum(counts)
                            weighted_means = [
                                mean * count / total_count
                                for mean, count in zip(padded_means, counts, strict=False)
                            ]
                            merged_stats[feature]["mean"] = np.sum(weighted_means, axis=0).tolist()
                        else:
                            merged_stats[feature]["mean"] = np.mean(padded_means, axis=0).tolist()

                    if "std" in merged_stats[feature] and all("std" in stat for stat in values):
                        # Pad std values
                        padded_stds = []
                        for val in values:
                            val_array = np.array(val["std"])
                            val_flat = val_array.flatten()
                            if len(val_flat) < max_dim:
                                padded = np.zeros(max_dim)
                                padded[: len(val_flat)] = val_flat
                                padded_stds.append(padded)
                            else:
                                padded_stds.append(val_flat)

                        if all("count" in stat for stat in values):
                            counts = [stat["count"][0] for stat in values]
                            total_count = sum(counts)
                            variances = [std**2 for std in padded_stds]
                            weighted_variances = [
                                var * count / total_count
                                for var, count in zip(variances, counts, strict=False)
                            ]
                            merged_stats[feature]["std"] = np.sqrt(
                                np.sum(weighted_variances, axis=0)
                            ).tolist()
                        else:
                            # Simple average of standard deviations
                            merged_stats[feature]["std"] = np.mean(padded_stds, axis=0).tolist()

        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    # Update and save info.json
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    # Update info with correct counts
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(all_tasks)
    info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info[
        "chunks_size"
    ]  # Ceiling division

    # Update splits
    info["splits"] = {"train": f"0:{total_episodes}"}

    # Update feature dimensions to the maximum dimension
    if "features" in info:
        # Find the maximum dimension across all folders
        actual_max_dim = max_dim  # 使用变量替代硬编码的18
        for _folder, dim in folder_dimensions.items():
            actual_max_dim = max(actual_max_dim, dim)

        # Update observation.state and action dimensions
        for feature_name in ["observation.state", "action"]:
            if feature_name in info["features"] and "shape" in info["features"][feature_name]:
                info["features"][feature_name]["shape"] = [actual_max_dim]
                print(f"Updated {feature_name} shape to {actual_max_dim}")

    # 更新视频总数 (Update total videos)
    info["total_videos"] = total_videos
    print(f"更新视频总数为: {total_videos} (Update total videos to: {total_videos})")

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Copy video and data files
    copy_videos(source_folders, output_folder, episode_mapping)
    copy_data_files(
        source_folders,
        output_folder,
        episode_mapping,
        max_dim=max_dim,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_task_mapping=folder_task_mapping,
        chunks_size=chunks_size,
    )

    print(f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}")
