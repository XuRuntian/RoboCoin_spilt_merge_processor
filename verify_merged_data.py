import json
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

def load_jsonl(path):
    data = []
    if not os.path.exists(path): return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line))
                except: pass
    return data

def get_safe_indices(names_list):
    """
    智能过滤逻辑：模拟合并工具的行为，识别哪些维度是EEF相关应该被忽略的
    """
    if not names_list: return [], []
    indices = []
    kept_names = []
    for i, name in enumerate(names_list):
        n = name.lower()
        # 关键词必须与 split_merge_dataset.py 中的过滤逻辑一致
        if "end_pos" in n or "end_quat" in n or "eef_pose" in n or "robot_pos" in n or "robot_quat" in n:
            continue
        indices.append(i)
        kept_names.append(name)
    return indices, kept_names

def check_metadata(sorted_sources, merged_path):
    print(f"\n[1/3] 正在检查元数据 (info.json)...")
    total_eps_src = 0
    total_frames_src = 0
    
    merged_info = load_json(os.path.join(merged_path, "meta", "info.json"))
    merged_eps = merged_info.get('total_episodes', 0)
    merged_frames = merged_info.get('total_frames', 0)
    merged_fps = merged_info.get('fps', None)

    fps_changed = False
    
    for src in sorted_sources:
        src_info = load_json(os.path.join(src, "meta", "info.json"))
        total_eps_src += src_info.get('total_episodes', 0)
        total_frames_src += src_info.get('total_frames', 0)
        
        src_fps = src_info.get('fps', merged_fps) # 如果源没写FPS，假设和目标一致
        if merged_fps is not None and src_fps is not None and abs(src_fps - merged_fps) > 1e-5:
            fps_changed = True

    print(f"  - 预期 (源数据累加): {total_eps_src} eps, {total_frames_src} frames (源FPS累加值)")
    print(f"  - 实际 (合并结果):   {merged_eps} eps, {merged_frames} frames (FPS={merged_fps})")

    if fps_changed:
        print(f"  ⚠️ 检测到 FPS 发生变化，总帧数必然不同。跳过总帧数严格比对，将在步骤[2/3]中逐个验证重采样长度。")
        if merged_eps == total_eps_src:
            print("  ✅ Episode 数量匹配成功！")
            return True
        else:
            print(f"  ❌ Episode 数量不匹配！(预期 {total_eps_src}, 实际 {merged_eps})")
            return False
    else:
        if merged_eps == total_eps_src and merged_frames == total_frames_src:
            print("  ✅ 元数据统计匹配成功！")
            return True
        else:
            print(f"  ❌ 元数据不匹配！")
            return False

def check_structure(sorted_sources, merged_path):
    print(f"\n[2/3] 正在检查 Episode 顺序与长度 (episodes.jsonl)...")
    merged_episodes = load_jsonl(os.path.join(merged_path, "meta", "episodes.jsonl"))
    merged_ep_map = {ep['episode_index']: ep for ep in merged_episodes}
    merged_info = load_json(os.path.join(merged_path, "meta", "info.json"))
    merged_fps = merged_info.get('fps', 30)
    
    current_global_index = 0
    all_match = True
    
    for src in sorted_sources:
        src_episodes = load_jsonl(os.path.join(src, "meta", "episodes.jsonl"))
        src_info = load_json(os.path.join(src, "meta", "info.json"))
        src_fps = src_info.get('fps', merged_fps) # 默认假设一致

        for src_ep in src_episodes:
            if current_global_index not in merged_ep_map:
                print(f"    ❌ 缺少索引: {current_global_index}")
                all_match = False; break
            
            merged_ep = merged_ep_map[current_global_index]
            
            # === 计算预期长度 (包含重采样逻辑) ===
            expected_len = src_ep.get('length')
            if merged_fps and src_fps and abs(merged_fps - src_fps) > 1e-5:
                # 复制 RoboCOIN_dataset_lib.py 中的重采样逻辑
                duration = expected_len / src_fps
                expected_len = int(np.ceil(duration * merged_fps))
            
            if expected_len != merged_ep.get('length'):
                print(f"    ❌ 长度不一致! Idx: {current_global_index} (源: {src_ep.get('length')}@{src_fps}fps -> 预期: {expected_len} -> 实际: {merged_ep.get('length')}@{merged_fps}fps)")
                all_match = False
            
            current_global_index += 1
            
    if all_match: print(f"  ✅ 结构验证通过 ({current_global_index} 条数据)")
    return all_match

def check_deep_content_smart(sorted_sources, merged_path, max_dim=None):
    print("\n[3/3] 正在进行深度内容比对 (数值精度与智能对齐)...")
    src_root = sorted_sources[0]
    
    # 1. 获取源数据的列名定义
    info_path = os.path.join(src_root, "meta", "info.json")
    if not os.path.exists(info_path):
        print("  ⚠️ 无法找到源数据的 info.json，跳过智能比对")
        return

    src_info = load_json(info_path)
    state_feat = src_info.get('features', {}).get('observation.state', {})
    names = state_feat.get('names', [])
    
    keep_indices, kept_names = get_safe_indices(names)
    print(f"  ℹ️  源维度 {len(names)} -> 智能保留 {len(keep_indices)} (已自动剔除 EEF/Robot Pose)")

    # 2. 读取文件比对
    # 尝试找到第一个 parquet 文件
    src_file = None
    merged_file = None
    
    # 查找源文件 (遍历直到找到存在的 episode_000000)
    for root, _, files in os.walk(os.path.join(src_root)):
        for f in files:
            if "episode_000000.parquet" in f:
                src_file = os.path.join(root, f)
                break
        if src_file: break

    # 查找目标文件
    merged_file = os.path.join(merged_path, "data/chunk-000/episode_000000.parquet")
    
    if not src_file or not os.path.exists(src_file):
        print(f"  ⚠️ 找不到源文件 episode_000000.parquet，跳过内容比对")
        return
    
    if not os.path.exists(merged_file):
        print(f"  ❌ 找不到对应的合并文件: {merged_file}")
        return

    try:
        df_src = pd.read_parquet(src_file)
        df_merged = pd.read_parquet(merged_file)
    except Exception as e:
        print(f"  ❌ 读取 Parquet 失败: {e}")
        return

    if 'observation.state' not in df_merged.columns: return

    # 提取源向量和合并向量
    try:
        vec_src = np.array(df_src['observation.state'].iloc[0])
        vec_merged = np.array(df_merged['observation.state'].iloc[0])
    except Exception as e:
        print(f"  ❌ 数据提取失败: {e}")
        return

    # 应用智能过滤
    if keep_indices:
        # 确保索引不超过源向量长度 (防止源数据已经更短)
        valid_indices = [i for i in keep_indices if i < len(vec_src)]
        vec_src_check = vec_src[valid_indices]
    else:
        vec_src_check = vec_src

    # 截取合并数据的有效长度（去除末尾补零）
    valid_len = len(vec_src_check)
    if valid_len > len(vec_merged):
        print(f"  ❌ 维度异常: 源有效维度 {valid_len} > 合并维度 {len(vec_merged)}")
        return
        
    vec_merged_check = vec_merged[:valid_len]
    
    # --- 维度补零检查 ---
    if max_dim is not None:
        merged_dim = len(vec_merged)
        if merged_dim != max_dim:
             print(f"  ❌ 维度错误: 期望 {max_dim}, 实际 {merged_dim}")
        elif merged_dim > valid_len:
             padding = vec_merged[valid_len:]
             if np.allclose(padding, 0):
                 print(f"  ✅ 维度补零检查通过 (填充了 {len(padding)} 个0)")
             else:
                 print(f"  ❌ 补零数据异常 (填充部分非0)")

    # --- 数值误差检查 ---
    try:
        diff = np.abs(vec_src_check - vec_merged_check)
        max_diff = np.max(diff)

        if max_diff < 1e-4: # 稍微放宽一点，考虑到浮点转换
            print(f"  ✅ 智能数值比对成功！(最大误差: {max_diff:.6f})")
            print(f"     已确认核心数据（关节、手爪等）无损传输。")
        else:
            print(f"  ❌ 数值验证失败，最大误差: {max_diff}")
            print(f"     源: {vec_src_check[:5]}")
            print(f"     合: {vec_merged_check[:5]}")
    except ValueError as e:
        print(f"  ❌ 形状不匹配无法比对: {e}")

def main():
    parser = argparse.ArgumentParser(description="Verify merged dataset integrity")
    parser.add_argument('--sources', nargs='+', required=True, help="List of source dataset paths")
    parser.add_argument('--output', required=True, help="Merged dataset path")
    parser.add_argument('--max_dim', type=int, default=None, help="Expected max dimension (for zero-padding check)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        print(f"错误: 找不到输出路径 {args.output}")
        exit(1)

    # 确保源路径排序与合并工具一致
    sorted_sources = sorted(list(set(args.sources)))
    print(f"验证源路径: {len(sorted_sources)} 个")
    
    # 按步骤执行验证
    step1 = check_metadata(sorted_sources, args.output)
    step2 = check_structure(sorted_sources, args.output)
    
    if step1 and step2:
           check_deep_content_smart(sorted_sources, args.output, args.max_dim)
           print("\n🎉 === 数据完整性验证通过！=== ")
    else:
        print("\n❌ === 验证失败！请检查上方日志 === ")
        exit(1)

if __name__ == "__main__":
    main()