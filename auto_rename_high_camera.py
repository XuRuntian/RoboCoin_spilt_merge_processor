import os
import json
import argparse
import shutil
from pathlib import Path

# === 配置区域 ===
# 目标标准名称 (LeRobot v3.0 默认期望的名称 - High Camera)
TARGET_HIGH_NAME = "observation.images.cam_high_rgb"

# 触发关键词 (只要现有相机名包含这些词，且不叫 cam_high_rgb，就会被重命名)
# 注意：这主要用于识别名字乱七八糟的主视角相机
KEYWORDS = ["head", "front", "font"] 

# =================

def update_jsonl_stats(file_path, info_rename_map, dry_run=False):
    """
    功能：逐行修复 episodes_stats.jsonl
    逻辑：1. 只重命名  2. 保留所有数据 (不再删除多余相机)
    """
    if not file_path.exists():
        print(f"    ⚠️ 未找到统计文件: {file_path.name}")
        return

    print(f"    🔍 正在扫描并重命名统计文件: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"      共读取 {len(lines)} 行数据")

    new_lines = []
    file_modified = False
    modified_count = 0

    for line in lines:
        line = line.strip()
        if not line: continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
            
        stats = data.get("stats", {})
        if not stats:
            new_lines.append(line)
            continue

        new_stats = {}
        row_modified = False
        
        # 遍历这一行的所有统计项
        for key, value in stats.items():
            current_key = key
            
            # --- 步骤 1: 重命名逻辑 ---
            # A. 优先使用 info.json 的映射 (这里包含了 _rgb_rgb 的修复映射)
            if current_key in info_rename_map:
                current_key = info_rename_map[current_key]
            
            # B. 其次检查关键词 (针对未在 info 中定义的漏网之鱼，防止万一 stats 里有 info 里没有的 key)
            elif current_key != TARGET_HIGH_NAME and current_key.startswith("observation.images."):
                # 额外修复：如果 stats 里的 key 也有 _rgb_rgb 错误
                if current_key.endswith("_rgb_rgb"):
                     current_key = current_key.replace("_rgb_rgb", "_rgb")
                     row_modified = True
                else:
                    is_bad_name = any(kw in current_key.lower() for kw in KEYWORDS)
                    if is_bad_name:
                        current_key = TARGET_HIGH_NAME
            
            # 检测是否发生了改名
            if current_key != key:
                row_modified = True

            # --- 步骤 2: 直接赋值 (不再进行白名单过滤) ---
            new_stats[current_key] = value
        
        if row_modified:
            data["stats"] = new_stats
            # 使用 separators 生成紧凑的 JSON
            new_lines.append(json.dumps(data, separators=(',', ':'))) 
            file_modified = True
            modified_count += 1
        else:
            new_lines.append(line)

    # 写入文件
    if file_modified:
        if dry_run:
            print(f"    [Dry Run] 拟更新 {modified_count} 行 (仅重命名)")
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in new_lines:
                    f.write(line + '\n')
            print(f"    📝 {file_path.name} 已修复 (更新了 {modified_count} 行)")
    else:
        print(f"    ✅ {file_path.name} 内容无需修改")

def process_single_dataset(dataset_path, dry_run=False):
    """
    核心逻辑：处理单个数据集 (Info Rename + Stats Rename + Video Rename)
    """
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta/info.json"
    stats_jsonl_path = dataset_path / "meta/episodes_stats.jsonl"
    
    if not info_path.exists():
        return False, f"跳过 (无 meta/info.json): {dataset_path.name}"

    print(f"\n>>> 正在扫描: {dataset_path.name}")

    # --- 1. 读取 info.json ---
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except json.JSONDecodeError:
        return False, f"JSON 解析失败: {info_path}"

    features = info.get("features", {})
    rename_map = {}
    
    # --- 2. 构建重命名映射 (基于 info.json) ---
    for key in features.keys():
        if not key.startswith("observation.images."): continue

        # === 新增逻辑：优先修复双重后缀错误 ===
        # 检查是否以 _rgb_rgb 结尾
        if key.endswith("_rgb_rgb"):
            # 去掉最后4个字符 (_rgb)
            corrected_name = key[:-4] 
            print(f"    🛠️ 发现双重后缀错误: '{key}' -> 修正为 '{corrected_name}'")
            rename_map[key] = corrected_name
            # 如果匹配了这个错误，直接进入下一次循环，不再进行后续关键词检查
            continue 
        # =================================

        if key == TARGET_HIGH_NAME: continue

        #原本的关键词逻辑 (处理 head/front 等不规范命名)
        lower_key = key.lower()
        for kw in KEYWORDS:
            if kw in lower_key:
                print(f"    🎯 info.json 发现旧标准名: '{key}' -> 标记为 '{TARGET_HIGH_NAME}'")
                rename_map[key] = TARGET_HIGH_NAME
                break

    # --- 3. 执行修改 (info.json) ---
    # 仅重命名，不再删除未在白名单的 features
    info_modified = False
    new_features = {}
    
    for key, value in features.items():
        # 3.1 获取最终名称 (如果有映射就用新的，没有就用旧的)
        final_key = rename_map.get(key, key)
        
        # 3.2 直接保留 (不做过滤)
        new_features[final_key] = value

        if final_key != key:
            info_modified = True

    # 如果有 info.json 全局 stats，也只重命名不清理
    if "stats" in info:
        new_global_stats = {}
        for key, value in info["stats"].items():
            final_key = rename_map.get(key, key)
            new_global_stats[final_key] = value
            if final_key != key:
                info_modified = True # 标记需要保存
        info["stats"] = new_global_stats

    info["features"] = new_features

    # 保存 info.json
    if info_modified:
        if dry_run:
            print(f"    [Dry Run] 拟更新 info.json (仅重命名 key)")
        else:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4)
            print("    📝 info.json 已更新 (重命名完成)")
    else:
        print("    ✅ info.json 无需修改")

    # --- 4. 执行修改 (episodes_stats.jsonl) --- 
    # 这里不再过滤，只改名
    update_jsonl_stats(stats_jsonl_path, rename_map, dry_run)

    # --- 5. 执行修改 (视频文件夹) ---
    if rename_map:
        videos_root = dataset_path / "videos"
        if videos_root.exists():
            for chunk_dir in videos_root.iterdir():
                if not chunk_dir.is_dir(): continue
                
                for old_name, new_name in rename_map.items():
                    old_video_dir = chunk_dir / old_name
                    new_video_dir = chunk_dir / new_name
                    
                    if old_video_dir.exists():
                        if dry_run:
                            print(f"    [Dry Run] 拟重命名文件夹: {old_name} -> {new_name}")
                            continue

                        if new_video_dir.exists():
                            # 如果目标文件夹已存在 (极端情况)，把文件移过去
                            print(f"    ⚠️ 目标文件夹已存在，正在合并: {new_video_dir}")
                            for item in old_video_dir.iterdir():
                                try:
                                    target_file = new_video_dir / item.name
                                    if not target_file.exists():
                                        shutil.move(str(item), str(target_file))
                                except Exception as e:
                                    print(f"    ❌ 移动文件失败: {e}")
                            try:
                                old_video_dir.rmdir()
                            except:
                                pass
                        else:
                            try:
                                old_video_dir.rename(new_video_dir)
                                print(f"    ✨ 文件夹重命名成功: {old_name} -> {new_name}")
                            except OSError as e:
                                print(f"    ❌ 重命名失败: {e}")

    return True, "Success"

def auto_detect_and_run(input_path, dry_run=False):
    root = Path(input_path)
    if not root.exists():
        print(f"❌ 路径不存在: {input_path}")
        return

    if (root / "meta" / "info.json").exists():
        print(f"🤖 模式: 单数据集处理")
        process_single_dataset(root, dry_run)
    else:
        print(f"🤖 模式: 批量根目录扫描")
        subdirs = [x for x in root.iterdir() if x.is_dir()]
        count = 0
        for subdir in subdirs:
            if (subdir / "meta" / "info.json").exists():
                process_single_dataset(subdir, dry_run)
                count += 1
        print(f"\n🎉 处理完成，共扫描有效数据集: {count} 个")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动修复相机名称 (修复 _rgb_rgb 及规范化)")
    parser.add_argument("--input", required=True, help="数据集路径")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式")
    
    args = parser.parse_args()
    
    print("🚀 开始执行：修复相机名称...")
    auto_detect_and_run(args.input, args.dry_run)