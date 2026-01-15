# RoboCOIN Split & Merge Processor

一个用于机器人数据集处理的命令行工具集，主要包含两个功能模块：
1.  **Split & Merge**：数据集的拆分、合并、维度对齐、视频统一缩放及字段筛选。
2.  **Auto Rename**：自动化统一相机名称，修复数据集命名不规范问题。

## 项目结构
- `RoboCOIN_dataset_lib.py`：核心函数库（提供底层逻辑：Parquet 读写、视频 FFmpeg 处理、统计对齐等）。
- `split_merge_dataset.py`：主程序入口，用于数据集的**拆分**与**合并**。
- `auto_rename_high_camera.py`：独立工具，用于**批量修复/重命名**相机名称。

---

## 1. 安装与环境

### Python 依赖
推荐使用 Python 3.10 环境：
```bash
pip install -U pandas numpy pyarrow fastparquet opencv-pytho
```
### 系统依赖 (必须)
由于工具集使用 FFmpeg 进行视频的缩放与黑边填充（Letterbox），请确保系统已安装 FFmpeg：
- Ubuntu/Debian:
```bash
sudo apt-get install ffmpeg
```
- MacOS:
```bash
brew install ffmpeg
```

## 2. 数据集拆分与合并
### 基本用法
```bash
python split_merge_dataset.py [command] [args...]
```
支持两个子命令：split（拆分）和 merge（合并）。

通用特性
- 维度对齐：自动将 observation.state 和 action 零填充到 --max_dim 指定的维度。

- 视频处理：支持通过 --video_width 和 --video_height 将视频统一缩放并填充黑边（Letterbox），确保分辨率一致。

- 元数据修复：自动重新计算并合并 episodes_stats.jsonl、tasks.jsonl 和全局 stats.json。

---

### A. 拆分模式
从单个数据集中截取部分数据（按帧数或 episode 数）。
#### 参数说明
- --input：输入数据集路径（必填）。

- --output：输出数据集路径（必填）。

- --max_episodes：截取前 N 个 episode。

- --max_entries：截取前 N 帧（优先级高于 episode）。

- --start_episodes：跳过前 N 个 episode。

- --start_entries：跳过前 N 帧。

- --video_width / --video_height：(可选) 输出视频的目标分辨率，会自动进行缩放和填充黑边。

```bash
# 从第 2 个 episode 开始，截取 300 个 episode，并将视频统一为 1280x720
python split_merge_dataset.py split \
  --input /path/to/source_dataset \
  --output /path/to/output_dataset \
  --start_episodes 2 \
  --max_episodes 300 \
  --video_width 1280 \
  --video_height 720
```

### B. 合并模式 (merge)
将多个数据集合并为一个，支持自动移除末端位姿（EEF）以保证跨机器人兼容性。

#### 参数说明
- --sources：指定多个源数据集路径列表。

- --sources_dir：指定父目录，自动扫描一、二级子目录下的数据集。

- --output：输出数据集路径（必填）。

- --max_dim：目标状态/动作维度（例如 32），不足会自动补零。

- --fps：强制指定输出的帧率（默认 20）。

- --features：(可选) 字段筛选，指定仅保留哪些字段（如只保留图像和状态，丢弃多余数据）。

- --video_width / --video_height：(可选) 统一所有源视频的分辨率。
注意： 合并模式默认会移除末端位姿数据 (End Effector Pose)，仅保留关节角度 (Joints) 和夹爪状态，以解决不同机械臂末端定义不一致的问题。

#### 示例 1：指定多个源路径合并
``` bash
python split_merge_dataset.py merge \
  --sources /data/ds1 /data/ds2 /data/ds3 \
  --output /data/merged_ds \
  --max_episodes 1000 \
  --fps 30 \
  --max_dim 32
```

#### 示例 2：自动扫描目录 + 筛选字段 + 视频统一

```bash
python split_merge_dataset.py merge \
  --sources_dir /mnt/nas/raw_datasets \
  --output /mnt/nas/clean_merged_dataset \
  --max_dim 14 \
  --video_width 640 \
  --video_height 480 \
  --features observation.state action observation.images.cam_high_rgb
```

## 3. 数据集拆分与合并

用于批量修复旧数据集中相机命名不规范的问题（例如将 head_camera, front_cam 等统一重命名为标准名称）。
功能：

1. 扫描 `info.json` 和 `episodes_stats.jsonl`。

2. 识别包含 `head`, `front`, `font` 等关键词的相机字段。

3. 将其重命名为标准名称：`observation.images.cam_high_rgb`。

4. 自动重命名 `videos/` 下对应的文件夹，并移动文件。

5. 安全机制：只重命名，不删除任何数据。

用法：
```bash
# 试运行（不修改文件，仅打印预览）
python auto_rename_high_camera.py --input /path/to/dataset_root --dry-run

# 执行修改
python auto_rename_high_camera.py --input /path/to/dataset_root
```
输入路径支持：

- 可以是单个数据集文件夹。

- 也可以是包含多个数据集的根目录（会自动批量扫描处理）。

## 输出目录结构

生成的 LeRobot 格式数据集结构如下：

```Plaintext
/path/to/output
├── meta/
│   ├── info.json              # 汇总信息 (total_frames, splits, features 等)
│   ├── episodes.jsonl         # Episode 列表 (重新编号)
│   ├── episodes_stats.jsonl   # 逐 Episode 统计 (已对齐维度)
│   ├── tasks.jsonl            # 任务描述 (去重并重新映射 ID)
│   └── stats.json             # 全局统计 (均值/方差/极值)
├── annotations/               # 统一的注释定义表 (按描述去重)
│   ├── subtask_annotations.jsonl
│   ├── scene_annotations.jsonl
│   └── ...
├── videos/                    # 视频文件 (可选项：已缩放/填充)
│   └── chunk-000/
│       └── observation.images.xxx/episode_000000.mp4
└── data/                      # Parquet 数据 (已对齐维度与索引)
    └── chunk-000/
        └── episode_000000.parquet
```

## 常见问题

1. 视频处理失败：

- 请检查是否安装了 ffmpeg。

- 如果源视频损坏，工具会尝试直接复制原文件作为回退。

2. 维度不一致：

- 使用 --max_dim 参数（推荐 32 或更高），工具会自动对短向量进行零填充。

3. 合并后数据缺少末端位姿 (EEF)：

- 这是设计使然。为了兼容性，merge 模式默认只保留关节数据。如需修改，请调整 split_merge_dataset.py 中的 remove_eef=True 参数。