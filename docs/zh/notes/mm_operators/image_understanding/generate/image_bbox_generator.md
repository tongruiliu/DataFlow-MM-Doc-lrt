---
title: ImageBboxGenerator
createTime: 2026/01/11 21:49:37
permalink: /zh/mm_operators/generate/image_bbox_generator/
---
## 📘 概述

`ImageBboxGenerator` 是一个 **图像区域标注与提示词准备算子**。

该算子主要用于多模态任务（如 Grounding Caption）的数据预处理。它能够处理包含图像路径的原始数据，对感兴趣区域（Region of Interest）进行标准化处理和可视化绘制，并生成用于后续 VLM 推理的 Prompt。

核心能力：

1. **双模式 BBox 获取**：
* **已有框模式**：直接读取输入数据中已有的 BBox 坐标。
* **自动提取模式**：若无 BBox，利用 OpenCV（边缘检测 + 轮廓拟合）自动提取图像中的显著物体区域。


2. **坐标标准化**：将像素坐标转换为符合 VLM 输入规范的归一化坐标（0-1 或 0-1000）。
3. **可视化增强**：生成带有数字编号和彩色边框的可视化图像，辅助模型理解“第 N 个区域”的指代关系。
4. **Prompt 构造**：自动生成包含区域数量信息的 Prompt（如 "Describe the content of each marked region..."）。

## 🏗️ `__init__` 函数

```python
def __init__(self, config: Optional[ExistingBBoxDataGenConfig] = None):
    ...

```

### 🧾 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `config` | `ExistingBBoxDataGenConfig` | `None` | 配置对象，定义了输入输出路径及最大框数量限制。 |

#### `ExistingBBoxDataGenConfig` 配置详解

| 字段名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `max_boxes` | `int` | `10` | 单张图像保留的最大 BBox 数量（按面积排序）。不足部分补零。 |
| `input_jsonl_path` | `str` | `None` | **必须指定**。输入 JSONL 文件路径。 |
| `output_jsonl_path` | `str` | `None` | **必须指定**。处理结果的保存路径。 |

## ⚡ `run` 函数

```python
def run(
    self, 
    storage: DataFlowStorage, 
    input_image_key: str = "image", 
    input_bbox_key: str = "bbox"
):
    ...

```

执行算子主逻辑：

1. **数据读取**
从 `config.input_jsonl_path` 读取原始数据。
2. **BBox 获取 (Extract/Get)**
* 检查每行数据是否包含 `input_bbox_key`。
* **Type A (With BBox)**: 直接使用数据中的坐标。
* **Type B (Without BBox)**: 调用 `extract_boxes_from_image`，通过自适应阈值和形态学操作提取物体轮廓，并应用 NMS (非极大值抑制) 去重。


3. **标准化与可视化 (Normalize & Visualize)**
*
* **标准化**：将 `[x, y, w, h]` 转换为归一化的 `[x1, y1, x2, y2]` 格式，并根据 `max_boxes` 进行截断或补零 (`0.0, 0.0, 0.0, 0.0`)。
* **可视化**：在原图上绘制矩形框和数字标签，保存至 `storage.cache_path`。


4. **Prompt 生成**
* 根据有效框的数量，生成固定模板的 Prompt：
> "Describe the content of each marked region in the image. There are {N} regions: \<region1\> to \<regionN\>."


5. **结果导出**
* 将包含原始信息、标准化 BBox、可视化路径及 Prompt 的完整记录写入 `config.output_jsonl_path`。



### 🧾 `run` 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `storage` | `DataFlowStorage` | 无 | DataFlow 存储对象，主要用于获取缓存路径 (`cache_path`)。 |
| `input_image_key` | `str` | `"image"` | 输入 JSONL 中图像路径的字段名。 |
| `input_bbox_key` | `str` | `"bbox"` | 输入 JSONL 中 BBox 数据的字段名。 |

## 🧩 示例用法

```python
from dataflow.utils.storage import FileStorage
from dataflow.operators.cv import ImageBboxGenerator, ExistingBBoxDataGenConfig

cfg = ExistingBBoxDataGenConfig(
    max_boxes=10,
    input_jsonl_path="./data/image_region_caption/image_region_caption_demo.jsonl",
    output_jsonl_path="./cache/image_region_caption/image_with_bbox_result.jsonl",
)
generator = ImageBboxGenerator(config=config)

storage = FileStorage(
    first_entry_file_name="./data/image_region_caption/image_region_caption_demo.jsonl",
    cache_path="./cache/image_region_caption",
    file_name_prefix="region_caption",
    cache_type="jsonl"
)

generator.run(
    storage=storage,
    input_image_key="image",
    input_bbox_key="bbox"
)

```

### 🧾 输出数据格式 (Output JSONL)

生成的 `image_with_bbox_result.jsonl` 文件中，每一行包含以下结构：

```json
{
    "image": "./data/image_region_caption/20.png", 
    "type": "with_bbox", 
    "bbox": [[196, 104, 310, 495]], 
    "normalized_bbox": [[0.128, 0.125, 0.329, 0.72], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 
    "result_file": "./cache/image_region_caption", 
    "image_with_bbox": "./cache/image_region_caption\\2_bbox_vis.jpg", 
    "valid_bboxes_num": 1, 
    "prompt": "Describe the content of each marked region in the image. There are 1 regions: <region1> to <region1>."
}
```
