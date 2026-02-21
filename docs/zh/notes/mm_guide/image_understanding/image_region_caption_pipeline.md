---
title: 图像区域描述生成流水线RegionCap
createTime: 2026/01/11 22:04:27
icon: mdi:image-text
permalink: /zh/mm_guide/image_region_caption_pipeline/
---
## 1. 概述

**图像区域描述生成流水线 (Image Region Caption Pipeline)** 旨在为图像中的特定区域生成详细的文本描述。该流水线结合了计算机视觉的定位能力与多模态大模型的理解能力，能够识别图像中的感兴趣区域（ROI），并为其生成精确的自然语言标注。

该流水线支持处理**预定义边界框 (Bounding Box)** 数据，并将其可视化后输入 VLM 进行描述生成。

我们支持以下应用场景：

* **密集描述生成 (Dense Captioning)**：为图像中的多个物体分别生成描述。
* **细粒度图像理解**：关注图像的局部细节而非全局描述。
* **数据集增强**：构建带定位信息的图文对数据集。

流水线的主要流程包括：

1. **数据加载**：读取包含图像和边界框信息的源数据。
2. **边界框处理与可视化**：处理输入的边界框，生成带有可视化标记（如画框）的图像版本。
3. **区域描述生成**：利用 VLM 针对标记后的图像或特定区域生成文本描述。

---

## 2. 快速开始

### 第一步：创建新的 DataFlow 工作文件夹
```bash
mkdir run_dataflow
cd run_dataflow
```

### 第二步：初始化 DataFlow-MM
```bash
dataflowmm init
```
这时你会看到：
```bash
gpu_pipelines/image_region_caption_pipeline.py
```

### 第三步：下载示例数据
```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir data
```

### 第四步：一键运行

```bash
python gpu_pipelines/image_region_caption_pipeline.py

```

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

输入数据通常包含图像路径和对应的边界框列表（可选）：

* **image**：图像文件路径。
* **bbox**：边界框坐标列表，通常格式为 `[[x, y, w, h], ...]`。

**输入数据示例**：

```json
{
    "image": "./data/image_region_caption/20.jpg",
    "bbox": [[196, 104, 310, 495], [50, 60, 100, 200]]
}

```

### 2. **核心算子逻辑**

该流水线通过串联两个核心算子来完成任务：

#### A. **ImageBboxGenerator（边界框处理器）**

该算子负责处理视觉层面的任务。

* **输入**：原始图像 + `bbox` 数据。
* **功能**：读取边界框，将其绘制在图像上（可视化），或者根据配置进行预处理。
* **配置 (`ExistingBBoxDataGenConfig`)**：控制最大框数量 (`max_boxes`)和输入输出路径。
* **输出**：带有视觉标记的新图像的json文件输出路径。

#### B. **PromptedVQAGenerator（VQA 生成器）**

该算子负责利用 VLM 生成文本。

* **输入**：上一步的输出。
* **功能**：VLM 接收带有标记的图像，根据提示生成对应区域的描述。
* **输出**：区域描述文本。

### 3. **输出数据**

最终生成的输出数据将包含处理后的图像路径和生成的描述：
* **image**：输入的图片路径。
* **type**：是否给定边界框。
* **bbox**：边界框参数。
* **normalized_bbox**：标准化后的边界框参数。
* **result_file**：结果输出路径。
* **image_with_bbox**：画了框的图像路径。
* **valid_bboxes_num**：有效边界框数量。
* **prompt**：VLM接收的提示词。
* **answer**：生成的区域描述列表。

**输出数据示例**：

```json
{
    "image":".\/data\/image_region_caption\/20.png","type":"with_bbox",
    "bbox":[[196,104,310,495]],
    "normalized_bbox":[[0.128,0.125,0.329,0.72],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],
    "result_file":".\/cache\/image_region_caption","image_with_bbox":".\/cache\/image_region_caption\\2_bbox_vis.jpg",
    "valid_bboxes_num":1,
    "prompt":"Describe the content of each marked region in the image. There are 1 regions: <region1> to <region1>.",
    "answer":"In <region1>, the focus is on the lower half of a person wearing high-heeled shoes with an ornate design. The setting appears to be a kitchen, with items such as a table with floral tablecloth, a broom, and various kitchen utensils visible in the background. The legs of another person can also be seen, indicating there may be interaction happening in this domestic space. The overall scene captures a domestic and casual atmosphere."
}

```

---

## 4. 流水线示例

以下是完整的 `ImageRegionCaptionPipeline` 代码实现。

```python
import argparse
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm
from dataflow.operators.core_vision.generate.image_bbox_generator import (
    ImageBboxGenerator, 
    ExistingBBoxDataGenConfig
)
from dataflow.operators.core_vision.generate.prompted_vqa_generator import (
    PromptedVQAGenerator
)
from dataflow.utils.storage import FileStorage


class ImageRegionCaptionPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        first_entry_file: str = "./data/image_region_caption/image_region_caption_demo.jsonl",
        cache_path: str = "./cache/image_region_caption",
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_bbox_key: str = "bbox",
        image_with_bbox_path: str = 'image_with_bbox',
        max_boxes: int = 10,
        output_image_with_bbox_path: str = "./cache/image_region_caption/image_with_bbox_result.jsonl",
    ):
        self.bbox_storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )

        self.cfg = ExistingBBoxDataGenConfig(
            max_boxes=max_boxes,
            input_jsonl_path=first_entry_file,
            output_jsonl_path=output_image_with_bbox_path,
        )

        self.caption_storage = FileStorage(
            first_entry_file_name=output_image_with_bbox_path,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type
        )
        self.serving = LocalModelVLMServing_vllm(
            hf_model_name_or_path=model_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.7,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
        )
        self.bbox_generator = ImageBboxGenerator(config=self.cfg)
        self.caption_generator = PromptedVQAGenerator(serving=self.serving,)
        self.input_image_key = input_image_key
        self.input_bbox_key = input_bbox_key
        self.image_with_bbox_path=image_with_bbox_path
        self.bbox_record=None

    def forward(self):
        self.bbox_generator.run(
            storage=self.bbox_storage.step(),
            input_image_key=self.input_image_key,
            input_bbox_key=self.input_bbox_key,
        )

        self.caption_generator.run(
            storage=self.caption_storage.step(),
            input_image_key='image_with_bbox',
            input_prompt_key='prompt'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image region caption with DataFlow")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--hf_cache_dir", default="~/.cache/huggingface")
    parser.add_argument("--download_dir", default="./ckpt/models")
    parser.add_argument("--first_entry_file", default="./data/image_region_caption/image_region_caption_demo.jsonl")
    parser.add_argument("--cache_path", default="./cache/image_region_caption")
    parser.add_argument("--file_name_prefix", default="region_caption")
    parser.add_argument("--cache_type", default="jsonl")
    parser.add_argument("--input_image_key", default="image")
    parser.add_argument("--input_bbox_key", default="bbox")
    parser.add_argument("--max_boxes", type=int, default=10)
    parser.add_argument("--output_image_with_bbox_path", default="./cache/image_region_caption/image_with_bbox_result.jsonl")

    args = parser.parse_args()

    pipe = ImageRegionCaptionPipeline(
        model_path=args.model_path,
        hf_cache_dir=args.hf_cache_dir,
        download_dir=args.download_dir,
        first_entry_file=args.first_entry_file,
        cache_path=args.cache_path,
        file_name_prefix=args.file_name_prefix,
        cache_type=args.cache_type,
        input_image_key=args.input_image_key,
        input_bbox_key=args.input_bbox_key,
        max_boxes=args.max_boxes,
        output_image_with_bbox_path=args.output_image_with_bbox_path
    )
    pipe.forward()

```

