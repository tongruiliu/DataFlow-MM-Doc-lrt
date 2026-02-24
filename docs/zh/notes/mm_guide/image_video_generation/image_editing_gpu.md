---
title: 图像编辑流水线（GPU版）
createTime: 2026/02/15 10:19:19
permalink: /zh/mm_guide/itxtxm62/
icon: material-symbols:draw-outline-rounded
---

# 图像编辑流水线（GPU版）

## 1. 概述

**图像编辑流水线**的核心功能是根据已有的图片和编辑指令生成对应的编辑后图片。使用方式非常简单，即输入图片文件和编辑指令（文本prompt），输出编辑后的图片。

本版本使用**本地GPU模型**进行图片编辑，支持在本地部署FLUX.1-Kontext-dev模型。

> 💡 **提示**：如果要使用云端API模型进行图片编辑，请查看 [图像编辑流水线（API版）](/zh/mm_guide/ffwfa9ck/)

---

## 2. 快速开始
### 第一步：创建新的 DataFlow 工作文件夹
```bash
mkdir run_dataflow_mm
cd run_dataflow_mm
```

### 第二步：配置模型路径

在流水线代码中配置模型路径。支持两种方式：

（1）方式1：使用Hugging Face模型路径（自动下载）
```python
hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev"
```

（2）方式2：使用本地模型路径（已下载的模型）
```python
hf_model_name_or_path="/path/to/your/local/FLUX.1-Kontext-dev"
```

在 `image_editing_pipeline.py` 中修改 `LocalImageGenServing` 的 `hf_model_name_or_path` 参数：

```python
self.serving = LocalImageGenServing(
    image_io=ImageIO(save_path=image_save_path),
    hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",  # 模型路径
    hf_cache_dir="./cache_local",  # Hugging Face模型缓存目录
    hf_local_dir="./ckpt/models/",  # 本地模型存储目录
    Image_gen_task="imageedit",  # 图像编辑任务类型
    batch_size=4,  # 批次大小
    diffuser_model_name="FLUX-Kontext",  # 扩散模型名称
    diffuser_num_inference_steps=28,  # 扩散模型推理步数，可调整以平衡速度和质量
    diffuser_guidance_scale=3.5,  # 引导尺度，控制生成结果与提示词的匹配程度
)
```

### 第三步：准备图片和文本数据
我们使用 `jsonl` 文件来保存图片和文本数据，每行一个样本。下面是一个简单的输入数据样例：
```jsonl
{"images": "image.png", "conversations": [{"role": "user", "content": "Change the vase to red."}]}
{"images": "human_inpaint.jpg", "conversations": [{"role": "user", "content": "Complete the shadowed part into a generic person's portrait."}]}
```

`images` 是待编辑的图片路径；`conversations` 包含图片编辑指令的对话列表，`content` 字段是其中的文本提示词。

### 第四步：运行流水线
```bash
python dataflow/statics/pipelines/gpu_pipelines/image_editing_pipeline.py \
    --first_entry_file_name <your_input_text_file_path>
```

生成的文件会默认保存在 `./cache_local/image_edit_local` 文件夹内。

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据包括以下字段：

* **images**：待编辑的图片路径，支持 `png` 、 `jpg` 等图片格式。
* **conversations**：对话格式数据，包含图片编辑指令。

这些输入数据存储在 `jsonl` 文件中，并通过 `FileStorage` 对象进行管理和读取：

```python
self.storage = FileStorage(
    first_entry_file_name="<your_jsonl_file_path>",
    cache_path="./cache_local/image_edit_local",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)
```

### 2. **图像编辑生成（PromptedImageEditGenerator）**

流程的核心步骤是使用**提示式图像编辑生成器**（`PromptedImageEditGenerator`）结合本地GPU模型根据输入图片和编辑指令生成编辑后的图像。

**功能：**

* 利用本地GPU模型（如FLUX.1-Kontext-dev）根据图片和编辑指令生成编辑后的图像
* 支持配置推理步数、引导尺度等参数
* 可调整批次大小以优化GPU利用率
* 自动保存生成的图像到指定路径

**输入**：图片文件路径和对话格式数据（包含编辑指令）  
**输出**：编辑后的图像文件路径

**本地GPU服务配置**：

```python
self.serving = LocalImageGenServing(
    image_io=ImageIO(save_path=image_save_path),  # 图像保存路径
    hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",  # 模型路径
    hf_cache_dir="./cache_local",  # Hugging Face模型缓存目录
    hf_local_dir="./ckpt/models/",  # 本地模型存储目录
    Image_gen_task="imageedit",  # 任务类型：图像编辑
    batch_size=4,  # 批次大小
    diffuser_model_name="FLUX-Kontext",  # 扩散模型名称
    diffuser_num_inference_steps=28,  # 扩散模型推理步数
    diffuser_guidance_scale=3.5,  # 引导尺度
)
```

**算子初始化**：

```python
self.image_edit_generator = PromptedImageEditGenerator(
    image_edit_serving=self.serving,  # 图像编辑服务
    save_interval=10  # 保存间隔
)
```

**算子运行**：

```python
self.image_edit_generator.run(
    storage=self.storage.step(),
    input_image_key="images",  # 输入图像字段
    input_conversation_key="conversations",  # 输入对话字段
    output_image_key="output_image",  # 输出图像字段
)
```

### 3. **输出数据**

最终，流水线生成的输出数据将包含以下内容：

* **images**：原始输入图片路径
* **conversations**：原始对话数据（包含编辑指令）
* **output_image**：编辑后的图像文件路径列表

**输出数据示例**：

```jsonl
{"images":"image.png","conversations":[{"role":"user","content":"Make the whole picture more vibrant."}],"output_image":["./cache_local/image_edit_local/sample_0/sample_0_0.png"]}
```

---

## 4. 流水线示例

以下给出使用本地FLUX模型的图片编辑流水线示例：
```python
import os
from pathlib import Path
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageEditingPipeline():
    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        
        data_file = project_root / "dataflow" / "example" / "image_gen" / "image_edit" / "prompts_local.jsonl"
        
        # -------- 存储配置 --------
        self.storage = FileStorage(
            first_entry_file_name=str(data_file),
            cache_path="./cache_local/image_edit_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        image_save_path = str(project_root / "cache_local" / "image_edit_local")
        
        # -------- 本地GPU图像编辑服务 --------
        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=image_save_path),
            hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",  # 或模型在本地的路径
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            Image_gen_task="imageedit",
            batch_size=4,
            diffuser_model_name="FLUX-Kontext",
            diffuser_num_inference_steps=28,
            diffuser_guidance_scale=3.5,
        )

        # -------- 图像编辑生成算子 --------
        self.image_edit_generator = PromptedImageEditGenerator(
            image_edit_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        # 调用 PromptedImageEditGenerator 生成编辑后的图像
        self.image_edit_generator.run(
            storage=self.storage.step(),
            input_image_key="images",
            input_conversation_key="conversations",
            output_image_key="output_image",
        )

if __name__ == "__main__":
    # -------- 流水线入口 --------
    model = ImageEditingPipeline()
    model.forward()
```