---
title: 图像生成流水线（GPU版）
createTime: 2026/02/15 10:14:49
permalink: /zh/mm_guide/513sulfz/
icon: material-symbols:photo-size-select-large
---

# 图像生成流水线（GPU版）

## 1. 概述

**图像生成流水线**的核心功能是根据用户提供的文本生成目标图片，为后续图像理解、图像编辑等任务提供图片数据。

本版本使用**本地GPU模型**进行文本到图片生成，支持在本地部署FLUX.1-dev等模型。

> 💡 **提示**：如果要使用云端API模型进行文本到图片生成，请查看 [图像生成流水线（API版）](/zh/mm_guide/image_generation_api/)

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
hf_model_name_or_path="black-forest-labs/FLUX.1-dev"
```

（2）方式2：使用本地模型路径（已下载的模型）
```python
hf_model_name_or_path="/path/to/your/local/FLUX.1-dev"
```

在 `text_to_image_generation_pipeline.py` 中修改 `LocalImageGenServing` 的 `hf_model_name_or_path` 参数：

```python
self.serving = LocalImageGenServing(
    image_io=ImageIO(save_path=image_save_path),
    batch_size=4,
    hf_model_name_or_path="black-forest-labs/FLUX.1-dev",  # 模型路径
    hf_cache_dir="./cache_local",  # Hugging Face模型缓存目录
    hf_local_dir="./ckpt/models/",  # 本地模型存储目录
    diffuser_num_inference_steps=20,  # 扩散模型推理步数，可调整以平衡速度和质量
    diffuser_image_height=512,  # 生成图片的高度
    diffuser_image_width=512,  # 生成图片的宽度
)
```

### 第三步：准备文本数据
我们使用 `jsonl` 文件来保存文本数据，每行一个样本。下面是一个简单的输入数据样例：
```jsonl
{"conversations": [{"content": "a fox darting between snow-covered pines at dusk", "role": "user"}]}
{"conversations": [{"content": "a kite surfer riding emerald waves under a cloudy sky", "role": "user"}]}
```

`conversations` 包含图片生成描述的对话列表，`content` 字段是其中的文本提示词。

### 第四步：运行流水线

```bash
python dataflow/statics/pipelines/gpu_pipelines/text_to_image_generation_pipeline.py \
    --first_entry_file_name <your_input_text_file_path>
```

生成的文件会默认保存在 `./cache_local/text2image_local` 文件夹内。

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据包括以下字段：

* **conversations**：对话格式数据，包含文本提示词。

这些输入数据存储在 `jsonl` 文件中，并通过 `FileStorage` 对象进行管理和读取：

```python
self.storage = FileStorage(
    first_entry_file_name="<your_jsonl_file_path>",
    cache_path="./cache_local/text2image_local",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)
```

### 2. **文本到图像生成（PromptedImageGenerator）**

流程的核心步骤是使用**提示式图像生成器**（`PromptedImageGenerator`）结合本地GPU模型为每个文本提示词生成对应的图像。

**功能：**

* 利用本地GPU模型（如FLUX.1-dev）根据文本提示词生成图像
* 支持配置推理步数、图像尺寸等参数
* 可调整批次大小以优化GPU利用率
* 自动保存生成的图像到指定路径

**输入**：对话格式数据（包含文本提示词）  
**输出**：生成的图像文件路径

**本地GPU服务配置**：

```python
self.serving = LocalImageGenServing(
    image_io=ImageIO(save_path=image_save_path),  # 图像保存路径
    batch_size=4,  # 批次大小
    hf_model_name_or_path="black-forest-labs/FLUX.1-dev",  # 模型路径
    hf_cache_dir="./cache_local",  # Hugging Face模型缓存目录
    hf_local_dir="./ckpt/models/",  # 本地模型存储目录
    diffuser_num_inference_steps=20,  # 扩散模型推理步数
    diffuser_image_height=512,  # 生成图片的高度
    diffuser_image_width=512,  # 生成图片的宽度
)
```

**算子初始化**：

```python
self.text_to_image_generator = PromptedImageGenerator(
    t2i_serving=self.serving,  # 文本到图像服务
    save_interval=10  # 保存间隔
)
```

**算子运行**：

```python
self.text_to_image_generator.run(
    storage=self.storage.step(),
    input_conversation_key="conversations",  # 输入对话字段
    output_image_key="images",  # 输出图像字段
)
```

### 3. **输出数据**

最终，流水线生成的输出数据将包含以下内容：

* **conversations**：原始对话数据（包含文本提示词）
* **images**：生成的图像文件路径列表

**输出数据示例**：

```jsonl
{"conversations":[{"content":"a fox darting between snow-covered pines at dusk","role":"user"}],"images":["./cache_local/text2image_local/sample0_condition0/sample0_condition0_0.png"]}
```

---

## 4. 流水线示例

下面给出使用本地FLUX模型的文本到图片生成流水线示例：
```python
import os
from pathlib import Path
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.serving.local_image_gen_serving import LocalImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationPipeline():
    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        
        prompts_file = project_root / "dataflow" / "example" / "image_gen" / "text2image" / "prompts.jsonl"
        
        # -------- 存储配置 --------
        self.storage = FileStorage(
            first_entry_file_name=str(prompts_file),
            cache_path="./cache_local/text2image_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        image_save_path = str(project_root / "cache_local" / "text2image_local")
        
        # -------- 本地GPU图像生成服务 --------
        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=image_save_path),
            batch_size=4,
            hf_model_name_or_path="black-forest-labs/FLUX.1-dev",  # 或模型在本地的路径
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            diffuser_num_inference_steps=20,
            diffuser_image_height=512,
            diffuser_image_width=512,
        )

        # -------- 文本到图像生成算子 --------
        self.text_to_image_generator = PromptedImageGenerator(
            t2i_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        # 调用 PromptedImageGenerator 生成图像
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversations",
            output_image_key="images",
        )

if __name__ == "__main__":
    # -------- 流水线入口 --------
    model = ImageGenerationPipeline()
    model.forward()
```
