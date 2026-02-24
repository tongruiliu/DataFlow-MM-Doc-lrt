---
title: 图像编辑流水线（API版）
createTime: 2026/02/15 10:23:04
permalink: /zh/mm_guide/ffwfa9ck/
icon: material-symbols:landscape-2-edit-outline-rounded
---

# 图像编辑流水线（API版）

## 1. 概述

**图像编辑流水线**的核心功能是根据已有的图片和编辑指令生成对应的编辑后图片。使用方式非常简单，即输入图片文件和编辑指令（文本prompt），输出编辑后的图片。

本版本使用**云端API模型**进行图片编辑，目前支持的API模型包括：
- OpenAI格式：`dall-e-2`, `gpt-image-1`
- Gemini格式：`gemini-2.5-flash-image`, `gemini-3-pro-image-preview` 等

> 💡 **提示**：如果要使用本地GPU模型进行图片编辑，请查看 [图像编辑流水线（GPU版）](/zh/mm_guide/itxtxm62/)

不同模型支持的图片编辑能力有所不同，具体如下：
- **gpt-image-1**：
    - 编辑现有图像，包括替换部分区域（"内绘"）、使用蒙版等
    - 以其他图像为参考生成新图像

- **dall-e-2**：
    - 编辑现有图像
    - 生成现有图像的变体

- **Gemini系列模型（Nano banana、Nano banana pro）**：
  - 编辑现有图像，包括添加/移除/修改元素、更改样式、调整色彩分级等
  - **多图输入**：支持上传多张图片进行编辑
  - **多轮编辑**：支持使用多轮对话的方式来迭代修改图片

> **注意**：各模型的具体功能、限制和详细用法请参考各模型的官方API文档

---

## 2. 快速开始
### 第一步：创建新的 DataFlow 工作文件夹
```bash
mkdir run_dataflow_mm
cd run_dataflow_mm
```

### 第二步：配置 API KEY 和 BASE URL
通过设置环境变量来配置API KEY和BASE URL：

```bash
# 设置API密钥（必需）
export DF_API_KEY=<your_api_key>

# 设置API基础URL（可选）
# 如果不设置，将根据选用的API格式使用默认URL：
# - Gemini格式：https://generativelanguage.googleapis.com
# - OpenAI格式：https://api.openai.com/v1
export DF_BASE_URL=<your_base_url>  # 可选
```

### 第三步：准备图片和文本数据
我们使用 `jsonl` 文件来保存图片和文本数据，每行一个样本。下面是一个简单的输入数据样例：
```jsonl
{"images": "image.png", "conversations": [{"role": "user", "content": "Change the guitar to a piano."}]}
{"images": ["image.png", "human_inpaint.jpg"], "conversations": [{"role": "user", "content": "Change the color of the vase in the first picture to the background color of the second picture."}]}
{"images": "human_inpaint.jpg", "conversations": [{"role": "user", "content": "Complete the shadowed part into a generic person's portrait."}, {"role": "user", "content": "Update this graph to be in English."}]}
```

`images` 是待编辑的图片路径；`conversations` 包含图片编辑指令的对话列表，`content` 字段是其中的文本提示词。

对于Gemini系列模型，`images` 支持字符串（单张图片路径）或列表（多张图片路径），`conversations` 支持多轮对话。

### 第四步：运行流水线

- **基本用法**
```bash
python dataflow/statics/pipelines/api_pipelines/image_editing_api_pipeline.py \
    --first_entry_file_name <your_input_data_file_path>
```

生成的文件会默认保存在 `./cache_local/image_edit_api` 文件夹内。

- **命令行参数说明**

本流水线支持以下命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--api_format` | `str` | `gemini` | API格式类型，可选 `openai` 或 `gemini` |
| `--model_name` | `str` | `gemini-3-pro-image-preview` | 模型名称，可选 `dall-e-2`, `gpt-image-1`, `gemini-2.5-flash-image`, `gemini-3-pro-image-preview` 等 |
| `--batch_size` | `int` | `4` | 批次大小，控制每次处理的样本数量 |
| `--first_entry_file_name` | `str` | `None` | 输入数据文件路径（ `jsonl` 格式） |
| `--cache_path` | `str` | `./cache_local/image_edit_api` | 缓存路径，用于存储中间结果和最终生成的图片 |

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据包括以下字段：

* **images**：待编辑的图片路径，可以是字符串或列表，支持 `png` 、 `jpg` 等图片格式。
* **conversations**：对话格式数据，包含图片编辑指令。

这些输入数据存储在 `jsonl` 文件中，并通过 `FileStorage` 对象进行管理和读取：

```python
self.storage = FileStorage(
    first_entry_file_name="<your_jsonl_file_path>",
    cache_path="./cache_local/image_edit_api",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)
```

### 2. **图像编辑生成（PromptedImageEditGenerator）**

流程的核心步骤是使用**提示式图像编辑生成器**（`PromptedImageEditGenerator`）结合云端API服务根据输入图片和编辑指令生成编辑后的图像。

**功能：**

* 利用云端API模型根据图片和编辑指令生成编辑后的图像
* 支持多种API格式（OpenAI、Gemini等）
* 支持单图或多图输入（取决于模型能力）
* 支持多轮对话编辑（Gemini系列模型）
* 可配置批次大小和生成参数
* 自动保存生成的图像到指定路径

**输入**：图片文件路径和对话格式数据（包含编辑指令）  
**输出**：编辑后的图像文件路径

**API服务配置**：

```python
self.serving = APIImageGenServing(
    api_url=api_url,  # API服务地址
    image_io=ImageIO(save_path=image_save_path),  # 图像保存路径
    Image_gen_task="imageedit",  # 任务类型：图像编辑
    batch_size=4,  # 批次大小
    api_format="gemini",  # API格式：gemini 或 openai
    model_name="gemini-3-pro-image-preview",  # 模型名称
    api_key=api_key,  # API密钥
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
{"images":"image.png","conversations":[{"role":"user","content":"Change the guitar to a piano."}],"output_image":["./cache_local/image_edit_api/sample_0/sample_0_0.png"]}
```

---

## 4. 流水线示例

以下给出使用云端API模型的图片编辑流水线示例：
```python
import os
import argparse
from pathlib import Path
from dataflow.operators.core_vision import PromptedImageEditGenerator
from dataflow.serving.api_image_gen_serving import APIImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageEditingAPIPipeline():
    """
    Image Editing API Pipeline
    Supported Models:
        OpenAI format (api_format="openai"): dall-e-2, gpt-image-1
        Gemini format (api_format="gemini"): gemini-2.5-flash-image, gemini-3-pro-image-preview, etc.
    """
    def __init__(
        self, 
        api_format="gemini",
        model_name="gemini-3-pro-image-preview",
        batch_size=4,
        first_entry_file_name=None,
        cache_path="./cache_local/image_edit_api",
    ):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        
        if first_entry_file_name is None:
            data_file = project_root / "dataflow" / "example" / "image_gen" / "image_edit" / "prompts_api.jsonl"
            first_entry_file_name = str(data_file)
        
        # -------- 存储配置 --------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        if api_format not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported api_format: {api_format}. Only 'gemini' and 'openai' are supported for image editing.")
        
        # -------- API 配置 --------
        api_key = os.environ.get("DF_API_KEY")
        api_url = os.environ.get("DF_BASE_URL")
        
        if api_key is None:
            raise ValueError("API key is required. Please set it via environment variable DF_API_KEY")
        
        if api_url is None:
            if api_format == "gemini":
                api_url = "https://generativelanguage.googleapis.com"
            else:  # openai
                api_url = "https://api.openai.com/v1"
        
        image_save_path = str(project_root / "cache_local" / "image_edit_api")
        
        # -------- 图像编辑 API 服务 --------
        self.serving = APIImageGenServing(
            api_url=api_url,
            image_io=ImageIO(save_path=image_save_path),
            Image_gen_task="imageedit",
            batch_size=batch_size,
            api_format=api_format,
            model_name=model_name,
            api_key=api_key,
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
    # -------- 命令行参数解析 --------
    parser = argparse.ArgumentParser(description="Cloud API Image Editing Pipeline")
    parser.add_argument('--api_format', choices=['gemini', 'openai'], default='gemini',
                        help='API format type: gemini (Google Gemini) or openai (OpenAI DALL-E 2 / gpt-image-1)')
    parser.add_argument('--model_name', type=str, default='gemini-3-pro-image-preview',
                        help='Model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--first_entry_file_name', type=str, default=None,
                        help='Input data file path (default uses example_data)')
    parser.add_argument('--cache_path', type=str, default="./cache_local/image_edit_api",
                        help='Cache path')
    args = parser.parse_args()
    
    if not os.environ.get("DF_API_KEY"):
        parser.error("Environment variable DF_API_KEY is not set. Please use export DF_API_KEY=your_api_key to set it")
    
    # -------- 流水线入口 --------
    model = ImageEditingAPIPipeline(
        api_format=args.api_format,
        model_name=args.model_name,
        batch_size=args.batch_size,
        first_entry_file_name=args.first_entry_file_name,
        cache_path=args.cache_path,
    )
    model.forward()
```