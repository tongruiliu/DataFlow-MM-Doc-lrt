---
title: Image Editing Pipeline (API Version)
createTime: 2026/02/15 15:09:18
permalink: /en/mm_guide/image_editing_api/
icon: material-symbols:landscape-2-edit-outline-rounded
---

# Image Editing Pipeline (API Version)

## 1. Overview

The **Image Editing Pipeline** generates corresponding edited images based on existing images and editing instructions. The usage is very simple: input image files and editing instructions (text prompts), and output edited images.

This version uses **cloud API models** for image editing. Currently supported API models include:
- OpenAI format: `dall-e-2`, `gpt-image-1`
- Gemini format: `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, etc.

> 💡 **Tip**: If you want to use local GPU models for image editing, please see [Image Editing Pipeline (GPU Version)](/en/mm_guide/image_editing_gpu/)

Different models support different image editing capabilities, as follows:
- **gpt-image-1**:
    - Edit existing images, including replacing partial regions ("inpainting"), using masks, etc.
    - Generate new images using other images as references

- **dall-e-2**:
    - Edit existing images
    - Generate variants of existing images

- **Gemini series models (Nano banana, Nano banana pro)**:
  - Edit existing images, including adding/removing/modifying elements, changing styles, adjusting color grading, etc.
  - **Multi-image input**: Support uploading multiple images for editing
  - **Multi-round editing**: Support iterative image modification using multi-turn conversations

> **Note**: For specific features, limitations, and detailed usage of each model, please refer to the official API documentation of each model.

---

## 2. Quick Start
### Step 1: Create a New DataFlow Working Directory
```bash
mkdir run_dataflow_mm
cd run_dataflow_mm
```

### Step 2: Configure API KEY and BASE URL
Configure the API KEY and BASE URL by setting environment variables:

```bash
# Set API key (required)
export DF_API_KEY=<your_api_key>

# Set API base URL (optional)
# If not set, default URLs will be used based on the selected API format:
# - Gemini format: https://generativelanguage.googleapis.com
# - OpenAI format: https://api.openai.com/v1
export DF_BASE_URL=<your_base_url>  # optional
```

### Step 3: Prepare Image and Text Data
We use `jsonl` files to store image and text data, with one sample per line. Here is a simple example of input data:
```jsonl
{"images": "image.png", "conversations": [{"role": "user", "content": "Change the guitar to a piano."}]}
{"images": ["image.png", "human_inpaint.jpg"], "conversations": [{"role": "user", "content": "Change the color of the vase in the first picture to the background color of the second picture."}]}
{"images": "human_inpaint.jpg", "conversations": [{"role": "user", "content": "Complete the shadowed part into a generic person's portrait."}, {"role": "user", "content": "Update this graph to be in English."}]}
```

`images` is the path to the image to be edited; `conversations` contains a list of dialogues with image editing instructions, and the `content` field is the text prompt.

For Gemini series models, `images` supports strings (single image path) or lists (multiple image paths), and `conversations` supports multi-turn dialogues.

### Step 4: Run the Pipeline

- **Basic Usage**
```bash
python dataflow/statics/api_pipelines/image_editing_api_pipeline.py \
    --first_entry_file_name <your_input_data_file_path>
```

Generated files will be saved by default in the `./cache_local/image_edit_api` directory.

- **Command-line Arguments**

This pipeline supports the following command-line arguments:

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--api_format` | `str` | `gemini` | API format type, options: `openai` or `gemini` |
| `--model_name` | `str` | `gemini-3-pro-image-preview` | Model name, options: `dall-e-2`, `gpt-image-1`, `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, etc. |
| `--batch_size` | `int` | `4` | Batch size, controls the number of samples processed each time |
| `--first_entry_file_name` | `str` | `None` | Input data file path (jsonl format) |
| `--cache_path` | `str` | `./cache_local/image_edit_api` | Cache path for storing intermediate results and final generated images |

---

## 3. Data Flow and Pipeline Logic

### 1. **Input Data**

The input data for this pipeline includes the following fields:

* **images**: Path to the image(s) to be edited, can be a string or a list, supporting image formats such as `png`, `jpg`, etc.
* **conversations**: Dialogue format data containing image editing instructions.

This input data is stored in `jsonl` files and managed and read through the `FileStorage` object:

```python
self.storage = FileStorage(
    first_entry_file_name="<your_jsonl_file_path>",
    cache_path="./cache_local/image_edit_api",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)
```

### 2. **Image Editing Generation (PromptedImageEditGenerator)**

The core step of the pipeline is using the **Prompted Image Edit Generator** (`PromptedImageEditGenerator`) combined with cloud API services to generate edited images based on input images and editing instructions.

**Features:**

* Generate edited images from images and editing instructions using cloud API models
* Support multiple API formats (OpenAI, Gemini, etc.)
* Support single or multiple image input (depending on model capabilities)
* Support multi-turn dialogue editing (Gemini series models)
* Configurable batch size and generation parameters
* Automatically save generated images to specified paths

**Input**: Image file paths and dialogue format data (containing editing instructions)  
**Output**: Edited image file paths

**API Service Configuration**:

```python
self.serving = APIImageGenServing(
    api_url=api_url,  # API service address
    image_io=ImageIO(save_path=image_save_path),  # Image save path
    Image_gen_task="imageedit",  # Task type: image editing
    batch_size=4,  # Batch size
    api_format="gemini",  # API format: gemini or openai
    model_name="gemini-3-pro-image-preview",  # Model name
    api_key=api_key,  # API key
)
```

**Operator Initialization**:

```python
self.image_edit_generator = PromptedImageEditGenerator(
    image_edit_serving=self.serving,  # Image editing service
    save_interval=10  # Save interval
)
```

**Operator Execution**:

```python
self.image_edit_generator.run(
    storage=self.storage.step(),
    input_image_key="images",  # Input image field
    input_conversation_key="conversations",  # Input dialogue field
    output_image_key="output_image",  # Output image field
)
```

### 3. **Output Data**

Finally, the output data generated by the pipeline will include the following:

* **images**: Original input image path(s)
* **conversations**: Original dialogue data (containing editing instructions)
* **output_image**: List of edited image file paths

**Output Data Example**:

```jsonl
{"images":"image.png","conversations":[{"role":"user","content":"Change the guitar to a piano."}],"output_image":["./cache_local/image_edit_api/sample_0/sample_0_0.png"]}
```

---

## 4. Pipeline Example

Below is an example of an image editing pipeline using cloud API models:
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
        
        # -------- Storage Configuration --------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        if api_format not in ["gemini", "openai"]:
            raise ValueError(f"Unsupported api_format: {api_format}. Only 'gemini' and 'openai' are supported for image editing.")
        
        # -------- API Configuration --------
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
        
        # -------- Image Editing API Service --------
        self.serving = APIImageGenServing(
            api_url=api_url,
            image_io=ImageIO(save_path=image_save_path),
            Image_gen_task="imageedit",
            batch_size=batch_size,
            api_format=api_format,
            model_name=model_name,
            api_key=api_key,
        )

        # -------- Image Editing Generation Operator --------
        self.image_edit_generator = PromptedImageEditGenerator(
            image_edit_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        # Call PromptedImageEditGenerator to generate edited images
        self.image_edit_generator.run(
            storage=self.storage.step(),
            input_image_key="images",
            input_conversation_key="conversations",
            output_image_key="output_image",
        )

if __name__ == "__main__":
    # -------- Command-line Argument Parsing --------
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
    
    # -------- Pipeline Entry Point --------
    model = ImageEditingAPIPipeline(
        api_format=args.api_format,
        model_name=args.model_name,
        batch_size=args.batch_size,
        first_entry_file_name=args.first_entry_file_name,
        cache_path=args.cache_path,
    )
    model.forward()
```
