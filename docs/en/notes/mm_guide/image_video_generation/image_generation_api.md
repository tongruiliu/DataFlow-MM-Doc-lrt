---
title: Image Generation Pipeline (API Version)
createTime: 2026/02/15 15:07:26
permalink: /en/mm_guide/image_generation_api/
icon: material-symbols:photo-prints-outline-rounded
---

# Image Generation Pipeline (API Version)

## 1. Overview

The **Image Generation Pipeline** generates target images from user-provided text, providing image data for subsequent tasks such as image understanding and image editing.

This version uses **cloud API models** for text-to-image generation. Currently supported API models include:
- OpenAI format: `dall-e-2`, `dall-e-3`, `gpt-image-1`
- Gemini format: `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, etc.

For specific features and detailed usage of different models, please refer to the official API documentation of each model.

> 💡 **Tip**: If you want to use local GPU models for text-to-image generation, please see [Image Generation Pipeline (GPU Version)](/en/mm_guide/image_generation_gpu/)

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

### Step 3: Prepare Text Data
We use `jsonl` files to store text data, with one sample per line. Here is a simple example of input data:
```jsonl
{"conversations": [{"content": "a fox darting between snow-covered pines at dusk", "role": "user"}]}
{"conversations": [{"content": "a kite surfer riding emerald waves under a cloudy sky", "role": "user"}]}
```

`conversations` contains a list of dialogues for image generation descriptions, and the `content` field is the text prompt.

### Step 4: Run the Pipeline

- **Basic Usage**
```bash
python dataflow/statics/api_pipelines/text_to_image_generation_api_pipeline.py \
    --first_entry_file_name <your_input_text_file_path>
```

Generated files will be saved by default in the `./cache_local/text2image_api` directory.

- **Command-line Arguments**

This pipeline supports the following command-line arguments:

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--api_format` | `str` | `gemini` | API format type, options: `openai` or `gemini` |
| `--model_name` | `str` | `gemini-3-pro-image-preview` | Model name, options: `dall-e-2`, `dall-e-3`, `gpt-image-1`, `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, etc. |
| `--batch_size` | `int` | `4` | Batch size, controls the number of samples processed each time |
| `--first_entry_file_name` | `str` | `None` | Input data file path (jsonl format) |
| `--cache_path` | `str` | `./cache_local/text2image_api` | Cache path for storing intermediate results and final generated images |

---

## 3. Data Flow and Pipeline Logic

### 1. **Input Data**

The input data for this pipeline includes the following fields:

* **conversations**: Dialogue format data containing text prompts.

This input data is stored in `jsonl` files and managed and read through the `FileStorage` object:

```python
self.storage = FileStorage(
    first_entry_file_name="<your_jsonl_file_path>",
    cache_path="./cache_local/text2image_api",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)
```

### 2. **Text-to-Image Generation (PromptedImageGenerator)**

The core step of the pipeline is using the **Prompted Image Generator** (`PromptedImageGenerator`) combined with cloud API services to generate corresponding images for each text prompt.

**Features:**

* Generate images from text prompts using cloud API models
* Support multiple API formats (OpenAI, Gemini, etc.)
* Configurable batch size and generation parameters
* Automatically save generated images to specified paths

**Input**: Dialogue format data (containing text prompts)  
**Output**: Generated image file paths

**API Service Configuration**:

```python
self.serving = APIImageGenServing(
    api_url=api_url,  # API service address
    image_io=ImageIO(save_path=image_save_path),  # Image save path
    Image_gen_task="text2image",  # Task type: text-to-image
    batch_size=4,  # Batch size
    api_format="gemini",  # API format: gemini or openai
    model_name="gemini-3-pro-image-preview",  # Model name
    api_key=api_key,  # API key
)
```

**Operator Initialization**:

```python
self.text_to_image_generator = PromptedImageGenerator(
    t2i_serving=self.serving,  # Text-to-image service
    save_interval=10  # Save interval
)
```

**Operator Execution**:

```python
self.text_to_image_generator.run(
    storage=self.storage.step(),
    input_conversation_key="conversations",  # Input dialogue field
    output_image_key="images",  # Output image field
)
```

### 3. **Output Data**

Finally, the output data generated by the pipeline will include the following:

* **conversations**: Original dialogue data (containing text prompts)
* **images**: List of generated image file paths

**Output Data Example**:

```jsonl
{"conversations":[{"content":"a fox darting between snow-covered pines at dusk","role":"user"}],"images":["./cache_local/text2image_api/sample0_condition0/sample0_condition0_0.png"]}
```

---

## 4. Pipeline Example

Below is an example of a text-to-image generation pipeline using cloud API:
```python
import os
import argparse
from pathlib import Path
from dataflow.operators.core_vision import PromptedImageGenerator
from dataflow.serving.api_image_gen_serving import APIImageGenServing
from dataflow.utils.storage import FileStorage
from dataflow.io import ImageIO


class ImageGenerationAPIPipeline():
    """
    Text to Image Generation API Pipeline
    Supported Models:
        OpenAI format (api_format="openai"): dall-e-2, dall-e-3, gpt-image-1
        Gemini format (api_format="gemini"): gemini-2.5-flash-image, gemini-3-pro-image-preview, etc.
    """
    def __init__(
        self, 
        api_format="gemini",
        model_name="gemini-3-pro-image-preview",
        batch_size=4,
        first_entry_file_name=None,
        cache_path="./cache_local/text2image_api",
    ):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        
        if first_entry_file_name is None:
            data_file = project_root / "dataflow" / "example" / "image_gen" / "text2image" / "prompts.jsonl"
            first_entry_file_name = str(data_file)
        
        # -------- Storage Configuration --------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )
        
        # -------- API Configuration --------
        api_key = os.environ.get("DF_API_KEY")
        api_url = os.environ.get("DF_BASE_URL")
        
        if api_key is None:
            raise ValueError("API key is required. Please set it via environment variable DF_API_KEY")
        
        if api_url is None:
            if api_format == "gemini":
                api_url = "https://generativelanguage.googleapis.com"
            else:
                api_url = "https://api.openai.com/v1"
        
        image_save_path = str(project_root / "cache_local" / "text2image_api")
        
        # -------- Image Generation API Service --------
        self.serving = APIImageGenServing(
            api_url=api_url,
            image_io=ImageIO(save_path=image_save_path),
            Image_gen_task="text2image",
            batch_size=batch_size,
            api_format=api_format,
            model_name=model_name,
            api_key=api_key,
        )

        # -------- Text-to-Image Generation Operator --------
        self.text_to_image_generator = PromptedImageGenerator(
            t2i_serving=self.serving,
            save_interval=10
        )
    
    def forward(self):
        # Call PromptedImageGenerator to generate images
        self.text_to_image_generator.run(
            storage=self.storage.step(),
            input_conversation_key="conversations",
            output_image_key="images",
        )

if __name__ == "__main__":
    # -------- Command-line Argument Parsing --------
    parser = argparse.ArgumentParser(description="Cloud API Image Generation Pipeline")
    parser.add_argument('--api_format', choices=['openai', 'gemini'], default='gemini',
                        help='API format type: openai (OpenAI DALL-E) or gemini (Google Gemini)')
    parser.add_argument('--model_name', type=str, default='gemini-3-pro-image-preview',
                        help='Model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--first_entry_file_name', type=str, default=None,
                        help='Input data file path (default uses example_data)')
    parser.add_argument('--cache_path', type=str, default="./cache_local/text2image_api",
                        help='Cache path')
    args = parser.parse_args()
    
    if not os.environ.get("DF_API_KEY"):
        parser.error("Environment variable DF_API_KEY is not set. Please use export DF_API_KEY=your_api_key to set it")
    
    # -------- Pipeline Entry Point --------
    model = ImageGenerationAPIPipeline(
        api_format=args.api_format,
        model_name=args.model_name,
        batch_size=args.batch_size,
        first_entry_file_name=args.first_entry_file_name,
        cache_path=args.cache_path,
    )
    model.forward()
```

