---
title: Image Editing Pipeline (GPU Version)
createTime: 2026/02/15 15:12:51
permalink: /en/mm_guide/image_editing_gpu/
icon: material-symbols:draw-outline-rounded
---

# Image Editing Pipeline (GPU Version)

## 1. Overview

The **Image Editing Pipeline** generates corresponding edited images based on existing images and editing instructions. The usage is very simple: input image files and editing instructions (text prompts), and output edited images.

This version uses **local GPU models** for image editing, supporting local deployment of the FLUX.1-Kontext-dev model.

> 💡 **Tip**: If you want to use cloud API models for image editing, please see [Image Editing Pipeline (API Version)](/en/mm_guide/image_editing_api/)

---

## 2. Quick Start
### Step 1: Create a New DataFlow Working Directory
```bash
mkdir run_dataflow_mm
cd run_dataflow_mm
```

### Step 2: Configure Model Path

Configure the model path in the pipeline code. Two methods are supported:

(1) Method 1: Use Hugging Face model path (auto-download)
```python
hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev"
```

(2) Method 2: Use local model path (downloaded model)
```python
hf_model_name_or_path="/path/to/your/local/FLUX.1-Kontext-dev"
```

Modify the `hf_model_name_or_path` parameter of `LocalImageGenServing` in `image_editing_pipeline.py`:

```python
self.serving = LocalImageGenServing(
    image_io=ImageIO(save_path=image_save_path),
    hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",  # Model path
    hf_cache_dir="./cache_local",  # Hugging Face model cache directory
    hf_local_dir="./ckpt/models/",  # Local model storage directory
    Image_gen_task="imageedit",  # Image editing task type
    batch_size=4,  # Batch size
    diffuser_model_name="FLUX-Kontext",  # Diffusion model name
    diffuser_num_inference_steps=28,  # Diffusion model inference steps, adjustable to balance speed and quality
    diffuser_guidance_scale=3.5,  # Guidance scale, controls the match between generated results and prompts
)
```

### Step 3: Prepare Image and Text Data
We use `jsonl` files to store image and text data, with one sample per line. Here is a simple example of input data:
```jsonl
{"images": "image.png", "conversations": [{"role": "user", "content": "Change the vase to red."}]}
{"images": "human_inpaint.jpg", "conversations": [{"role": "user", "content": "Complete the shadowed part into a generic person's portrait."}]}
```

`images` is the path to the image to be edited; `conversations` contains a list of dialogues with image editing instructions, and the `content` field is the text prompt.

### Step 4: Run the Pipeline
```bash
python dataflow/statics/gpu_pipelines/image_editing_pipeline.py
```

Generated files will be saved by default in the `./cache_local/image_edit_local` directory.

---

## 3. Data Flow and Pipeline Logic

### 1. **Input Data**

The input data for this pipeline includes the following fields:

* **images**: Path to the image(s) to be edited, supporting image formats such as `png`, `jpg`, etc.
* **conversations**: Dialogue format data containing image editing instructions.

This input data is stored in `jsonl` files and managed and read through the `FileStorage` object:

```python
self.storage = FileStorage(
    first_entry_file_name="<your_jsonl_file_path>",
    cache_path="./cache_local/image_edit_local",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)
```

### 2. **Image Editing Generation (PromptedImageEditGenerator)**

The core step of the pipeline is using the **Prompted Image Edit Generator** (`PromptedImageEditGenerator`) combined with local GPU models to generate edited images based on input images and editing instructions.

**Features:**

* Generate edited images from images and editing instructions using local GPU models (e.g., FLUX.1-Kontext-dev)
* Support configuration of inference steps, guidance scale, and other parameters
* Adjustable batch size to optimize GPU utilization
* Automatically save generated images to specified paths

**Input**: Image file paths and dialogue format data (containing editing instructions)  
**Output**: Edited image file paths

**Local GPU Service Configuration**:

```python
self.serving = LocalImageGenServing(
    image_io=ImageIO(save_path=image_save_path),  # Image save path
    hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",  # Model path
    hf_cache_dir="./cache_local",  # Hugging Face model cache directory
    hf_local_dir="./ckpt/models/",  # Local model storage directory
    Image_gen_task="imageedit",  # Task type: image editing
    batch_size=4,  # Batch size
    diffuser_model_name="FLUX-Kontext",  # Diffusion model name
    diffuser_num_inference_steps=28,  # Diffusion model inference steps
    diffuser_guidance_scale=3.5,  # Guidance scale
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
{"images":"image.png","conversations":[{"role":"user","content":"Make the whole picture more vibrant."}],"output_image":["./cache_local/image_edit_local/sample_0/sample_0_0.png"]}
```

---

## 4. Pipeline Example

Below is an example of an image editing pipeline using local FLUX models:
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
        
        # -------- Storage Configuration --------
        self.storage = FileStorage(
            first_entry_file_name=str(data_file),
            cache_path="./cache_local/image_edit_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        image_save_path = str(project_root / "cache_local" / "image_edit_local")
        
        # -------- Local GPU Image Editing Service --------
        self.serving = LocalImageGenServing(
            image_io=ImageIO(save_path=image_save_path),
            hf_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",  # Or local model path
            hf_cache_dir="./cache_local",
            hf_local_dir="./ckpt/models/",
            Image_gen_task="imageedit",
            batch_size=4,
            diffuser_model_name="FLUX-Kontext",
            diffuser_num_inference_steps=28,
            diffuser_guidance_scale=3.5,
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
    # -------- Pipeline Entry Point --------
    model = ImageEditingPipeline()
    model.forward()
```
