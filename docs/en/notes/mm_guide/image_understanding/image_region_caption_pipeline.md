---
title: Image Region Caption Pipeline
createTime: 2026/01/11 22:04:27
icon: mdi:image-text
permalink: /en/mm_guide/image_region_caption_pipeline/
---
## 1. Overview

The **Image Region Caption Pipeline** is designed to generate detailed text descriptions for specific regions within an image. Combining the localization capabilities of Computer Vision with the understanding of Multimodal Large Models (VLMs), this pipeline identifies Regions of Interest (ROI) and generates precise natural language annotations for them.

This pipeline supports processing **pre-defined Bounding Box** data, visualizing these boxes, and then feeding them into a VLM for caption generation.

We support the following application scenarios:

* **Dense Captioning**: Generating descriptions for multiple objects within a single image.
* **Fine-grained Image Understanding**: Focusing on local details rather than global descriptions.
* **Dataset Augmentation**: Constructing image-text pair datasets that include localization information.

The main process of the pipeline includes:

1. **Data Loading**: Reading source data containing image paths and bounding box information.
2. **BBox Processing & Visualization**: Processing input bounding boxes and generating a version of the image with visual markers (e.g., drawn boxes).
3. **Region Caption Generation**: Using a VLM to generate text descriptions based on the marked images or specific regions.

---

## 2. Quick Start

### Step 1: Create a New DataFlow Working Directory

```bash
mkdir run_dataflow
cd run_dataflow

```

### Step 2: Initialize DataFlow-MM

```bash
dataflowmm init

```

You will then see:

```bash
gpu_pipelines/image_region_caption_pipeline.py
```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data
```

### Step 4: Configure Parameters
```python
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        hf_cache_dir: str = "~/.cache/huggingface",
        download_dir: str = "../ckpt/models/Qwen2.5-VL-3B-Instruct",
        first_entry_file: str = "../example_data/image_region_caption/image_region_caption_demo.jsonl",
        cache_path: str = "../cache/image_region_caption",
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_bbox_key: str = "bbox",
        max_boxes: int = 10,
        output_image_with_bbox_path: str = "../cache/image_region_caption/image_with_bbox_result.jsonl",
    ):
```
> **⚠️ Important Note on Model Path Configuration (Taking `Qwen2.5-VL-3B-Instruct` as an example):**
> 
> * **If you have already downloaded the model files:** Please change `model_path` to your local model path. **Crucially**, ensure that the model folder is named exactly `Qwen2.5-VL-3B-Instruct`; otherwise, the framework will fail to recognize it.
> * **If you haven't downloaded the model yet:** You must specify a `download_dir` parameter that ends with `Qwen2.5-VL-3B-Instruct` (as shown in the default parameters). Failure to do so will also result in the model not being recognized after downloading.


### Step 5: Run

```bash
cd gpu_pipelines
python image_region_caption_pipeline.py
```
> **🛠️ Troubleshooting**
> 
> **Issue 1:** If you encounter a CUDA library conflict error similar to the following:
> `ImportError: .../miniconda3/envs/Dataflow-MM/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`
> 
> **Solution:** This is usually caused by conflicting environment variables. Run the script with an empty `LD_LIBRARY_PATH`:
> ```bash
> LD_LIBRARY_PATH="" python image_region_caption_pipeline.py
> ```
> 
> **Issue 2:** If you are using **Qwen series models** and encounter the following error:
> `KeyError: "Missing required keys in rope_scaling for 'rope_type'='None': {'rope_type'}"`
> 
> **Solution:** Open the `config.json` file located in your model folder, find the `rope_scaling` section, and change the key `"type"` to `"rope_type"`.
> 
> **Before modification:**
> ```json
> "rope_scaling": {
>   "type": "mrope",
>   "mrope_section": [
>     16,
>     24,
>     24
>   ]
> }
> ```
> 
> **After modification:**
> ```json
> "rope_scaling": {
>   "rope_type": "mrope",
>   "mrope_section": [
>     16,
>     24,
>     24
>   ]
> }
> ```

---

## 3. Data Flow & Logic

### 1. **Input Data**

The input data typically contains the image path and a list of corresponding bounding boxes (optional):

* **image**: Path to the image file.
* **bbox**: List of bounding box coordinates, typically in `[[x, y, w, h], ...]` format.

**Input Data Example**:

```json
{
    "image": "../example_data/image_region_caption/20.jpg",
    "bbox": [[196, 104, 310, 495], [50, 60, 100, 200]]
}

```

### 2. **Core Operator Logic**

This pipeline chains two core operators to complete the task:

#### A. **ImageBboxGenerator**

This operator handles the vision-level tasks.

* **Input**: Raw image + `bbox` data.
* **Functionality**: Reads bounding boxes and draws them onto the image (visualization) or preprocesses them according to configuration.
* **Configuration (`ExistingBBoxDataGenConfig`)**: Controls the maximum number of bounding boxes and the input/output paths.
* **Output**: JSON output path for the new image with visual markers.

#### B. **PromptedVQAGenerator**

This operator is responsible for generating text using the VLM.

* **Input**: The result generated in the previous step.
* **Functionality**: The VLM receives the marked image and generates descriptions for the corresponding regions based on prompts.
* **Output**: Region description text.

### 3. **Output Data**

The final generated output data includes the processed image path and the generated descriptions:

* **image**: The input image path.
* **type**: Indicates whether a bounding box is provided.
* **bbox**: Bounding box parameters.
* **normalized_bbox**: Normalized bounding box parameters.
* **result_file**: The output path for the results.
* **image_with_bbox**: Path to the image with drawn bounding boxes.
* **valid_bboxes_num**: The number of valid bounding boxes.
* **prompt**: The prompt received by the VLM.
* **answer**: The list of generated region descriptions.

**Output Data Example**:

```json
{
    "image":"..\/example_data\/image_region_caption\/20.png",
    "type":"with_bbox",
    "bbox":[[196,104,310,495]],
    "normalized_bbox":[[0.128,0.125,0.329,0.72],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],
    "result_file":"..\/cache\/image_region_caption",
    "image_with_bbox":"..\/cache\/image_region_caption\\2_bbox_vis.jpg",
    "valid_bboxes_num":1,
    "prompt":"Describe the content of each marked region in the image. There are 1 regions: <region1> to <region1>.",
    "answer":"In <region1>, the focus is on the lower half of a person wearing high-heeled shoes with an ornate design. The setting appears to be a kitchen, with items such as a table with floral tablecloth, a broom, and various kitchen utensils visible in the background. The legs of another person can also be seen, indicating there may be interaction happening in this domestic space. The overall scene captures a domestic and casual atmosphere."
}
```

---

## 4. Pipeline Example

Below is the complete `ImageRegionCaptionPipeline` code implementation.

```python
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
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        hf_cache_dir: str = "~/.cache/huggingface",
        download_dir: str = "../ckpt/models/Qwen2.5-VL-3B-Instruct",
        first_entry_file: str = "../example_data/image_region_caption/image_region_caption_demo.jsonl",
        cache_path: str = "../cache/image_region_caption",
        file_name_prefix: str = "region_caption",
        cache_type: str = "jsonl",
        input_image_key: str = "image",
        input_bbox_key: str = "bbox",
        max_boxes: int = 10,
        output_image_with_bbox_path: str = "../cache/image_region_caption/image_with_bbox_result.jsonl",
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
    pipe = ImageRegionCaptionPipeline()
    pipe.forward()

```

