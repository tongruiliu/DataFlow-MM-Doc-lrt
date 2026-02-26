---
title: Image Grounded CoT (GCoT) Pipeline (API version)
icon: mdi:image-text
createTime: 2026/01/11 20:44:55
permalink: /en/mm_guide/image_gcot_api/
---
## 1. Overview

The **Image Grounded Chain-of-Thought (GCoT) Pipeline** is designed to automatically generate **Grounded Chain-of-Thought** data. This pipeline generates multi-step reasoning to answer a question and simultaneously spatially locates (via Bounding Boxes) the key objects mentioned during the reasoning process. This significantly enhances the interpretability and precision of multimodal data.

Unlike traditional methods, this pipeline uses a **Single VLM (e.g., GPT-5)** to handle both "Reasoning" and "Grounding" tasks, making the process streamlined and efficient.

We support the following application scenarios:

* **Enhanced Multimodal Data Construction**: Adding interpretability and grounding annotations to VQA datasets.
* **Complex Scene Understanding**: Generating detailed reasoning steps containing object coordinates.
* **Model Reasoning Training**: Building data to train models to be "grounded" and reduce hallucinations.

The main process of the pipeline includes:

1. **CoT Generation**: The model generates step-by-step reasoning text and extracts key nouns.
2. **Keyword Parsing**: Cleaning and extracting keywords to be grounded from the generated text.
3. **Visual Grounding**: The model generates bounding boxes (BBoxes) for the extracted keywords.
4. **Information Injection**: Injecting BBox coordinates back into the reasoning text to form the final GCoT.

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
gpu_pipelines/image_gcot_pipeline.py
```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data
```

### Step 4: Configure API Key

Set your API Key environment variable in `api_pipelines/image_gcot_api_pipeline.py`:

```python
import os
os.environ["DF_API_KEY"] = "your_api_key"

```


### Step 5: Configure Parameters

Configure the API service and input data paths in `api_pipelines/image_gcot_api_pipeline.py`:

```python
    def __init__(
        self,
        *,
        first_entry_file: str,
        cache_path: str = "../cache/cache_gcot",
        file_name_prefix: str = "gcot",
        question_key: str = "question",
        answer_key: str = "answer",
        image_key: str = "image",
        output_key: str = "gcot",
        vllm_max_tokens: int = 512
    ):
```

```python
    pipe = ImageGCoTPipeline(
        first_entry_file="../example_data/capsbench_images/image_gcot_demo.jsonl"
    )
```

```python
self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Any API platform compatible with OpenAI format
            model_name="gpt-4o-mini",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

```
### Step 6: Run with One Command

```bash
cd api_pipelines
python image_gcot_api_pipeline.py
```

---

## 3. Data Flow & Logic

### 1. **Input Data**

The input data for this process typically consists of standard VQA data:

* **image**: Path to the image file.
* **question**: Question about the image.
* **answer**: Standard answer to the question (used to assist CoT generation).

**Input Data Example**:

```json
{
    "image":"../example_data/capsbench_images/0.png",
    "question":"Who is the lead actor in the movie \"Nightmare Alley\"?", 
    "answer": "Bradley Cooper."
}

```

### 2. **Core Operator Logic**

This pipeline combines multiple fine-grained operators to achieve complex GCoT generation logic:

#### A. **CoT Generation (PromptTemplatedVQAGenerator)**

Uses a predefined `GCOT_PROMPT_TEMPLATE` to guide the model to generate "Step-by-step Reasoning" and a "Keyword List".

* **Prompt Strategy**: Asks the model to output in the format `Step 1: ...`, `Step 2: ...`, `Keywords: ...`.
* **Output**: Raw string containing reasoning text and keywords.

#### B. **Text Cleaning & Extraction (FunctionalRefiner)**

Uses custom functions to parse the output from the previous step:

* `extract_clean_cot_logic`: Strips the keyword section, keeping pure CoT text.
* `extract_keywords_logic`: Parses the content after `Keywords:` to generate a Python List.

#### C. **Visual Grounding (VLMBBoxGenerator)**

Calls the VLM's grounding capability to generate bounding boxes for each extracted keyword.

* **Input**: Image + List of Keywords.
* **Output**: Dictionary mapping keywords to bounding box coordinates.

#### D. **Coordinate Injection (FunctionalRefiner)**

Uses the `inject_bboxes_logic` function to intelligently insert the generated BBox coordinates back into the original CoT text after the corresponding words.

### 3. **Output Data**

Finally, the output data generated by the pipeline will contain the following key fields:

* **raw_cot_output**: Raw text generated by the model.
* **cleaned_cot**: Cleaned reasoning text.
* **bbox_mapping**: Mapping of keywords to their coordinates.
* **gcot**: Final result, reasoning chain containing coordinate information.

**Output Data Example (gcot field)**:

```text
Step 1: Analyze the text visible in the image, which includes a list of actors beneath the title of the movie \"Nightmare Alley.\"\n\nStep 2: Identify the names listed. The first name listed is \"Bradley Cooper,\" indicating he is prominent in the film.\n\nStep 3: Recognize that the image is a promotional poster for \"Nightmare Alley,\" suggesting the individuals mentioned are likely key cast members.\n\nStep 4: Confirm that Bradley Cooper is identified as the lead actor based on his position at the top of the cast list.\n\nAnswer: Bradley Cooper.  \nKeywords: Nightmare Alley, cast list, poster.","cleaned_cot":"Step 1: Analyze the text visible in the image, which includes a list of actors beneath the title of the movie \"Nightmare Alley.\"\n\nStep 2: Identify the names listed. The first name listed is \"Bradley Cooper,\" indicating he is prominent in the film.\n\nStep 3: Recognize that the image is a promotional poster for \"Nightmare Alley,\" suggesting the individuals mentioned are likely key cast members.\n\nStep 4: Confirm that Bradley Cooper is identified as the lead actor based on his position at the top of the cast list.\n\nAnswer: Bradley Cooper.","extracted_keywords":["Nightmare Alley","cast list","poster"],"bbox_mapping":{},"gcot":"Step 1: Analyze the text visible in the image, which includes a list of actors beneath the title of the movie \"Nightmare Alley.\"\n\nStep 2: Identify the names listed. The first name listed is \"Bradley Cooper,\" indicating he is prominent in the film.\n\nStep 3: Recognize that the image is a promotional poster for \"Nightmare Alley,\" suggesting the individuals mentioned are likely key cast members.\n\nStep 4: Confirm that Bradley Cooper is identified as the lead actor based on his position at the top of the cast list.\n\nAnswer: Bradley Cooper.

```

---

## 4. Pipeline Example

Below is the complete `ImageGCoTAPIPipeline` code implementation.

```python
import os
os.environ["DF_API_KEY"] = "sk-xxxx"

import re
from typing import List, Dict, Any
import argparse
import gc
import torch
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import PromptTemplatedVQAGenerator, VLMBBoxGenerator
from dataflow.operators.core_text import FunctionalRefiner
from dataflow.prompts.prompt_template import NamedPlaceholderPromptTemplate
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
GCOT_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Task: Provide a detailed step-by-step reasoning (Chain-of-Thought) that explains "
    "how to arrive at this answer based on the image.\n"
    "Then, extract key nouns and objects mentioned in your reasoning that are "
    "visible in the image and can be spatially located.\n\n"
    "Format:\n"
    "Step 1: ...\n"
    "Step 2: ...\n"
    "Answer: {answer}\n"
    "Keywords: object1, object2\n"
)

DEFAULT_BBOX_PROMPT = 'Detect "{keyword}".'

def _parse_base(text: str) -> Dict[str, Any]:
    """基础解析逻辑（内部复用）"""
    if not text: return {"cot": "", "keywords": []}
    lines = text.split('\n')
    cot_lines = []
    keywords = []
    for line in lines:
        if line.strip().lower().startswith('keywords:'):
            keyword_str = line.split(':', 1)[-1].strip()
            raw_kws = [kw.strip().strip('.,;:!?"\'') for kw in keyword_str.replace(';', ',').split(',')]
            keywords = [k for k in raw_kws if k]
        else:
            cot_lines.append(line)
    return {"cot": '\n'.join(cot_lines).strip(), "keywords": keywords}

def extract_clean_cot_logic(text: str) -> str:
    """[For FunctionalRefiner] 仅返回清洗后的 CoT 文本"""
    return _parse_base(text)["cot"]

def extract_keywords_logic(text: str) -> List[str]:
    """[For FunctionalRefiner] 提取并合并关键词"""
    parsed = _parse_base(text)
    kws = parsed["keywords"]
    cot = parsed["cot"]
    
    if not kws or len(kws) <= 1:
        return kws
    
    # 简单的相邻合并逻辑
    cot_lower = cot.lower()
    merged = []
    skip_indices = set()
    for i in range(len(kws)):
        if i in skip_indices: continue
        best_match = kws[i]
        best_indices = [i]
        # 尝试向后合并 3 个词
        for j in range(i + 1, min(i + 4, len(kws))):
            if j in skip_indices: break
            combined = ' '.join(kws[i:j+1])
            if combined.lower() in cot_lower:
                best_match = combined
                best_indices = list(range(i, j+1))
            else: break
        merged.append(best_match)
        skip_indices.update(best_indices)
    return merged

def inject_bboxes_logic(cot_text: str, bbox_map: Dict[str, List[str]]) -> str:
    """[For FunctionalRefiner] 将 BBox 注入回 CoT"""
    if not cot_text or not bbox_map: return cot_text
    # 优先匹配长词
    sorted_keywords = sorted(bbox_map.keys(), key=lambda x: len(x), reverse=True)
    result_text = cot_text
    replaced = set()
    
    for keyword in sorted_keywords:
        if keyword in replaced: continue
        # 简单策略：只在 'Answer:' 之前注入，防止破坏答案区
        answer_pos = result_text.find('Answer:')
        search_limit = answer_pos if answer_pos != -1 else len(result_text)
        
        pos = result_text.lower().find(keyword.lower(), 0, search_limit)
        if pos == -1: continue
        
        boxes = bbox_map[keyword] # List[str]
        box_str = "".join(boxes)
        replacement = f"{keyword} {box_str}"
        
        result_text = result_text[:pos] + replacement + result_text[pos + len(keyword):]
        replaced.add(keyword)
    return result_text

class ImageGCoTPipeline:
    def __init__(
        self,
        *,
        first_entry_file: str,
        cache_path: str = "../cache/cache_gcot",
        file_name_prefix: str = "gcot",
        # Keys
        question_key: str = "question",
        answer_key: str = "answer",
        image_key: str = "image",
        output_key: str = "gcot",
        # Config
        vllm_max_tokens: int = 512
    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )

        self.vlm_serving = APIVLMServing_openai(
            api_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Any API platform compatible with OpenAI format
            model_name="gpt-4o-mini",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )
        
        self.keys = {
            "q": question_key,
            "a": answer_key,
            "img": image_key,
            "raw_cot": "raw_cot_output",
            "clean_cot": "cleaned_cot",
            "keywords": "extracted_keywords",
            "bbox_map": "bbox_mapping",
            "final": output_key
        }

        # ================== Operators ==================
        
        # 1. Generate CoT (通用 Generator)
        self.op_gen_cot = PromptTemplatedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant.",
            prompt_template=NamedPlaceholderPromptTemplate(template=GCOT_PROMPT_TEMPLATE)
        )
        
        # 2. Extract Clean CoT (通用 Refiner + Helper)
        self.op_extract_cot = FunctionalRefiner(func=extract_clean_cot_logic)
        
        # 3. Extract Keywords (通用 Refiner + Helper)
        self.op_extract_kws = FunctionalRefiner(func=extract_keywords_logic)

        # 4. Generate BBox (专用 Generator, 因为涉及行内 Batch)
        self.op_bbox_gen = VLMBBoxGenerator(
            serving=self.vlm_serving,
            prompt_template=DEFAULT_BBOX_PROMPT
        )
        
        # 5. Inject GCoT (通用 Refiner + Helper)
        self.op_inject = FunctionalRefiner(func=inject_bboxes_logic)

    def forward(self):
        print(">>> [Pipeline] Step 1: Generating CoT...")
        self.op_gen_cot.run(
            self.storage.step(),
            input_image_key=self.keys["img"],
            output_answer_key=self.keys["raw_cot"],
            question=self.keys["q"], # Template mapping
            answer=self.keys["a"]
        )
        
        print(">>> [Pipeline] Step 2: Parsing Outputs...")
        self.op_extract_cot.run(
            self.storage.step(),
            output_key=self.keys["clean_cot"],
            text=self.keys["raw_cot"] # Param mapping
        )
        self.op_extract_kws.run(
            self.storage.step(),
            output_key=self.keys["keywords"],
            text=self.keys["raw_cot"]
        )
        
        print(">>> [Pipeline] Step 3: Generating BBoxes (Grounding)...")
        self.op_bbox_gen.run(
            self.storage.step(),
            input_image_key=self.keys["img"],
            input_kws_key=self.keys["keywords"],
            output_key=self.keys["bbox_map"]
        )
        
        print(">>> [Pipeline] Step 4: Injecting GCoT...")
        self.op_inject.run(
            self.storage.step(),
            output_key=self.keys["final"],
            cot_text=self.keys["clean_cot"],
            bbox_map=self.keys["bbox_map"]
        )
        
        print(f">>> [Pipeline] Done. Final GCoT saved to: {self.keys['final']}")


if __name__ == "__main__":
    pipe = ImageGCoTPipeline(
        first_entry_file="../example_data/capsbench_images/image_gcot_demo.jsonl"
    )
    pipe.forward()
```
