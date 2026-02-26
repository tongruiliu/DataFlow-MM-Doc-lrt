---
title: ScaleCap High-Density Caption Pipeline (API version)
icon: mdi:image-text
createTime: 2026/01/11 22:08:57
permalink: /en/mm_guide/image_scale_caption_pipeline_api/
---

## 1. Overview

The **Image Scale Caption Pipeline (ScaleCap)** is an advanced image captioning solution based on a **"Generate-Verify-Expand-Integrate"** paradigm. This pipeline is designed to generate image descriptions with **extremely high information density** and **ultra-low hallucination rates**, making it particularly suitable for scenarios requiring deep understanding of image details.

The theoretical foundation of this method is derived from the paper *ScaleCap: Inference-Time Scalable Image Captioning via Dual-Modality Debiasing*. It gradually uncovers object and spatial details through multi-turn dialogue and visual grounding, effectively filtering out hallucinations produced by the model.

We support the following application scenarios:

* **High-Quality Multimodal Dataset Construction**: Generating training data that is more detailed and accurate than standard captions.
* **Fine-Grained Image Retrieval**: Providing highly detailed text for indexing.
* **Blind Assistance / Image Accessibility**: Generating "what-you-see-is-what-you-get" detailed narrations.

The main process of the pipeline includes:

1. **Initial Caption Generation**: The VLM generates a basic description.
2. **Visual Debiasing**: The description is split into sentences, and each is verified against visual evidence (Visual Grounding).
3. **Detail Questioning**: Targeted questions regarding object attributes and spatial relations are generated based on the verified "Golden Sentences".
4. **Answering & Secondary Verification**: The VLM answers the detail questions, followed by another round of visual grounding to filter out incorrect details.
5. **Final Integration**: All verified information is woven into a coherent, comprehensive long caption.

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
api_pipelines/image_scale_caption_api_pipeline.py

```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data

```

### Step 4: Configure API Key

Set your API Key environment variable in `api_pipelines/image_scale_caption_api_pipeline.py`:

```python
import os
os.environ["DF_API_KEY"] = "your_api_key"

```

### Step 5: Configure Parameters

Configure the API service and input data paths in `api_pipelines/image_scale_caption_api_pipeline.py`:

```python
    def __init__(
        self,
        # Storage params
        first_entry_file: str = "../example_data/capsbench_images/image_scale_caption_demo.jsonl",
        cache_path: str = "../cache/image_scale_caption",
        file_name_prefix: str = "scalecap",
        cache_type: str = "jsonl",
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_caption",
    ):

```

```python
        self.vlm_serving = APIVLMServing_openai(
            api_url="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)", # Any API platform compatible with OpenAI format
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
python image_scale_caption_api_pipeline.py

```

---

## 3. Data Flow & Logic

### 1. **Input Data**

The input data for this process is very simple, requiring only the image path:

* **image**: Path to the image file.

**Input Data Example**:

```json
{
    "image": "../example_data/capsbench_images/0.png"
}

```

### 2. **Core Operator Logic**

This pipeline orchestrates multiple fine-grained operators to achieve the complex ScaleCap logic:

#### A. **Initial Generation (PromptedVQAGenerator)**

* **Function**: Uses a basic prompt to generate a preliminary description of the image (`init_caption`).

#### B. **Visual Debiasing (VisualGroundingRefiner)**

* **Function**: The core anti-hallucination mechanism of ScaleCap.
* **Logic**:
1. Uses `split_sentences` to break the initial draft into single sentences.
2. Asks the VLM: "Given the image, is the description '{text}' directly supported by visual evidence?".
3. Retains only the sentences that receive a "Yes", forming **"Golden Sentences"**.



#### C. **Question Generation & Parsing (PromptTemplatedQAGenerator)**

* **Function**: Uses LLM capabilities to generate targeted follow-up questions based on the Golden Sentences.
* **Logic**: The model generates text like "Describe more details about the [Object]". The `parse_questions_logic` function automatically expands these into two categories: **object details** and **spatial relationships**.

#### D. **Batch Answering & Secondary Filtering (BatchVQAGenerator & Refiner)**

* **Function**: Deeply mines visual information.
* **Logic**:
1. Uses `BatchVQAGenerator` to have the VLM answer all generated questions in a single batch.
2. Uses `VisualGroundingRefiner` again to verify if these newly generated details are accurate.
3. Retains reliable details (`final_details`).



#### E. **Final Integration (PromptTemplatedQAGenerator)**

* **Function**: Rewrites the "Golden Sentences" and "Verified Details" into a fluent, cohesive text.
* **Output**: `final_caption`.

### 3. **Output Data**

The output data records the entire pipeline process for easy debugging and analysis:

* **init_caption**: The original initial draft.
* **golden_sentences**: List of sentences that passed the first debiasing check.
* **q_list**: List of generated follow-up questions.
* **final_details**: Detailed answers that passed the secondary check.
* **final_caption**: The final high-density description.

**Output Data Example**:

```json
{
    "image": "../example_data/capsbench_images/0.png",
    "init_caption": "A dog sitting on a bench.",
    "golden_sentences": ["A dog is sitting on a wooden bench."],
    "q_list": ["Describe more details about the dog.", "Describe more details about the position of the bench."],
    "final_details": ["The dog is a Golden Retriever with a red collar.", "The bench is located in a park."],
    "final_caption": "A Golden Retriever with a red collar is sitting on a wooden bench located in a park."
}

```

---

## 4. Pipeline Example

Below is the complete `ImageScaleCaptionPipeline` code implementation (API Version).

```python
import os
os.environ["DF_API_KEY"] = "sk-xxxx"


import re
import argparse
from typing import Callable, Any, List

from dataflow.utils.storage import FileStorage

from dataflow.prompts.prompt_template import NamedPlaceholderPromptTemplate
from dataflow.prompts.image import ImageScaleCaptionPrompt

from dataflow.operators.core_vision import PromptedVQAGenerator, BatchVQAGenerator, VisualGroundingRefiner
from dataflow.operators.core_text import PromptTemplatedQAGenerator, FunctionalRefiner
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai

def split_sentences(text: str) -> List[str]:
    """将文本拆分为句子列表"""
    if not text or not isinstance(text, str):
        return []
    # 使用正则按标点符号分割 (. ! ? 。 ！ ？)
    _SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])

def join_list(data: Any, separator: str = "\n") -> str:
    """将列表连接为字符串"""
    if isinstance(data, list):
        # 过滤掉非字符串元素或空字符串
        valid_items = [str(x) for x in data if x]
        return separator.join(valid_items)
    return str(data) if data is not None else ""

def parse_questions_logic(text: str, max_q: int = 20) -> List[str]:
    """
    解析 LLM 生成的 "Describe more details about..." 文本，
    并自动扩展 position 问题。
    """
    if not text or not isinstance(text, str):
        return []

    lines = [t.strip() for t in text.split("\n") if t.strip()]
    obj_qs = []
    
    for line in lines:
        # 提取包含 "Describe more details about" 的行
        if "Describe more details about" in line:
            # 去除可能的序号 (如 "1. Describe...")
            try:
                start_idx = line.find("Describe")
                clean = line[start_idx:]
                # 去除句末多余内容，保留到第一个句号
                if "." in clean:
                    clean = clean.split(".")[0] + "."
                obj_qs.append(clean)
            except Exception:
                continue
    
    # 去重并保持顺序
    seen = set()
    unique_obj_qs = []
    for q in obj_qs:
        if q not in seen:
            unique_obj_qs.append(q)
            seen.add(q)
    
    # 截断
    unique_obj_qs = unique_obj_qs[:max_q]
    
    # 扩展 Position 问题
    pos_qs = [
        q.replace("Describe more details about", "Describe more details about the position of")
        for q in unique_obj_qs
    ]
    
    # 返回合并后的列表 (对象问题 + 位置问题)
    return unique_obj_qs + pos_qs


class ImageScaleCaptionPipeline:
    def __init__(
        self,
        # Storage params
        first_entry_file: str = "images.jsonl",
        cache_path: str = "./cache_scalecap",
        file_name_prefix: str = "scalecap",
        cache_type: str = "jsonl",
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_caption",
        # VLLM Config
        vllm_tensor_parallel_size: int = 1,
        vllm_temperature: float = 0.7,
        vllm_top_p: float = 0.9,
        vllm_max_tokens: int = 512,
    ):
        # 1. Storage
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # 2. Serving
        self.vlm_serving = APIVLMServing_openai(
            api_url="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)", # Any API platform compatible with OpenAI format
            model_name="gpt-4o-mini",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )

        # 3. Prompts
        self.prompts_db = ImageScaleCaptionPrompt().build_prompt()

        # 4. Keys
        self.input_image_key = input_image_key
        self.output_key = output_key

        # ================== Operator Initialization ==================

        # --- Step A: Generate Init Caption ---
        # 构造固定 Prompt 列
        self.refine_const_prompt = FunctionalRefiner(func=lambda: self.prompts_db["VLM_PROMPT_1"])
        
        # 生成初稿 (使用通用 PromptedVQAGenerator)
        self.gen_init_caption = PromptedVQAGenerator(
            serving=self.vlm_serving,
            system_prompt="You are a helpful assistant."
        )

        # --- Step B: Refine Golden Sentences ---
        # 分句
        self.refine_split = FunctionalRefiner(func=split_sentences)
        
        # 视觉自检 (保留 Yes 的句子)
        self.refine_golden = VisualGroundingRefiner(
            serving=self.vlm_serving,
            prompt_template="Given the image, is the description '{text}' directly supported by visual evidence? Answer strictly yes or no."
        )

        # --- Step C: Generate Questions ---
        # 列表转字符串
        self.refine_join = FunctionalRefiner(func=join_list)
        
        # 文本生成问题 (Text-to-Text)
        tpl_q = NamedPlaceholderPromptTemplate(
            template=self.prompts_db["LLM_PROMPT_1"], 
            join_list_with="\n"
        )
        self.gen_questions_text = PromptTemplatedQAGenerator(
            serving=self.vlm_serving,
            prompt_template=tpl_q
        )
        
        # 解析问题文本为列表
        self.refine_parse_qs = FunctionalRefiner(func=parse_questions_logic)

        # --- Step D: Generate Answers ---
        # 批量回答 (One Image -> Many Qs)
        self.gen_answers = BatchVQAGenerator(serving=self.vlm_serving)
        
        # 回答过滤
        self.refine_answers = VisualGroundingRefiner(
            serving=self.vlm_serving,
            prompt_template="Given the image, is the statement '{text}' grounded in the image and not generic? Answer strictly yes or no."
        )

        # --- Step E: Integrate Final Caption ---
        # 融合 (Text-to-Text)
        tpl_final = NamedPlaceholderPromptTemplate(
            template=self.prompts_db["LLM_PROMPT_4"], 
            join_list_with="\n"
        )
        self.gen_final_caption = PromptTemplatedQAGenerator(
            serving=self.vlm_serving,
            prompt_template=tpl_final
        )

    def forward(self):
        print(">>> [Pipeline] Step 0: Preparing Prompts...")
        # 构造 init_prompt 列
        self.refine_const_prompt.run(
            self.storage.step(), 
            output_key="init_prompt"
        )

        print(">>> [Pipeline] Step 1: Generating Initial Caption...")
        self.gen_init_caption.run(
            self.storage.step(),
            input_prompt_key="init_prompt",
            input_image_key=self.input_image_key,
            output_answer_key="init_caption"
        )

        print(">>> [Pipeline] Step 2: Refining Golden Sentences...")
        self.refine_split.run(
            self.storage.step(), 
            output_key="sentences", 
            text="init_caption"
        )
        self.refine_golden.run(
            self.storage.step(), 
            input_list_key="sentences", 
            input_image_key=self.input_image_key, 
            output_key="golden_sentences"
        )

        print(">>> [Pipeline] Step 3: Generating Details Questions...")
        self.refine_join.run(
            self.storage.step(), 
            output_key="golden_str", 
            data="golden_sentences"
        )
        
        # template: "{sentence}" -> map to col "golden_str"
        self.gen_questions_text.run(
            self.storage.step(), 
            output_answer_key="raw_q_text", 
            sentence="golden_str"
        )
        
        self.refine_parse_qs.run(
            self.storage.step(), 
            output_key="q_list", 
            text="raw_q_text"
        )

        print(">>> [Pipeline] Step 4: Generating & Filtering Answers...")
        self.gen_answers.run(
            self.storage.step(), 
            input_prompts_key="q_list", 
            input_image_key=self.input_image_key, 
            output_key="raw_answers"
        )
        
        self.refine_answers.run(
            self.storage.step(), 
            input_list_key="raw_answers", 
            input_image_key=self.input_image_key, 
            output_key="final_details"
        )

        print(">>> [Pipeline] Step 5: Integrating Final Caption...")
        self.refine_join.run(
            self.storage.step(), 
            output_key="details_str", 
            data="final_details"
        )
        
        # template keys: context, object_info, position_info
        self.gen_final_caption.run(
            self.storage.step(),
            output_answer_key=self.output_key,
            context="golden_str",
            object_info="details_str",
            position_info="details_str" # 简化：同时作为 object 和 position 信息
        )

        print(f">>> [Pipeline] All Done. Result saved to: {self.storage.cache_path}")


if __name__ == "__main__":

    pipe = ImageScaleCaptionPipeline( 
        first_entry_file="../example_data/capsbench_images/image_scale_caption_demo.jsonl",
        cache_path="../cache/image_scale_caption",
        file_name_prefix="scalecap",
        input_image_key="image",
        output_key="final_caption",
        vllm_tensor_parallel_size=1,
        vllm_max_tokens=1024
    )
    
    pipe.forward()

```
