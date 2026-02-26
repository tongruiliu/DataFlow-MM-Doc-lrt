---
title: Visual-Only MCQ Pipeline
createTime: 2026/01/11 22:13:45
icon: mdi:image-text
permalink: /en/mm_guide/image_visual_only_mcq_pipeline/
---

## 1. Overview

The **Visual-Only MCQ Pipeline** is a core component within the CapRL (Caption Reinforcement Learning) framework. Its goal is to generate a set of high-quality Multiple-Choice Questions (MCQs) that strictly satisfy **strong visual dependency**: the model must "see" the image to answer correctly, and cannot rely merely on text guessing or common sense.

This pipeline utilizes a **"Generate-Parse-Verify"** three-step approach, employing **Option Rotation** and **Blind Test (Text-Only)** mechanisms to rigorously filter out model hallucinations or overly simple questions. The generated questions can be used as reward signals (Reward Model) for reinforcement learning.

The main process includes:

1. **MCQ Generation**: The VLM generates raw Question-Answer text blocks based on the image.
2. **Structured Parsing**: Uses regex logic to parse the raw text into standard question and option structures.
3. **Visual Dependency Verification**:
   * **Rotation Test**: Randomly shuffles the order of options multiple times to eliminate positional bias.
   * **Dual Filtering**: Requires a high "Visual Accuracy" (with image) and a low "Textual Accuracy" (without image).

---

## 2. Quick Start

### Step 1: Create a New DataFlow Working Directory

```bash
mkdir run_vis_mcq
cd run_vis_mcq

```

### Step 2: Initialize DataFlow-MM

```bash
dataflowmm init

```

You will then see:

```bash
gpu_pipelines/image_visual_only_mcq_pipeline.py

```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data

```

### Step 4: Configure Parameters

Configure the model path and filtering thresholds (e.g., requiring 100% visual accuracy and less than 25% textual accuracy):

```python
if __name__ == "__main__":
    pipe = VisualOnlyMCQPipeline(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        first_entry_file="../example_data/capsbench_images/image_visual_only_mcq_demo.jsonl",
        hf_cache_dir="~/.cache/huggingface",
        download_dir="../ckpt/models/Qwen2.5-VL-3B-Instruct",
        rotate_num=4,
        pass_visual_min=1.0,
        pass_textual_max=0.25
    )
    pipe.forward()

```

> **⚠️ Important Note on Model Path Configuration (Taking `Qwen2.5-VL-3B-Instruct` as an example):**
> * **If you have already downloaded the model files:** Please change `model_path` to your local model path. **Crucially**, ensure that the model folder is named exactly `Qwen2.5-VL-3B-Instruct`; otherwise, the framework will fail to recognize it.
> * **If you haven't downloaded the model yet:** You must specify a `download_dir` parameter that ends with `Qwen2.5-VL-3B-Instruct` (as shown in the default parameters). Failure to do so will also result in the model not being recognized after downloading.
> 
> 

### Step 5: Run

```bash
cd gpu_pipelines
python image_visual_only_mcq_pipeline.py

```

> **🛠️ Troubleshooting**
> **Issue 1:** If you encounter a CUDA library conflict error similar to the following:
> `ImportError: .../miniconda3/envs/Dataflow-MM/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`
> **Solution:** This is usually caused by conflicting environment variables. Run the script with an empty `LD_LIBRARY_PATH`:
> ```bash
> LD_LIBRARY_PATH="" python image_visual_only_mcq_pipeline.py
> 
> ```
> 
> 
> **Issue 2:** If you are using **Qwen series models** and encounter the following error:
> `KeyError: "Missing required keys in rope_scaling for 'rope_type'='None': {'rope_type'}"`
> **Solution:** Open the `config.json` file located in your model folder, find the `rope_scaling` section, and change the key `"type"` to `"rope_type"`.
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
> 
> ```
> 
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
> 
> ```
> 
> 

---

## 3. Data Flow & Logic

### 1. **Input Data**

The input data only requires the image path:

* **image**: Path to the image file.

**Input Data Example**:

```json
{
    "image": "./images/sample_01.jpg"
}

```

### 2. **Core Operator Logic**

This pipeline is chained together by three key operators:

#### A. **Raw Generation (FixPromptedVQAGenerator)**

* **Function**: Uses the preset CapRL prompt templates (`SYS_PROMPT_MCQ` / `USER_PROMPT_MCQ`) to instruct the VLM to generate 5 MCQs in one go.
* **Output**: Unstructured text blocks containing multiple `#### Question` headers and options.

#### B. **Structured Parsing (FunctionalRefiner)**

* **Logic Function**: `parse_mcq_text_logic`
* **Function**: Extracts the questions, options (A-F), and correct answers from the raw text using regular expressions.
* **Output**: A structured list of MCQs (`parsed_mcq_list`).

#### C. **Dependency Verification (VisualDependencyRefiner)**

This is the core filter of the pipeline. It performs N inferences (N = `rotate_num`) for each question:

1. **Option Rotation**: Randomly shuffles the option order (e.g., moving the answer from A to C) to prevent the model from cheating by "always choosing A".
2. **Visual Pass**: Inputs Image + Question. Records the proportion of correct answers.
3. **Textual Pass (Blind Test)**: Inputs Question only (No Image). Records the proportion of correct blind guesses.
4. **Filtering Criteria**:
* Retains the question if and only if: `Visual_Acc >= pass_visual_min` **AND** `Textual_Acc <= pass_textual_max`.
* *Example*: If a question can be answered correctly without looking at the image (high textual accuracy), it relies on common sense rather than visual info, and is **discarded**.



### 3. **Output Data**

The output data (`final_mcqs`) only contains questions that have passed the rigorous verification. These questions possess extremely high quality and visual relevance.

**Output Data Example**:

```json
{
    "image": "./images/sample_01.jpg",
    "final_mcqs": [
        {
            "question": "What is the color of the car on the far left?\n - A) Red\n - B) Blue...",
            "answer": "A",
            "stats": {
                "visual_acc": 1.0,  
                "text_acc": 0.0     
            }
        }
    ]
}

```

---

## 4. Pipeline Example

Below is the complete `VisualOnlyMCQPipeline` code implementation (GPU Version).

```python
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import FixPromptedVQAGenerator, VisualDependencyRefiner
from dataflow.operators.core_text import FunctionalRefiner
from dataflow.prompts.image import ImageCaprlPrompt

import re
from typing import List, Dict, Any

_Q_BLOCK_SPLIT = re.compile(r"^####\s*\d+\.\s*\*\*(.*?)\*\*\s*$", re.M)
_OPT_LINE_RE = re.compile(r"^\s*-\s*([A-F])\)\s*(.+?)\s*$")
_ANS_LINE_RE = re.compile(r"^\s*\*\*Answer:\*\*\s*([A-F])\)\s*(.+?)\s*$", re.I)

def parse_mcq_text_logic(mcq_text: str, expected: int = 5) -> List[Dict[str, Any]]:
    if not mcq_text or not isinstance(mcq_text, str): return []
    
    indices = [m.start() for m in _Q_BLOCK_SPLIT.finditer(mcq_text)]
    if not indices: return []
    indices.append(len(mcq_text))
    blocks = [mcq_text[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1)]
    
    parsed = []
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        q_title_m = _Q_BLOCK_SPLIT.search(block)
        if not q_title_m: continue
        
        q_title = q_title_m.group(1).strip()
        options = {}
        ans_letter, ans_text = None, None
        
        for ln in lines:
            m_opt = _OPT_LINE_RE.match(ln)
            if m_opt:
                options[m_opt.group(1)] = m_opt.group(2).strip()
                continue
            m_ans = _ANS_LINE_RE.match(ln)
            if m_ans:
                ans_letter = m_ans.group(1).upper()
                ans_text = m_ans.group(2).strip()
                break
        
        if options and ans_letter and ans_letter in options:
            q_lines = [q_title]
            for lbl in ["A", "B", "C", "D", "E", "F"]:
                if lbl in options:
                    q_lines.append(f"   - {lbl}) {options[lbl]}")
            
            parsed.append({
                "question": "\n".join(q_lines),
                "question_title": q_title,
                "options": options,
                "answer": ans_letter,
                "answer_text": ans_text
            })
            
    if expected > 0:
        parsed = parsed[:expected]
        
    uniq = []
    seen = set()
    for it in parsed:
        key = (it["question_title"], it["answer"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq


class VisualOnlyMCQPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        first_entry_file: str,
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
        cache_path: str = "../cache/cache_mcq",
        file_name_prefix: str = "vis_mcq",
        # Config
        rotate_num: int = 4,
        pass_visual_min: float = 1.0,
        pass_textual_max: float = 0.25,
        add_none_above: bool = True,
        # Keys
        input_image_key: str = "image",
        output_key: str = "final_mcqs",
        # VLLM
        device: str = "cuda",
        vllm_max_tokens: int = 2048
    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )
        
        self.serving = LocalModelVLMServing_vllm(
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=download_dir,
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=1,
            vllm_temperature=0.1, 
            vllm_max_tokens=vllm_max_tokens
        )
        
        # Keys
        self.keys = {
            "img": input_image_key,
            "raw_text": "raw_mcq_text",
            "parsed_list": "parsed_mcq_list",
            "final": output_key
        }
        
        # --- Prompts ---
        self.prompts_db = ImageCaprlPrompt().build_prompt()

        # ================== Operators ==================
        
        # 1. Generate Raw MCQs (FixPromptedVQAGenerator)
        # 直接使用 prompt 类中的字符串
        self.op_gen_raw = FixPromptedVQAGenerator(
            serving=self.serving,
            system_prompt=self.prompts_db["SYS_PROMPT_MCQ"],
            user_prompt=self.prompts_db["USER_PROMPT_MCQ"]
        )
        
        # 2. Parse MCQs (Refine)
        self.op_parse = FunctionalRefiner(func=parse_mcq_text_logic)
        
        # 3. Verify Visual Dependency (Refine)
        # 传入 prompt 模板
        self.op_verify = VisualDependencyRefiner(
            serving=self.serving,
            instruction_template=self.prompts_db["ANSWER_INSTRUCTION"],
            rotate_num=rotate_num,
            pass_visual_min=pass_visual_min,
            pass_textual_max=pass_textual_max,
            add_none_above_visual=add_none_above
        )

    def forward(self):
        print(">>> [Pipeline] Step 1: Generating Raw MCQs (FixPrompted)...")
        self.op_gen_raw.run(
            self.storage.step(),
            input_image_key=self.keys["img"],
            output_answer_key=self.keys["raw_text"]
        )
        
        print(">>> [Pipeline] Step 2: Parsing MCQs...")
        self.op_parse.run(
            self.storage.step(),
            output_key=self.keys["parsed_list"],
            mcq_text=self.keys["raw_text"], 
            expected=5
        )
        
        print(">>> [Pipeline] Step 3: Verifying Visual Dependency (Rotation Check)...")
        self.op_verify.run(
            self.storage.step(),
            input_list_key=self.keys["parsed_list"],
            input_image_key=self.keys["img"],
            output_key=self.keys["final"]
        )
        
        print(f">>> [Pipeline] Done. Results in: {self.keys['final']}")

if __name__ == "__main__":
    pipe = VisualOnlyMCQPipeline(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        first_entry_file="../example_data/capsbench_images/image_visual_only_mcq_demo.jsonl",
        hf_cache_dir="~/.cache/huggingface",
        download_dir="../ckpt/models/Qwen2.5-VL-3B-Instruct",
        rotate_num=4,
        pass_visual_min=1.0,
        pass_textual_max=0.25
    )
    pipe.forward()

```
