---
title: Vision MCTS Reasoning Pipeline (API version)
icon: mdi:image-text
createTime: 2026/01/11 21:59:59
permalink: /en/mm_guide/vision_mct_reasoning_pipeline_api/
---

## 1. Overview

The **Vision MCTS Reasoning Pipeline** is designed to build high-quality **Process Supervision Data** for multimodal large models. This pipeline handles two sources of data: existing Monte Carlo Tree Search (MCTS) trajectory data, or generating new reasoning chains directly using a VLM.

This pipeline is a core tool for **Grounded-RL** and **SFT Data Construction**. It "linearizes" complex tree-like search processes into a `<think>...</think><answer>...</answer>` format that the model can learn from.

We support the following application scenarios:

* **Data Extraction from MCTS Trees**: Converts high-value paths (Rollouts) in the search tree into linear training data.
* **Hybrid Data Construction**: Automatically falls back to using the VLM for CoT generation for samples without a search tree.
* **Spatial Reasoning Enhancement**: Supports generating spatial reasoning chains that include explicit coordinates (Bounding Boxes).

The main process of the pipeline includes:

1. **MCTS Tree Parsing**: Parses the search tree structure in the input data and extracts successful reasoning paths.
2. **Visual Reasoning Generation (Fallback)**: For samples with missing tree structures or failed parsing, the VLM is used to regenerate the reasoning chain.
3. **Data Standardization**: Outputs reasoning chain data in a unified format.

---

## 2. Quick Start

### Step 1: Create a New DataFlow Working Directory

```bash
mkdir run_mcts_reasoning
cd run_mcts_reasoning

```

### Step 2: Initialize DataFlow-MM

```bash
dataflowmm init

```

You will then see:

```bash
api_pipelines/vision_mcts_api_pipeline.py

```

### Step 3: Download Sample Data

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data

```

### Step 4: Configure API Key

Set your API Key environment variable in `api_pipelines/vision_mcts_api_pipeline.py`:

```python
import os
os.environ["DF_API_KEY"] = "your_api_key"

```

### Step 5: Configure Parameters

Configure the API service and input data paths in `api_pipelines/vision_mcts_api_pipeline.py`. Ensure the input file (jsonl) contains a `tree` field (for extraction) or just `question`/`image` (for generation).

```python
    pipe = VisionMCTSReasoningPipeline(
        first_entry_file="../example_data/capsbench_images/visual_mct_reasoning_demo.jsonl",
        prompt_type="spatial",
    )

```

### Step 6: Run with One Command

```bash
cd api_pipelines
python vision_mcts_api_pipeline.py

```

---

## 3. Data Flow & Logic

### 1. **Input Data**

Input data typically originates from MCTS search process logs, or unannotated image-text pairs:

* **image**: Path to the image.
* **question**: The visual question.
* **tree** (Optional): JSON structure of the MCTS search tree, containing node Values, Visits, and Actions.

**Input Data Example**:

```json
{
    "image": "./images/puzzle.jpg",
    "question": "What is the next step to solve this?",
    "tree": { "root": { "children": [...], "value": 1.0, "text": "Step 1..." } }
}

```

### 2. **Core Operator Logic**

This pipeline uses a hybrid strategy of **"Extraction First, Generation as Fallback"**:

#### A. **MCTSTreeRefiner (Tree Structure Parser)**

This operator handles the `tree` field. It traverses the tree structure and filters out the best path from the root node to a leaf node based on the node's Q-value.

* **Input**: `tree` object.
* **Function**: Linearizes tree paths, filtering out low-value or incomplete search branches.
* **Output**: A list of extracted reasoning chains (`mcts_chains`).

#### B. **VisualReasoningGenerator (Visual Reasoning Generator)**

This operator is the "generation engine" of the pipeline. It receives the extraction result from the previous step as input.

* **Mechanism**: Checks `input_existing_chains_key` (i.e., `mcts_chains`).
* If MCTS parsing is successful (chain exists), it is reused directly without inference (saving computational resources).
* If the MCTS chain is empty (tree does not exist or parsing failed), it calls the VLM to generate the reasoning chain from scratch based on `prompt_type` (e.g., `spatial`).


* **Prompt Types**: Supports modes like `spatial` (spatial coordinate reasoning) and `logical` (logical reasoning).

### 3. **Output Data**

The finally generated output data (`final_reasoning_chains`) will contain high-quality chains of thought that can be directly used for SFT training.

**Output Example**:

```json
{
    "image": "./images/puzzle.jpg",
    "final_reasoning_chains": [
        "<think>First, locate the red block at [100, 200]. To solve the puzzle, it needs to move right...</think><answer>Move Red Block</answer>"
    ]
}

```

---

## 4. Pipeline Example

Below is the complete `VisionMCTSReasoningPipeline` code implementation (API Version).

```python
import os
os.environ["DF_API_KEY"] = "sk-xxxx"
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

# 引入原子算子
from dataflow.operators.core_text import MCTSTreeRefiner
from dataflow.operators.core_vision import VisualReasoningGenerator
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai

class VisionMCTSReasoningPipeline:
    def __init__(
        self,
        first_entry_file: str,
        cache_path: str = "../cache/cache_mcts",
        file_name_prefix: str = "mcts_reason",
        # Config
        prompt_type: str = "spatial",
        max_samples_per_file: int = 10000,
        # Keys
        input_question_key: str = "question",
        input_image_key: str = "image",
        input_tree_key: str = "tree",
        output_key: str = "final_reasoning_chains",

    ):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )
        
        self.vlm_serving = APIVLMServing_openai(
            api_url="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)", # Any API platform compatible with OpenAI format
            model_name="gpt-4o-mini",
            image_io=None,
            send_request_stream=False,
            max_workers=10,
            timeout=1800
        )
        
        self.keys = {
            "q": input_question_key,
            "img": input_image_key,
            "tree": input_tree_key,
            "mcts_chains": "mcts_extracted_chains",
            "final": output_key
        }

        # ================== Operators ==================
        
        # 1. Refiner: MCTS -> Chains
        self.op_mcts_refine = MCTSTreeRefiner(
            max_chains_per_sample=max_samples_per_file
        )
        
        # 2. Generator: VLM -> Chains (Fallback)
        self.op_vlm_gen = VisualReasoningGenerator(
            serving=self.vlm_serving,
            prompt_type=prompt_type
        )

    def forward(self):
        print(">>> [Pipeline] Step 1: Extracting Chains from MCTS Trees...")
        self.op_mcts_refine.run(
            self.storage.step(),
            input_tree_key=self.keys["tree"],
            output_key=self.keys["mcts_chains"]
        )
        
        print(">>> [Pipeline] Step 2: Generating Chains via VLM (Fallback)...")
        # 将 mcts_chains 作为 input_existing_chains_key 传入
        # 如果 MCTS 解析成功，则复用；否则调用 VLM 生成
        self.op_vlm_gen.run(
            self.storage.step(),
            input_question_key=self.keys["q"],
            input_image_key=self.keys["img"],
            input_existing_chains_key=self.keys["mcts_chains"],
            output_key=self.keys["final"]
        )
        
        
if __name__ == "__main__":
    pipe = VisionMCTSReasoningPipeline(
        first_entry_file="../example_data/capsbench_images/visual_mct_reasoning_demo.jsonl",
        prompt_type="spatial",
    )
    pipe.forward()

```
