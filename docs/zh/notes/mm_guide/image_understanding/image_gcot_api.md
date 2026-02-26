---
title: 图像定位思维链 (GCoT) 生成流水线（API版）
icon: mdi:image-text
createTime: 2026/01/11 20:44:55
permalink: /zh/mm_guide/image_gcot_api/
---
## 1. 概述

**图像定位思维链 (GCoT) 生成流水线** 旨在自动化生成**带视觉定位的思维链（Grounded Chain-of-Thought）**数据。该流水线通过多步推理，不仅生成回答问题的逻辑步骤，还将推理过程中提到的关键物体在图像中进行空间定位（Bounding Box），从而显著提升多模态数据的可解释性和精确度。

与传统方法不同，本流水线采用 **单一 VLM（如 GPT-5）** 同时完成“推理”和“定位”任务，流程更加精简高效。

我们支持以下应用场景：

* **增强型多模态数据构建**：为 VQA 数据集增加解释性和定位标注。
* **复杂场景理解**：生成包含物体坐标的详细推理步骤。
* **模型推理能力训练**：构建数据以训练模型“言之有物”，减少幻觉。

流水线的主要流程包括：

1. **CoT 生成**：模型生成分步推理文本，并提取关键名词。
2. **关键词解析**：从生成的文本中清洗并提取待定位的关键词。
3. **视觉定位 (Grounding)**：模型针对提取的关键词生成边界框 (BBox)。
4. **信息注入**：将 BBox 坐标回填至推理文本中，形成最终的 GCoT。

---

## 2. 快速开始

### 第一步：创建新的 DataFlow 工作文件夹
```bash
mkdir run_dataflow
cd run_dataflow
```

### 第二步：初始化 DataFlow-MM
```bash
dataflowmm init
```
这时你会看到：
```bash
gpu_pipelines/image_gcot_pipeline.py
```

### 第三步：下载示例数据
```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data
```

### 第四步：配置 API Key

在 `api_pipelines/image_gcot_api_pipeline.py` 中设置 API Key 环境变量：

```python
import os
os.environ["DF_API_KEY"] = "your_api_key"
```

### 第五步：配置参数

在 `api_pipelines/image_gcot_api_pipeline.py` 中配置 API 服务和输入数据路径：

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

### 第六步：一键运行
```bash
cd api_pipelines
python image_gcot_api_pipeline.py
```

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

该流程的输入数据通常是标准的 VQA 数据：

* **image**：图像文件路径。
* **question**：关于图像的问题。
* **answer**：问题的标准答案（用于辅助生成 CoT）。

**输入数据示例**：

```json
{
    "image":"../example_data/capsbench_images/0.png",
    "question":"Who is the lead actor in the movie \"Nightmare Alley\"?", 
    "answer": "Bradley Cooper."
}

```

### 2. **核心算子逻辑**

本流水线通过组合多个细粒度算子来实现复杂的 GCoT 生成逻辑：

#### A. **CoT 生成 (PromptTemplatedVQAGenerator)**

利用预设的 `GCOT_PROMPT_TEMPLATE`，引导模型生成“步骤化推理”和“关键词列表”。

* **Prompt 策略**：要求模型按 `Step 1: ...`, `Step 2: ...`, `Keywords: ...` 格式输出。
* **输出**：包含推理文本和关键词的原始字符串。

#### B. **文本清洗与提取 (FunctionalRefiner)**

使用自定义函数对上一步的输出进行解析：

* `extract_clean_cot_logic`：剥离关键词部分，保留纯净的 CoT 文本。
* `extract_keywords_logic`：解析 `Keywords:` 后的内容，生成 Python List。

#### C. **视觉定位 (VLMBBoxGenerator)**

针对提取出的每一个关键词，调用 VLM 的定位能力生成边界框。

* **输入**：图像 + 关键词列表。
* **输出**：关键词到边界框坐标的映射字典 (Map)。

#### D. **坐标注入 (FunctionalRefiner)**

使用 `inject_bboxes_logic` 函数，将生成的 BBox 坐标智能插入回原始 CoT 文本中对应的单词之后。

### 3. **输出数据**

最终，流水线生成的输出数据将包含以下关键字段：

* **raw_cot_output**：模型原始生成的文本。
* **cleaned_cot**：清洗后的纯推理文本。
* **bbox_mapping**：关键词与其坐标的映射。
* **gcot**：最终结果，包含坐标信息的推理链。

**输出数据示例 (gcot 字段)**：

```text
Step 1: Analyze the text visible in the image, which includes a list of actors beneath the title of the movie \"Nightmare Alley.\"\n\nStep 2: Identify the names listed. The first name listed is \"Bradley Cooper,\" indicating he is prominent in the film.\n\nStep 3: Recognize that the image is a promotional poster for \"Nightmare Alley,\" suggesting the individuals mentioned are likely key cast members.\n\nStep 4: Confirm that Bradley Cooper is identified as the lead actor based on his position at the top of the cast list.\n\nAnswer: Bradley Cooper.  \nKeywords: Nightmare Alley, cast list, poster.","cleaned_cot":"Step 1: Analyze the text visible in the image, which includes a list of actors beneath the title of the movie \"Nightmare Alley.\"\n\nStep 2: Identify the names listed. The first name listed is \"Bradley Cooper,\" indicating he is prominent in the film.\n\nStep 3: Recognize that the image is a promotional poster for \"Nightmare Alley,\" suggesting the individuals mentioned are likely key cast members.\n\nStep 4: Confirm that Bradley Cooper is identified as the lead actor based on his position at the top of the cast list.\n\nAnswer: Bradley Cooper.","extracted_keywords":["Nightmare Alley","cast list","poster"],"bbox_mapping":{},"gcot":"Step 1: Analyze the text visible in the image, which includes a list of actors beneath the title of the movie \"Nightmare Alley.\"\n\nStep 2: Identify the names listed. The first name listed is \"Bradley Cooper,\" indicating he is prominent in the film.\n\nStep 3: Recognize that the image is a promotional poster for \"Nightmare Alley,\" suggesting the individuals mentioned are likely key cast members.\n\nStep 4: Confirm that Bradley Cooper is identified as the lead actor based on his position at the top of the cast list.\n\nAnswer: Bradley Cooper.

```

---

## 4. 流水线示例

以下是完整的 `ImageGCoTAPIPipeline` 代码实现。

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
