---
title: 视觉依赖 MCQ 生成流水线（API版）
createTime: 2026/01/11 22:13:45
icon: mdi:image-text
permalink: /zh/mm_guide/image_visual_only_mcq_pipeline_api/
---
## 1. 概述

**视觉依赖 MCQ 生成流水线 (Visual-Only MCQ Pipeline)** 是 CapRL (Caption Reinforcement Learning) 框架中的核心组件。它的目标是生成一组高质量的多项选择题 (MCQ)，且这些题目必须满足**强视觉依赖性**：即模型必须“看”图才能答对，仅凭文本（猜题或常识）无法作答。

该流水线通过**生成-解析-验证**三步法，利用**选项旋转 (Rotation)** 和**无图盲测 (Blind Test)** 机制，严格过滤掉模型幻觉或过于简单的题目。生成的题目可作为强化学习的奖励信号（Reward Model）。

主要流程包括：

1. **MCQ 生成**：VLM 基于图像生成原始的问答对文本。
2. **结构化解析**：利用正则逻辑将文本解析为标准的题目与选项结构。
3. **视觉依赖验证**：
* **旋转测试**：多次打乱选项顺序，消除位置偏见。
* **双重过滤**：要求“有图答对率”高，“无图答对率”低。

---

## 2. 快速开始

### 第一步：创建工作目录

```bash
mkdir run_vis_mcq
cd run_vis_mcq

```

### 第二步：初始化 DataFlow-MM

```bash
dataflowmm init

```

这时你会看到：

```bash
api_pipelines/image_visual_only_mcq_api_pipeline.py

```

### 第三步：下载示例数据

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data

```

### 第四步：配置 API Key

在 `api_pipelines/image_visual_only_mcq_api_pipeline.py` 中设置 API Key 环境变量：

```python
import os
os.environ["DF_API_KEY"] = "your_api_key"

```

### 第五步：配置参数

在 `api_pipelines/image_visual_only_mcq_api_pipeline.py` 中配置过滤阈值，例如，要求有图 100% 正确，无图正确率低于 25%：

```python
    pipe = VisualOnlyMCQPipeline(
        first_entry_file="../example_data/capsbench_images/image_visual_only_mcq_demo.jsonl",
        rotate_num=4,
        pass_visual_min=1.0,
        pass_textual_max=0.25
    )

```

### 第六步：一键运行

```bash
cd api_pipelines
python image_visual_only_mcq_api_pipeline.py

```

---

## 3. 数据流与流水线逻辑

### 1. **输入数据**

输入仅需包含图像路径：

* **image**：图像文件路径。

**输入数据示例**：

```json
{
    "image": "./images/sample_01.jpg"
}

```

### 2. **核心算子逻辑**

该流水线由三个关键算子串联而成：

#### A. **FixPromptedVQAGenerator（原始生成）**

* **功能**：使用 CapRL 预设的 Prompt 模板（`SYS_PROMPT_MCQ` / `USER_PROMPT_MCQ`），让 VLM 一次性生成 5 道 MCQ。
* **输出**：包含多个 `#### Question` 和选项的非结构化文本块。

#### B. **FunctionalRefiner（正则解析）**

* **逻辑函数**：`parse_mcq_text_logic`
* **功能**：利用正则表达式从原始文本中提取题目、选项（A-F）和正确答案。
* **输出**：结构化的 MCQ 列表 (`parsed_mcq_list`)。

#### C. **VisualDependencyRefiner（依赖性验证）**

这是本流水线的核心过滤器。它对每道题进行 N 次推理（N = `rotate_num`）：

1. **选项旋转**：随机打乱选项顺序（例如将答案从 A 换到 C），防止模型通过“总是选 A”来作弊。
2. **有图推理 (Visual Pass)**：输入图像 + 题目。记录模型答对的比例。
3. **无图推理 (Textual Pass)**：仅输入题目（无图像进行盲测）。记录模型盲猜对的比例。
4. **过滤判据**：

* 保留题目，当且仅当：`Visual_Acc >= pass_visual_min` **且** `Textual_Acc <= pass_textual_max`。
* *示例*：如果一道题不看图也能答对（无图准确率高），说明它考的是常识而非视觉，**剔除**。

### 3. **输出数据**

输出数据 (`final_mcqs`) 仅包含通过了严苛验证的题目。这些题目具有极高的质量和视觉相关性。

**输出数据示例**：

```json
{
    "image": "./images/sample_01.jpg",
    "final_mcqs": [
        {
            "question": "What is the color of the car on the far left?\n - A) Red\n - B) Blue...",
            "answer": "A",
            "stats": {
                "visual_acc": 1.0,  # 4次全对
                "text_acc": 0.0     # 盲猜全错
            }
        }
    ]
}

```

---

## 4. 流水线示例

以下是完整的 `VisualOnlyMCQPipeline` 代码实现 (API 版本)。

```python
import os
os.environ["DF_API_KEY"] = "sk-xxxx"
import argparse
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

from dataflow.operators.core_vision import FixPromptedVQAGenerator, VisualDependencyRefiner
from dataflow.operators.core_text import FunctionalRefiner
from dataflow.prompts.image import ImageCaprlPrompt
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
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
        *,
        first_entry_file: str,
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
        vllm_max_tokens: int = 2048
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
            serving=self.vlm_serving,
            system_prompt=self.prompts_db["SYS_PROMPT_MCQ"],
            user_prompt=self.prompts_db["USER_PROMPT_MCQ"]
        )
        
        # 2. Parse MCQs (Refine)
        self.op_parse = FunctionalRefiner(func=parse_mcq_text_logic)
        
        # 3. Verify Visual Dependency (Refine)
        # 传入 prompt 模板
        self.op_verify = VisualDependencyRefiner(
            serving=self.vlm_serving,
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
        first_entry_file="../example_data/capsbench_images/image_visual_only_mcq_demo.jsonl",
        rotate_num=4,
        pass_visual_min=1.0,
        pass_textual_max=0.25
    )
    pipe.forward()

```
