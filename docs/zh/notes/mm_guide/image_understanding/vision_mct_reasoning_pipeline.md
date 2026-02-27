---
title: 视觉 MCTS 推理链生成流水线
icon: mdi:image-text
createTime: 2026/01/11 21:59:59
permalink: /zh/mm_guide/vision_mct_reasoning_pipeline/
---

## 1. 概述

**视觉 MCTS 推理链生成流水线 (Vision MCTS Reasoning Pipeline)** 旨在为多模态大模型构建高质量的**过程监督数据（Process Supervision Data）**。该流水线能够处理两种数据来源：已有的蒙特卡洛树搜索（MCTS）轨迹数据，或直接利用 VLM 生成新的推理链。

该流水线是 **Grounded-RL** 和 **SFT 数据构建**的核心工具，它将复杂的树状搜索过程“线性化”为模型可学习的 `<think>...</think><answer>...</answer>` 格式。

我们支持以下应用场景：

* **从 MCTS 树提取数据**：将搜索树中高价值的路径（Rollouts）转化为线性训练数据。
* **混合数据构建**：对于没有搜索树的样本，自动回退到使用 VLM 进行 CoT 生成。
* **空间推理增强**：支持生成包含显式坐标（Bounding Box）的空间推理链。

流水线的主要流程包括：

1. **MCTS 树解析**：解析输入数据中的搜索树结构，提取成功的推理路径。
2. **视觉推理生成 (Fallback)**：对于缺失树结构或解析失败的样本，利用 VLM 重新生成推理链。
3. **数据标准化**：输出统一格式的推理链数据。

---

## 2. 快速开始

### 第一步：准备工作目录

```bash
mkdir run_mcts_reasoning
cd run_mcts_reasoning

```

### 第二步：初始化 DataFlow-MM

```bash
dataflowmm init

```

这时你会看到：

```bash
gpu_pipelines/vision_mct_reasoning_pipeline.py

```

### 第三步：下载示例数据

```bash
huggingface-cli download --repo-type dataset OpenDCAI/dataflow-demo-image --local-dir ./example_data

```

### 第四步：配置参数

确保输入文件（jsonl）包含 `tree` 字段（用于提取）或仅包含 `question/image`（用于生成）：

```python
if __name__ == "__main__":
    pipe = VisionMCTSReasoningPipeline(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        first_entry_file="../example_data/capsbench_images/visual_mct_reasoning_demo.jsonl",
        prompt_type="spatial",
        hf_cache_dir="~/.cache/huggingface",
        download_dir="../ckpt/models/Qwen2.5-VL-3B-Instruct",
    )
    pipe.forward()

```

> **⚠️ 模型路径配置的重要提示（以 `Qwen2.5-VL-3B-Instruct` 为例）：**
> * **如果您已经下载好了模型文件**：请将 `model_path` 修改为您的本地模型路径。**务必保证**模型存放的最终文件夹名称精确为 `Qwen2.5-VL-3B-Instruct`，否则底层解析时将无法正确匹配和识别该模型。
> * **如果您还未下载模型（需要自动下载）**：请一定要指定 `download_dir` 参数，并且该目录路径**必须以** `Qwen2.5-VL-3B-Instruct` **结尾**（正如默认参数所示），否则下载完成后同样会导致框架无法识别模型。
> 
> 

### 第五步：一键运行

```bash
cd gpu_pipelines
python vision_mct_reasoning_pipeline.py

```

> **🛠️ 常见问题排查 (Troubleshooting)**
> **问题 1：** 如果遇到类似如下的动态链接库冲突报错：
> `ImportError: .../miniconda3/envs/Dataflow-MM/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`
> **解决方法：** 这通常是环境变量干扰导致的。请在运行命令前清空 `LD_LIBRARY_PATH`：
> ```bash
> LD_LIBRARY_PATH="" python vision_mcts_pipeline.py
> 
> ```
> 
> 
> **问题 2：** 如果您使用的是 **Qwen 系列模型**，并且遇到以下报错：
> `KeyError: "Missing required keys in rope_scaling for 'rope_type'='None': {'rope_type'}"`
> **解决方法：** 打开模型文件夹下的 `config.json` 文件，找到 `rope_scaling` 配置块，将 `"type"` 字段修改为 `"rope_type"` 即可。
> **修改前：**
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
> **修改后：**
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

## 3. 数据流与流水线逻辑

### 1. **输入数据**

输入数据通常来源于 MCTS 搜索过程的日志，或未标注的图文对：

* **image**：图像路径。
* **question**：视觉问题。
* **tree**（可选）：MCTS 搜索树的 JSON 结构，包含节点值（Value）、访问次数（Visits）和动作（Actions）。

**输入数据示例**：

```json
{
    "image": "./images/puzzle.jpg",
    "question": "What is the next step to solve this?",
    "tree": { "root": { "children": [...], "value": 1.0, "text": "Step 1..." } }
}

```

### 2. **核心算子逻辑**

该流水线采用 **“提取优先，生成兜底”** 的混合策略：

#### A. **MCTSTreeRefiner（树结构解析器）**

该算子负责处理 `tree` 字段。它遍历树结构，根据节点价值（Q-value）筛选出从根节点到叶子节点的最佳路径。

* **输入**：`tree` 对象。
* **功能**：线性化树路径，过滤掉低价值或未完成的搜索分支。
* **输出**：提取出的推理链列表（`mcts_chains`）。

#### B. **VisualReasoningGenerator（视觉推理生成器）**

该算子是流水线的“生成引擎”。它接收上一步的提取结果作为输入。

* **机制**：检查 `input_existing_chains_key`（即 `mcts_chains`）。
* 如果 MCTS 解析成功（链存在），则直接复用，不进行推理（节省计算资源）。
* 如果 MCTS 链为空（树不存在或解析失败），则调用 VLM，根据 `prompt_type`（如 `spatial`）从头生成推理链。


* **Prompt 类型**：支持 `spatial`（空间坐标推理）、`logical`（逻辑推理）等模式。

### 3. **输出数据**

最终生成的输出数据（`final_reasoning_chains`）将包含高质量的思维链，可直接用于 SFT 训练。

**输出示例**：

```json
{
    "image": "./images/puzzle.jpg",
    "final_reasoning_chains": [
        "<think>First, locate the red block at [100, 200]. To solve the puzzle, it needs to move right...</think><answer>Move Red Block</answer>"
    ]
}

```

---

## 4. 流水线示例

以下是完整的 `VisionMCTSReasoningPipeline` 代码实现 (GPU 版本)。

```python
from dataflow.utils.storage import FileStorage
from dataflow.serving.local_model_vlm_serving import LocalModelVLMServing_vllm

# 引入原子算子
from dataflow.operators.core_text import MCTSTreeRefiner
from dataflow.operators.core_vision import VisualReasoningGenerator

class VisionMCTSReasoningPipeline:
    def __init__(
        self,
        model_path: str,
        *,
        # Storage
        hf_cache_dir: str | None = None,
        download_dir: str = "./ckpt/models",
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
        # VLLM
        vllm_max_tokens: int = 1024
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
            vllm_temperature=0.7,
            vllm_max_tokens=vllm_max_tokens
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
            serving=self.serving,
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
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        first_entry_file="../example_data/capsbench_images/visual_mct_reasoning_demo.jsonl",
        prompt_type="spatial",
        hf_cache_dir="~/.cache/huggingface",
        download_dir="../ckpt/models/Qwen2.5-VL-3B-Instruct",
    )
    pipe.forward()

```
