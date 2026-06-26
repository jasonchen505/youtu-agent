# Youtu-Agent 项目复现 Plan

> 基于 8卡4090 资源的完整全流程复现计划
> 适用于：LLM算法实习面试准备、Agent系统学习、RL后训练实践

---

## 目录

1. [资源评估与可行性分析](#1-资源评估与可行性分析)
2. [复现目标与范围](#2-复现目标与范围)
3. [分阶段复现计划](#3-分阶段复现计划)
4. [详细执行步骤](#4-详细执行步骤)
5. [预期成果与验证指标](#5-预期成果与验证指标)
6. [风险与应对策略](#6-风险与应对策略)

---

## 1. 资源评估与可行性分析

### 1.1 项目GPU需求分析

| 模块 | GPU需求 | 说明 |
|------|---------|------|
| **Training-Free GRPO** | ❌ 不需要GPU | 只需LLM API调用，成本约$8 |
| **Agent RL (rl/agl分支)** | ✅ 需要GPU | 原设计128 GPU，可适配8卡 |
| **本地模型推理** | ✅ 可选GPU | 使用API则不需要 |
| **评估系统** | ❌ 不需要GPU | CPU + 网络IO |
| **工具系统** | ❌ 不需要GPU | CPU + 网络IO |

### 1.2 8卡4090资源评估

**4090规格**：
- 显存：24GB GDDR6X
- FP16算力：约82.6 TFLOPS
- 适合：推理、小规模训练、LoRA微调

**资源适配性**：

| 复现模块 | 可行性 | 资源需求 | 备注 |
|----------|--------|----------|------|
| 基础Agent运行 | ✅ 完全可行 | 无需GPU | 使用API |
| Training-Free GRPO | ✅ 完全可行 | 无需GPU | 核心可复现模块 |
| 本地模型推理 | ✅ 可行 | 1-2卡 | 7B-13B模型 |
| LoRA微调 | ✅ 可行 | 4-8卡 | 7B模型LoRA |
| 全参数微调 | ⚠️ 受限 | 8卡 | 仅7B模型 |
| Agent RL训练 | ⚠️ 需适配 | 8卡 | 需要减小batch |

### 1.3 复现策略选择

**推荐策略**：API + 本地混合模式

```
Phase 1: 使用API复现核心功能（无需GPU）
Phase 2: 使用本地模型验证推理能力（1-2卡）
Phase 3: 使用LoRA微调验证训练能力（4-8卡）
Phase 4: 适配Agent RL到8卡环境（8卡）
```

---

## 2. 复现目标与范围

### 2.1 核心复现目标

**必须完成（Must Have）**：
1. ✅ 环境搭建与配置
2. ✅ 基础Agent运行（SimpleAgent + OrchestraAgent）
3. ✅ 评估系统复现（WebWalkerQA基准）
4. ✅ Training-Free GRPO完整流程
5. ✅ 工具系统验证

**应该完成（Should Have）**：
1. ✅ 本地模型推理
2. ✅ 自定义工具开发
3. ✅ Meta-Agent自动生成
4. ✅ Tracing系统搭建

**可以完成（Nice to Have）**：
1. ⚠️ LoRA微调实验
2. ⚠️ Agent RL适配
3. ⚠️ 多数据集评估

### 2.2 预计时间安排

| 阶段 | 时间 | 内容 |
|------|------|------|
| Phase 1 | Day 1-2 | 环境搭建 + 基础运行 |
| Phase 2 | Day 3-5 | 评估系统 + Training-Free GRPO |
| Phase 3 | Day 6-8 | 本地模型 + 工具系统 |
| Phase 4 | Day 9-12 | LoRA微调 + Agent RL |
| Phase 5 | Day 13-14 | 总结整理 + 文档输出 |

---

## 3. 分阶段复现计划

### Phase 1: 环境搭建与基础运行（Day 1-2）

**目标**：跑通基础Agent，理解框架架构

**任务清单**：

```bash
# Day 1: 环境搭建
□ 1.1 克隆项目
   git clone https://github.com/TencentCloudADP/youtu-agent.git
   cd youtu-agent

□ 1.2 环境配置
   uv venv
   source .venv/bin/activate
   uv sync --group dev
   cp .env.example .env

□ 1.3 配置API密钥
   # 编辑.env文件
   UTU_LLM_TYPE=chat.completions
   UTU_LLM_MODEL=deepseek-chat
   UTU_LLM_BASE_URL=https://api.deepseek.com/v1
   UTU_LLM_API_KEY=your_api_key

□ 1.4 运行基础Agent
   python scripts/cli_chat.py --config simple/base

# Day 2: 框架理解
□ 1.5 阅读核心代码
   - utu/agents/simple_agent.py
   - utu/tools/base.py
   - utu/config/agent_config.py

□ 1.6 运行示例
   python examples/svg_generator/main.py

□ 1.7 理解配置系统
   - configs/agents/simple/base.yaml
   - configs/model/base.yaml
   - configs/tools/search.yaml
```

**验证指标**：
- [ ] Agent能正常对话
- [ ] 搜索工具能正常工作
- [ ] 理解Agent、Tool、Env的关系

### Phase 2: 评估系统与Training-Free GRPO（Day 3-5）

**目标**：复现核心训练流程

**任务清单**：

```bash
# Day 3: 评估系统
□ 2.1 准备评估数据
   python scripts/data/process_web_walker_qa.py

□ 2.2 运行基线评估
   python scripts/run_eval.py --config_name ww --exp_id baseline_001 --dataset WebWalkerQA_15 --concurrency 5

□ 2.3 理解评估流程
   - utu/eval/benchmarks/base_benchmark.py
   - utu/eval/processer/base_processor.py

# Day 4: Training-Free GRPO
□ 2.4 准备训练数据
   python scripts/data/process_training_free_GRPO_data.py

□ 2.5 运行Training-Free GRPO
   python scripts/run_training_free_GRPO.py --config_name math_reasoning

□ 2.6 理解训练流程
   - utu/practice/training_free_grpo.py
   - utu/practice/experience_updater.py

# Day 5: 验证与分析
□ 2.7 评估增强后的Agent
   python scripts/run_eval.py --config_name math/math_practice_AIME24

□ 2.8 分析实验结果
   - 对比baseline和enhanced的准确率
   - 分析提取的经验内容
   - 理解Training-Free GRPO的原理
```

**验证指标**：
- [ ] 评估系统正常运行
- [ ] Training-Free GRPO完整流程跑通
- [ ] 能够对比baseline和enhanced的性能

### Phase 3: 本地模型与工具系统（Day 6-8）

**目标**：验证本地模型推理，理解工具系统

**任务清单**：

```bash
# Day 6: 本地模型推理
□ 3.1 安装Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

□ 3.2 下载模型
   ollama pull qwen2.5:7b
   ollama pull llama3.1:8b

□ 3.3 配置本地模型
   # 修改.env
   UTU_LLM_TYPE=chat.completions
   UTU_LLM_MODEL=qwen2.5:7b
   UTU_LLM_BASE_URL=http://localhost:11434/v1
   UTU_LLM_API_KEY=ollama

□ 3.4 测试本地模型
   python scripts/cli_chat.py --config simple/base

# Day 7: 工具系统
□ 3.5 理解工具注册机制
   - @register_tool装饰器
   - AsyncBaseToolkit基类
   - TOOLKIT_MAP注册

□ 3.6 运行MCP工具示例
   python examples/mcp/stdio_example/main.py

□ 3.7 开发自定义工具
   - 创建新的Toolkit类
   - 实现@register_tool方法
   - 配置YAML文件

# Day 8: Meta-Agent
□ 3.8 运行自动生成
   python scripts/gen_simple_agent.py

□ 3.9 理解生成流程
   - utu/meta/simple_agent_generator.py
   - 四步生成流程
```

**验证指标**：
- [ ] 本地模型能正常推理
- [ ] 理解工具系统的架构
- [ ] 能够开发自定义工具
- [ ] Meta-Agent能自动生成配置

### Phase 4: LoRA微调与Agent RL（Day 9-12）

**目标**：在8卡4090上验证训练能力

**任务清单**：

```bash
# Day 9-10: LoRA微调准备
□ 4.1 收集训练数据
   - 从评估系统导出轨迹数据
   - 转换为训练格式

□ 4.2 准备LoRA训练环境
   pip install peft transformers datasets

□ 4.3 设计LoRA训练脚本
   - 基于Qwen2.5-7B
   - 使用LoRA微调
   - 8卡数据并行

# Day 11-12: Agent RL适配
□ 4.4 理解Agent RL架构
   - 阅读rl/agl分支代码
   - 理解与Agent-Lightning的集成

□ 4.5 适配到8卡环境
   - 减小batch_size
   - 使用梯度累积
   - 优化显存使用

□ 4.6 运行小规模实验
   - 使用小数据集
   - 验证训练流程
```

**验证指标**：
- [ ] LoRA微调流程跑通
- [ ] Agent RL能在8卡上运行
- [ ] 理解RL训练的原理

### Phase 5: 总结整理（Day 13-14）

**目标**：整理学习成果，准备面试

**任务清单**：

```bash
# Day 13: 成果整理
□ 5.1 整理实验结果
   - 各模块的性能指标
   - 对比分析

□ 5.2 编写项目文档
   - 架构设计文档
   - 实验报告
   - 问题与解决方案

# Day 14: 面试准备
□ 5.3 准备项目介绍
   - 项目背景
   - 技术方案
   - 个人贡献
   - 实验结果

□ 5.4 准备技术问答
   - 底层原理
   - 实验设计
   - 问题定位
   - 工程落地
   - 业务理解
```

---

## 4. 详细执行步骤

### 4.1 环境搭建详细步骤

```bash
# 1. 系统要求
# - Python 3.12+
# - uv包管理器
# - Git

# 2. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 克隆项目
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent

# 4. 创建虚拟环境
uv venv
source .venv/bin/activate

# 5. 安装依赖
uv sync --group dev

# 6. 配置环境变量
cp .env.example .env
# 编辑.env文件，填入API密钥

# 7. 验证安装
python scripts/cli_chat.py --config simple/base
```

### 4.2 API密钥配置

```bash
# .env文件配置

# LLM API（必填）
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=sk-xxx  # 替换为你的API Key

# 搜索工具（可选，用于搜索相关任务）
SERPER_API_KEY=xxx  # 从 https://serper.dev 获取
JINA_API_KEY=xxx    # 从 https://jina.ai 获取

# 数据库（可选，默认SQLite）
UTU_DB_URL=sqlite:///test.db

# Tracing（可选）
PHOENIX_ENDPOINT=http://127.0.0.1:6006/v1/traces
PHOENIX_PROJECT_NAME=youtu_agent
```

### 4.3 本地模型配置（使用Ollama）

```bash
# 1. 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. 下载模型
ollama pull qwen2.5:7b      # 约4.7GB
ollama pull qwen2.5:14b     # 约9GB
ollama pull llama3.1:8b     # 约4.7GB

# 3. 配置.env
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=qwen2.5:7b
UTU_LLM_BASE_URL=http://localhost:11434/v1
UTU_LLM_API_KEY=ollama

# 4. 测试
python scripts/cli_chat.py --config simple/base
```

### 4.4 Training-Free GRPO详细步骤

```bash
# 1. 准备数据
python scripts/data/process_training_free_GRPO_data.py

# 2. 查看数据
# 数据会存储在SQLite数据库中

# 3. 运行baseline评估
python scripts/run_eval.py \
  --config_name math/math_AIME24 \
  --exp_id baseline_$(date +%Y%m%d)

# 4. 运行Training-Free GRPO
python scripts/run_training_free_GRPO.py \
  --config_name math_reasoning \
  --exp_id grpo_$(date +%Y%m%d)

# 5. 评估增强后的Agent
python scripts/run_eval.py \
  --config_name math/math_practice_AIME24 \
  --exp_id enhanced_$(date +%Y%m%d)

# 6. 分析结果
python scripts/db/dump_db.py --exp_id <your_exp_id>
```

### 4.5 LoRA微调脚本示例

```python
# scripts/lora_finetune.py
"""
基于8卡4090的LoRA微调脚本
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.nn.parallel import DataParallel

# 配置
MODEL_NAME = "Qwen/Qwen2.5-7B"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
BATCH_SIZE = 4  # 每卡batch size
GRADIENT_ACCUMULATION = 4  # 梯度累积
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def main():
    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. 配置LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # 3. 使用DataParallel进行多卡训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    # 4. 准备数据集
    dataset = load_dataset("your_dataset_name")
    
    # 5. 训练循环
    # ... 实现训练逻辑

if __name__ == "__main__":
    main()
```

---

## 5. 预期成果与验证指标

### 5.1 核心成果

| 阶段 | 成果 | 验证指标 |
|------|------|----------|
| Phase 1 | 框架跑通 | Agent能正常对话 |
| Phase 2 | 评估+训练 | 准确率提升5%+ |
| Phase 3 | 本地模型 | 本地模型正常推理 |
| Phase 4 | 微调验证 | LoRA训练流程跑通 |

### 5.2 性能指标预期

**WebWalkerQA基准**：
- Baseline（DeepSeek-V3）: ~60%
- Enhanced（Training-Free GRPO）: ~65%+

**AIME数学推理**：
- Baseline: ~30%
- Enhanced: ~35%+

**本地模型（Qwen2.5-7B）**：
- 简单问答: 可用
- 复杂推理: 需要优化

### 5.3 学习成果

**必须掌握**：
- [ ] Agent框架的架构设计
- [ ] ReAct模式的原理与实现
- [ ] Training-Free GRPO的算法原理
- [ ] 评估系统的设计与实现
- [ ] 工具系统的架构

**应该掌握**：
- [ ] LoRA微调的原理与实现
- [ ] 分布式训练的基础知识
- [ ] Agent RL的基本概念
- [ ] 系统性能优化方法

---

## 6. 风险与应对策略

### 6.1 技术风险

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| API调用限制 | 高 | 使用多个API Key，或切换到本地模型 |
| 数据集下载失败 | 中 | 提前下载，准备备用数据源 |
| 内存不足 | 中 | 减小batch_size，使用梯度累积 |
| 训练不收敛 | 中 | 调整学习率，增加数据量 |

### 6.2 时间风险

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 环境配置问题 | 高 | 提前准备，参考Docker部署 |
| 代码理解困难 | 中 | 先跑通再理解，逐步深入 |
| 实验结果不理想 | 中 | 分析原因，调整参数 |

### 6.3 资源风险

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| API费用超预算 | 高 | 控制调用次数，使用缓存 |
| GPU显存不足 | 中 | 使用更小的模型，优化显存 |
| 网络问题 | 低 | 提前下载数据和模型 |

---

## 附录：关键命令速查

### 环境管理
```bash
# 创建环境
uv venv && source .venv/bin/activate

# 安装依赖
uv sync --group dev

# 更新依赖
uv lock --upgrade
```

### 运行Agent
```bash
# 基础对话
python scripts/cli_chat.py --config simple/base

# 带搜索的Agent
python scripts/cli_chat.py --config simple/base_search

# 运行示例
python examples/svg_generator/main.py
```

### 评估系统
```bash
# 准备数据
python scripts/data/process_web_walker_qa.py

# 运行评估
python scripts/run_eval.py --config_name ww --exp_id test_001

# 导出结果
python scripts/db/dump_db.py --exp_id test_001
```

### Training-Free GRPO
```bash
# 准备数据
python scripts/data/process_training_free_GRPO_data.py

# 运行训练
python scripts/run_training_free_GRPO.py --config_name math_reasoning

# 评估结果
python scripts/run_eval.py --config_name math/math_practice_AIME24
```

---

*最后更新：2026年6月*
*基于Youtu-Agent项目和8卡4090资源*
