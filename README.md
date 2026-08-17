# 企业文档智能管理系统 (Agentic_Doc_Manager)

> 一个端到端的 **Agentic RAG** 企业文档知识库系统：从 PDF 摄取、知识蒸馏、混合检索、智能问答到自动化评估，形成完整闭环。

基于 **检索增强生成 (RAG)** 架构，融合 **微调 Embedding / Reranker**、**LLM-as-Judge自动化评估**与**RL提示词优化**，支持本地大模型推理与Web服务化部署。

---

## 核心特性

- **五阶段智能管道**：摄取 → 蒸馏 → 检索 → 问答 → 评估，全链路自动化，无需人工干预
- **父子切片 (Parent-Child Chunking)**：句子级语义切分，父块600Chunk / 子块200Chunk，兼顾召回完整性与语义粒度
- **混合检索**：向量语义检索 (BGE) + BM25 关键词检索 (jieba 自实现) → RRF 融合 → CrossEncoder精排，四路召回互补
- **知识蒸馏三层过滤**：忠实度 (LLM CoT 校验) → 多样性 (向量去重) → 相关性 (相似度阈值)，自动生成高质量数据
- **为了系统性能将React转为确定性Agent Pipline**：改写 → 检索 → 生成固定序列，内置防幻觉红线（强制来源校验 + 无依据即弃权）
- **双库职责分离**：SQLite 管理结构化元数据 (7 张表)，ChromaDB 负责纯向量索引 (HNSW + Cosine)
- **LLM-as-Judge 评估闭环**：F (忠实度) / A (准确性) / R (相关性) 三维度百分比评分，版本化存储对比
- **模型微调**：Embedding / Reranker 领域微调，DPO 偏好对齐，CMA-ES 提示词优化
- **工程化与安全**：4bit 量化推理、批量forward、JWT认证、滑动窗口限流、安全响应头、操作审计日志

---

## 系统架构

```
                        ┌─────────────────────┐
                        │   Run.py 统一入口    │
                        │  server / ingest /  │
                        │  search / chat / eval│
                        └──────────┬──────────┘
                                   │
     ┌─────────────────┬───────────┼───────────┬─────────────────┐
     ▼                 ▼           ▼           ▼                 ▼
┌──────────┐    ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐
│ ① 摄取    │    │ ② 知识蒸馏  │ │ ③ 混合检索  │ │ ④ Agent    │ │ ⑤ 评估   │
│Ingestion │───▶│Knowledge   │ │ Retrieval  │ │ 问答       │ │ F/A/R    │
└──────────┘    └────────────┘ └────────────┘ └────────────┘ └──────────┘
 PDF解析         QA生成+三层过滤   向量+BM25→RRF    改写→检索→生成   LLM-Judge
 父子切片        忠实/多样/相关     →Reranker精排    防幻觉约束      三维评分
     │                 │              ▲               │             │
     └────────┬────────┘              │               │             │
              ▼                       │               │             │
   ┌──────────────────────┐   ┌──────┴───────┐       │             │
   │ 双库存储              │   │ 微调模型      │       │             │
   │ SQLite + ChromaDB    │   │ Embedding/   │       │             │
   └──────────────────────┘   │ Reranker     │       │             │
              ▲                └──────────────┘       │             │
              └─────────────── 检索依赖 ───────────────┘             │
                                                                    │
                                          ┌─────────────────────────┘
                                          ▼
                                 版本化评估报告 → 反馈驱动模型迭代
```

### 混合检索链路

```
用户查询 ──▶ 向量检索 (BGE) ──┐
                              ├──▶ RRF 融合 (k=60) ──▶ Top-60 候选 ──▶ CrossEncoder 精排 ──▶ Top-K
           BM25 检索 (jieba) ─┘                                                              │
                                                                                             ▼
                                                                                    [来源: 文件] + 关联QA
```

---

## 技术栈

| 层 | 技术选型 |
|---|---|
| **生成 LLM** | DeepSeek V4 Pro (API) / Qwen2.5-7B|
| **Embedding** | BAAI/bge-small-zh-v1.5 (领域微调) |
| **Reranker** | BAAI/bge-reranker-v2-m3 (CrossEncoder, 领域微调) |
| **向量数据库** | ChromaDB (HNSW + Cosine) |
| **元数据存储** | SQLite (7 表) |
| **后端** | FastAPI + Uvicorn |
| **前端** | Vanilla JS SPA |
| **认证 / 安全** | 自实现 JWT (HS256) + 限流 + 安全头 |
| **PDF 解析** | PyMuPDF |
| **分词 / BM25** | jieba + 自实现 BM25 |
| **微调框架** | PyTorch + Transformers + Sentence-Transformers + TRL |
| **RL 优化** | CMA-ES 进化算法 |

---

## 目录结构

```
Agentic_Doc_System/
├── Run.py                      # 统一入口 (server/ingest/search/chat/eval/status)
├── requirements.txt            # 依赖清单
├── .env.example                # 环境变量模板
├── src/
│   ├── config.py               # 全局配置 (.env 加载)
│   ├── ingestion/              # ① PDF 解析 + 父子切片
│   ├── knowledge/              # ② QA 生成 + 三层蒸馏过滤
│   ├── storage/                #    SQLite 元数据 + ChromaDB 向量
│   ├── retrieval/              # ③ 混合检索 + BM25 + Reranker
│   ├── agent/                  # ④ 确定性 Agent 管道 + 工具层
│   ├── api/                    #    FastAPI 服务 + JWT + Web 前端
│   ├── evaluation/             # ⑤ LLM-as-Judge F/A/R 评估
│   └── rl/                     #    CMA-ES 提示词优化
├── scripts/                    # 训练 / 评估 / 服务脚本
│   ├── run_pipeline.py         # 全链路入库
│   ├── train_embedding.py      # Embedding 微调
│   ├── train_reranker.py       # Reranker 微调 (困难负样本)
│   ├── train_dpo.py            # DPO 偏好对齐
│   ├── build_dpo_data.py       # DPO 数据自动构造
│   ├── eval_reranker.py        # Reranker ON/OFF 消融评估
│   └── run_qwen_server.py      # 本地 Qwen 推理服务 (4bit)
└── tests/                      # 单元测试
```

---

## 快速开始

### 环境准备

```bash
# 建议使用 conda 创建独立环境
conda create -n agent_doc python=3.10 -y
conda activate agent_doc

pip install -r requirements.txt
```


### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

`.env` 关键项：

```ini
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-v4-pro
```

### 3. 首次初始化

```bash
python Run.py setup   # 检查依赖 + 配置 .env + 可选批量入库
```

### 4. 常用命令

```bash
python Run.py server                    # 启动 Web 服务 → http://127.0.0.1:8000
python Run.py ingest --all --with-qa     # 批量入库 + QA 蒸馏
python Run.py search                     # 命令行交互检索
python Run.py chat                       # 命令行 AI 对话
python Run.py eval --dataset 500         # F/A/R 系统评估
python Run.py status                     # 查看知识库状态
```

**Web 界面**：默认账号 `admin / admin123`（生产环境请务必修改）。

---

## 核心流程详解

### Ingestion

- **PDF 解析**：PyMuPDF 提取文本、页数、文件大小
- **父子切片**：句子级贪心拼接（中文语义单元为句子，避免截断破坏 Embedding 质量）
  - 父块：600 字（召回时返回完整上下文）
  - 子块：200 字（精确语义检索）

### Knowledge Distillation

- **QA 生成**：LLM 批量为每个 chunk 生成问答对
- **三层串行过滤**：
  1. **忠实度 (Faithfulness)**：LLM CoT 校验答案是否基于原文
  2. **多样性 (Diversity)**：向量相似度去重，防止同质化
  3. **相关性 (Relevancy)**：相似度阈值过滤弱相关 QA

### Retrieval

- 粗排：向量 + BM25 双路召回 → RRF 等权融合
- 精排：CrossEncoder Reranker 重排序
- 元数据增强：文件名注入 BM25 与 Reranker，打破「元数据致盲」

### Agent问答

确定性RAG管道：**改写 → 检索 → 生成**

- 强约束 System Prompt 内置防幻觉红线
- 强制来源校验：问题指定文档 → 上下文必须匹配，否则弃权
- XML 结构化输出：`<thinking>` 思维链 + `<answer>` 结论

### Evaluation

- **LLM-as-Judge**：DeepSeek V4 Pro 作为裁判，对回答打 F / A / R 三维分
- 版本化存储：`QAEval_result/eval_vXXX.json` + `index.json` 生成对比曲线

---

## 模型微调

| 脚本 | 目标 | 方法 |
|---|---|---|
| `train_embedding.py` | Embedding 领域适配 | 对比学习 / 监督微调 |
| `train_reranker.py` | Reranker 精排优化 | CrossEncoder 点式二分类，60% 困难负样本 (同类/检索/跨类) |
| `train_dpo.py` | 生成模型偏好对齐 | DPO，自动构造 chosen/rejected 对 |
| `run_rl_optimizer.py` | 提示词优化 | CMA-ES 进化搜索 |

> **注**：模型权重 (~46GB) 未纳入仓库。微调后的 Embedding 配置见 `models/embedding-finetuned/`，权重可通过上述脚本在本地复现，或从 Hugging Face / 网盘下载。

---

## 性能优化 (算子优化)

- **4bit 量化**：NF4 + 双重量化 (double quant)，bfloat16 计算精度，显著降低显存占用
- **批量前向**：`/v1/chat/batch` 端点，单次 forward pass 处理多 prompt
- **向量索引**：HNSW 图 + Cosine 相似度，近邻检索亚线性复杂度
- **零依赖 BM25**：自实现轻量级关键词检索，无需 Elasticsearch

---

## 安全设计

- API Key 经 `.env` 加载，不入库、不上传
- 自实现 JWT 认证 (HS256)
- 滑动窗口限流 (200 次 / 60s)
- 安全响应头：CSP / XSS 防护 / 点击劫持防护
- 操作审计日志：全量记录 API 调用轨迹
- PDF 上传魔数校验 + 50MB 大小限制

---

## License

[MIT](./LICENSE) — 仅供学习与简历展示使用。

---

## 关于本项目

这是一个从零实现的RAG 全链路项目，覆盖了文档智能领域的核心工程环节：从数据摄取、知识工程、检索优化，到 Agent 推理与自动化评估的完整闭环。适合作为AI应用工程/RAG方向的学习与作品展示。
