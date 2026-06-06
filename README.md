# EduAgent：职业教育 AI 学习助手

> 面向香港职业教育学生的轻量级 LLM Agent 演示系统，展示 **Agent 编排 · Tool Calling · Planning · Memory** 四大核心能力。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 1. 项目背景

职业教育（Vocational Education）强调**理论与实践结合**。学生在实训与考核中常面临：概念理解困难、缺乏针对性练习、难以自我诊断薄弱点等问题。

**EduAgent** 是一个可运行的 Web 演示，展示如何用大语言模型（LLM）构建智能学习助手，帮助职教学生：

- 分步骤**讲解**专业概念
- **生成**贴合实训场景的练习题
- **诊断**知识薄弱点并给出学习建议

本项目适合作为 LLM 智能教育、AI Agent 系统相关的 **Portfolio Demo** 或研究助理申请作品。

---

## 2. 为何与 LLM 职业教育相关

| 痛点 | EduAgent 方案 |
|------|---------------|
| 导师时间有限，无法 1 对 1 答疑 | Tutor Agent 24/7 概念讲解 |
| 练习题缺乏针对性 | Quiz Agent 按主题/薄弱点出题 |
| 学生不知从何补弱 | Diagnosis Agent 识别 gap 并给建议 |
| 对话无上下文 | Session Memory 记录历史与薄弱概念 |

系统采用 **意图路由 + 专用 Agent + 工具调用** 架构，比单一 Chatbot 更可控、更可解释，便于教学研究场景评估与扩展。

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (app.py)                   │
│   聊天界面 · 记忆侧边栏 · Agent 工作流可视化                    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              EduAgentOrchestrator (agents.py)                │
│  ① 意图分类 → ② Planning → ③ Tool Calling → ④ 更新 Memory    │
└─────┬──────────────┬──────────────┬─────────────────────────┘
      │              │              │
┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────────┐
│Tutor Agent│  │ Quiz Agent│  │Diagnosis Agent│
│explain_   │  │generate_  │  │diagnose_      │
│concept    │  │quiz       │  │learning_gap   │
└───────────┘  └───────────┘  └───────────────┘
                             │
                    ┌────────▼────────┐
                    │ SessionMemory   │
                    │ (memory.py)     │
                    └─────────────────┘
```

**模块说明：**

| 文件 | 职责 |
|------|------|
| `app.py` | Streamlit 前端、聊天 UI、侧边栏记忆展示 |
| `agents.py` | 编排器、意图分类、Planning、Tool 注册与调用 |
| `memory.py` | 近期问题、薄弱概念、语言偏好 |
| `prompts.py` | 所有 Prompt 与工具元数据 |

---

## 4. Agent 工作流

每次用户提问，系统执行以下流水线：

```
用户输入
   │
   ▼
┌──────────────────┐
│ 1. 意图理解       │  → concept_explanation / practice_generation /
│    (Classify)     │    learning_diagnosis / general_chat
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Planning      │  → 生成步骤计划，选择工具
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Tool Calling  │  → 调用对应 Agent 工具函数
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. Memory Update │  → 记录问题；诊断时提取薄弱概念
└────────┬─────────┘
         ▼
      返回回复 + 工作流轨迹
```

界面中可展开 **Agent 工作流** 卡片，查看意图、工具、规划步骤（便于教学演示与 Debug）。

---

## 5. 三大 Agent 与 Tool Calling

本 Demo **显式模拟** LangChain 风格的 Tool Calling（函数即工具）：

| 工具 ID | Agent | 功能 |
|---------|-------|------|
| `explain_concept_tool` | Tutor Agent | 分步骤概念讲解 |
| `generate_quiz_tool` | Quiz Agent | 生成带解析的练习题 |
| `diagnose_learning_gap_tool` | Diagnosis Agent | 薄弱点诊断 + 学习建议 |
| `general_chat_tool` | EduAgent Core | 通用对话与引导 |

工具注册于 `agents.py` 的 `TOOL_REGISTRY`，Planning 模块根据意图选择工具后调用。

---

## 6. Memory 机制

`SessionMemory`（`memory.py`）在 Streamlit `session_state` 中持久化：

| 字段 | 说明 |
|------|------|
| `recent_questions` | 最近 8 条用户问题（含意图、主题、时间） |
| `weak_concepts` | 诊断 Agent 提取的薄弱概念列表 |
| `preferred_language` | 简体中文 / 繁體中文 / English |

记忆会注入各 Agent 的 System Prompt，实现**跨轮次个性化**（例如 Quiz Agent 针对薄弱点出题）。

---

## 7. 如何运行

### 环境要求

- Python 3.10+
- 可选：OpenAI 或 OpenAI 兼容 API Key

### 安装

```bash
git clone <your-repo-url>
cd eduagent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
# 若使用兼容端点，同时设置 OPENAI_BASE_URL 和 OPENAI_MODEL
```

### 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

> **无 API Key？** 系统自动进入**演示模式**，使用模拟数据展示完整 Agent 工作流，适合本地预览与 GitHub 展示。

---

## 8. How this demo maps to LangChain / LangGraph concepts

当前轻量实现与主流 Agent 框架的对应关系：

| 本 Demo | LangChain | LangGraph |
|---------|-----------|-----------|
| `TOOL_REGISTRY` | `@tool` 装饰的 Tool 列表 | `ToolNode` 中的 tools |
| `EduAgentOrchestrator.plan()` | `AgentExecutor` 的规划步骤 | Planner 节点 |
| `classify_intent()` | RouterChain / 意图分类链 | 条件边 `conditional_edges` |
| `Tutor/Quiz/Diagnosis Agent` | 不同 Chain / Sub-agent | 子图节点 `Subgraph` |
| `SessionMemory` | `ConversationBufferMemory` | `State` 中的 `messages` / 自定义字段 |
| `AgentTrace` | Callback 日志 | `stream_mode="updates"` 状态流 |

**升级路径示例（LangGraph）：**

```python
# 伪代码：未来可将 orchestrator 改写为 StateGraph
graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("plan", plan_node)
graph.add_node("tools", ToolNode([explain_concept, generate_quiz, diagnose]))
graph.add_conditional_edges("classify", route_by_intent)
graph.add_edge("plan", "tools")
graph.add_edge("tools", "update_memory")
```

现有 `prompts.py`、`memory.py`、工具函数可直接复用，仅需替换编排层。

---

## 9. 未来改进

- [ ] **LangChain / LangGraph** 正式集成与可视化 DAG
- [ ] **RAG**：接入 VTC 课程 PDF / 实训手册向量检索
- [ ] **数据库**：PostgreSQL / SQLite 持久化学习记录
- [ ] **学习分析 Dashboard**：练习正确率、知识点掌握热力图
- [ ] **多模态**：上传实训照片进行步骤纠错
- [ ] **评估基准**：Intent 分类准确率、题目质量人工评分

---

## 10. 项目结构

```
eduagent/
├── app.py              # Streamlit 主程序
├── agents.py           # Agent 编排 + Tool Calling
├── memory.py           # 会话记忆
├── prompts.py          # Prompt 与工具定义
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## License

MIT — 可自由用于学习、演示与 Portfolio 展示。
