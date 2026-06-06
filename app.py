"""
EduAgent — 职业教育 AI 学习助手
Streamlit 网页前端：聊天界面 + 记忆侧边栏 + Agent 工作流可视化
"""

from __future__ import annotations

from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # 从 .env 加载 OPENAI_API_KEY 等环境变量

from agents import EduAgentOrchestrator
from memory import SessionMemory
from prompts import INTENT_LABELS_ZH, TOOLS

# ── 页面配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduAgent | 职业教育 AI 学习助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式（学术风 + 现代 UI）──────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #3d8fd1 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.25);
    }
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    .badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .trace-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2d6a9f;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .trace-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.4rem 0;
        font-size: 0.9rem;
        color: #334155;
    }
    .step-num {
        background: #2d6a9f;
        color: white;
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
    }

    .memory-item {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .memory-tag {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 0.15rem 0.15rem 0 0;
    }
    .weak-tag {
        background: #fee2e2;
        color: #991b1b;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #ffffff 100%);
    }

    .stChatMessage {
        border-radius: 12px !important;
    }

    .demo-banner {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session 初始化 ──────────────────────────────────────────────────────────
def init_session() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = EduAgentOrchestrator()
    if "show_trace" not in st.session_state:
        st.session_state.show_trace = True
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None


init_session()
memory: SessionMemory = st.session_state.memory
orchestrator: EduAgentOrchestrator = st.session_state.orchestrator

# ── 侧边栏：记忆与设置 ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 会话记忆")
    st.caption("Agent Memory — 个性化学习上下文")

    language = st.selectbox(
        "回答语言",
        ["简体中文", "繁體中文", "English"],
        index=["简体中文", "繁體中文", "English"].index(memory.preferred_language)
        if memory.preferred_language in ["简体中文", "繁體中文", "English"]
        else 0,
    )
    memory.set_language(language)

    st.markdown("---")
    st.markdown("**📋 近期问题**")
    if memory.recent_questions:
        for entry in memory.recent_questions[:6]:
            intent_label = INTENT_LABELS_ZH.get(entry.intent, entry.intent)
            st.markdown(
                f'<div class="memory-item">'
                f'<small style="color:#64748b">{entry.timestamp} · {intent_label}</small><br>'
                f'{entry.question[:60]}{"…" if len(entry.question) > 60 else ""}'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("暂无记录，开始提问吧！")

    st.markdown("**⚠️ 薄弱概念**")
    if memory.weak_concepts:
        tags = "".join(
            f'<span class="memory-tag weak-tag">{c}</span>'
            for c in memory.weak_concepts[:10]
        )
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.caption("使用「学习诊断」功能后会自动记录")

    st.markdown("---")
    st.markdown("**🔧 可用工具 (Tool Calling)**")
    for tool_id, meta in TOOLS.items():
        with st.expander(f"{meta['name']}", expanded=False):
            st.write(meta["description"])
            st.caption(f"Agent: {meta['agent']}")

    st.markdown("---")
    st.session_state.show_trace = st.toggle("显示 Agent 工作流", value=True)

    if st.button("🗑️ 清空记忆", use_container_width=True):
        memory.clear()
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    # API 状态
    st.markdown("---")
    if orchestrator.is_mock:
        st.warning("⚙️ 演示模式\n\n请设置 `OPENAI_API_KEY` 环境变量以启用真实 LLM。")
    else:
        st.success(f"✅ LLM 已连接\n\n模型: `{orchestrator.model}`")

# ── 主标题 ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>🎓 EduAgent · 职业教育 AI 学习助手</h1>
    <p>
        面向香港职业教育学生的智能学习伙伴 — 支持概念讲解、练习生成、学习诊断。
        演示 <strong>Agent 编排</strong> · <strong>Tool Calling</strong> ·
        <strong>Planning</strong> · <strong>Memory</strong> 四大 AI Agent 核心能力。
    </p>
    <div class="badge-row">
        <span class="badge">Tutor Agent</span>
        <span class="badge">Quiz Agent</span>
        <span class="badge">Diagnosis Agent</span>
        <span class="badge">Session Memory</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if orchestrator.is_mock:
    st.markdown(
        '<div class="demo-banner">'
        "💡 <strong>演示模式</strong>：未检测到 API Key，系统使用模拟数据展示完整 Agent 工作流。"
        "配置环境变量后即可接入真实大模型。"
        "</div>",
        unsafe_allow_html=True,
    )

# ── 快捷示例 ──────────────────────────────────────────────────────────────
st.markdown("**快速开始** — 点击示例问题：")
example_cols = st.columns(3)
examples = [
    "请用简单步骤解释什么是「焊接安全规范」？",
    "给我出 3 道关于电工基础的选择题",
    "我觉得电路分析很薄弱，该怎么提高？",
]
for col, ex in zip(example_cols, examples):
    with col:
        if st.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
            st.session_state.pending_question = ex

# ── 聊天历史 ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace") and st.session_state.show_trace:
            trace = msg["trace"]
            intent_label = INTENT_LABELS_ZH.get(trace.intent, trace.intent)
            steps_html = "".join(
                f'<div class="trace-step">'
                f'<span class="step-num">{i+1}</span><span>{s}</span></div>'
                for i, s in enumerate(trace.plan_steps)
            )
            st.markdown(
                f'<div class="trace-card">'
                f"<strong>🔄 Agent 工作流</strong> "
                f"（意图: {intent_label} · 工具: {trace.selected_tool} · {trace.agent_name}）"
                f"<br><small style='color:#64748b'>{trace.plan_reasoning}</small>"
                f"{steps_html}</div>",
                unsafe_allow_html=True,
            )

# ── 处理待发送问题 ──────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_question", None)

# ── 用户输入 ──────────────────────────────────────────────────────────────
user_input = st.chat_input("输入您的学习问题，例如：解释某个概念、生成练习题、诊断薄弱点…")

question = pending or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent 正在规划并调用工具…"):
            result = orchestrator.run(question, memory)
        st.markdown(result.content)
        st.session_state.last_trace = result.trace

        if st.session_state.show_trace and result.trace:
            trace = result.trace
            intent_label = INTENT_LABELS_ZH.get(trace.intent, trace.intent)
            steps_html = "".join(
                f'<div class="trace-step">'
                f'<span class="step-num">{i+1}</span><span>{s}</span></div>'
                for i, s in enumerate(trace.plan_steps)
            )
            st.markdown(
                f'<div class="trace-card">'
                f"<strong>🔄 Agent 工作流</strong> "
                f"（意图: {intent_label} · 置信度: {trace.intent_confidence:.0%} · "
                f"工具: {trace.selected_tool}）"
                f"<br><small style='color:#64748b'>{trace.plan_reasoning}</small>"
                f"{steps_html}</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.content,
            "trace": result.trace,
        }
    )
    st.rerun()

# ── 页脚 ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "EduAgent Demo · 职业教育 LLM Agent 系统 · "
    "架构可升级至 LangChain / LangGraph · Portfolio Project"
)
