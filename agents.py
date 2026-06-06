"""
EduAgent 核心 Agent 模块
实现：意图识别 → 规划(Planning) → 工具调用(Tool Calling) → 多 Agent 路由 → 记忆更新

代码结构便于后续迁移至 LangChain Tools / LangGraph Nodes。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 本地 .env 加载 API 配置（不会提交到 GitHub）

from memory import SessionMemory, extract_weak_concepts_from_diagnosis
from prompts import (
    DIAGNOSIS_AGENT_PROMPT,
    GENERAL_CHAT_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    INTENT_TO_TOOL,
    PLANNING_PROMPT,
    QUIZ_AGENT_PROMPT,
    TOOLS,
    TUTOR_AGENT_PROMPT,
)

# ── 运行轨迹（供 UI 展示 Agent 工作流）──────────────────────────────────────


@dataclass
class AgentTrace:
    """单次请求的 Agent 执行轨迹"""

    intent: str = ""
    intent_confidence: float = 0.0
    topic: str = ""
    plan_steps: list[str] = field(default_factory=list)
    selected_tool: str = ""
    plan_reasoning: str = ""
    agent_name: str = ""
    mock_mode: bool = False
    error: str | None = None


@dataclass
class AgentResponse:
    """Agent 最终输出"""

    content: str
    trace: AgentTrace


# ── LLM 客户端 ────────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    """LLM 连接配置（支持 UI 输入 / Streamlit Secrets / 环境变量）"""

    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-5.4"


def normalize_base_url(url: str) -> str:
    """补全 OpenAI 兼容端点，自动追加 /v1"""
    url = url.strip().rstrip("/")
    if not url:
        return ""
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def resolve_llm_config(overrides: LLMConfig | None = None) -> LLMConfig:
    """
    解析 LLM 配置，优先级：UI 传入 > 环境变量 > 默认值。
    """
    cfg = LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=normalize_base_url(os.getenv("OPENAI_BASE_URL", "")),
        model=os.getenv("OPENAI_MODEL", "gpt-5.4").strip() or "gpt-5.4",
    )
    if overrides:
        if overrides.api_key.strip():
            cfg.api_key = overrides.api_key.strip()
        if overrides.base_url.strip():
            cfg.base_url = normalize_base_url(overrides.base_url)
        if overrides.model.strip():
            cfg.model = overrides.model.strip()
    return cfg


def create_llm_client(config: LLMConfig | None = None) -> tuple[OpenAI | None, str, bool, LLMConfig]:
    """
    创建 OpenAI 兼容客户端。
    返回 (client, model_name, is_mock, resolved_config)。
    无 API Key 时进入 mock 模式。
    """
    cfg = resolve_llm_config(config)

    if not cfg.api_key:
        return None, cfg.model, True, cfg

    kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs), cfg.model, False, cfg


def test_llm_connection(config: LLMConfig) -> tuple[bool, str]:
    """测试 API 连接是否正常"""
    client, model, is_mock, cfg = create_llm_client(config)
    if is_mock or client is None:
        return False, "请先填写 API Key。"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请只回复：连接成功"}],
            max_tokens=20,
            temperature=0,
        )
        reply = (response.choices[0].message.content or "").strip()
        endpoint = cfg.base_url or "OpenAI 默认端点"
        return True, f"连接成功 · 模型 `{model}` · 端点 `{endpoint}`\n\n测试回复：{reply}"
    except Exception as exc:
        return False, f"连接失败：{exc}"


def _parse_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中解析 JSON（容错 markdown 代码块）"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


def _call_llm(
    client: OpenAI | None,
    model: str,
    system: str,
    user: str,
    mock_fn: Callable[[], str] | None = None,
    is_mock: bool = False,
) -> str:
    """统一 LLM 调用，失败时抛出异常"""
    if is_mock or client is None:
        if mock_fn:
            return mock_fn()
        return "【演示模式】请配置 OPENAI_API_KEY 以获取真实 AI 回复。"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


# ── Mock 响应（无 API Key 时仍可演示完整工作流）────────────────────────────


def _mock_classify(question: str) -> str:
    q = question.lower()
    if any(w in question for w in ["练习", "测验", "题目", "quiz", "测试题"]):
        intent, topic = "practice_generation", "职业技能"
    elif any(w in question for w in ["薄弱", "诊断", "哪里不好", "怎么提高", "建议"]):
        intent, topic = "learning_diagnosis", "综合技能"
    elif any(w in question for w in ["是什么", "解释", "原理", "如何", "什么是", "explain"]):
        intent, topic = "concept_explanation", "职业概念"
    else:
        intent, topic = "general_chat", "未知"
    return json.dumps(
        {"intent": intent, "confidence": 0.85, "topic": topic},
        ensure_ascii=False,
    )


def _mock_plan(intent: str, topic: str) -> str:
    tool = INTENT_TO_TOOL.get(intent, "general_chat_tool")
    return json.dumps(
        {
            "steps": [
                "理解用户意图与上下文",
                f"选择工具 {tool}",
                "生成个性化回复",
                "更新会话记忆",
            ],
            "selected_tool": tool,
            "reasoning": f"根据意图 {intent} 路由至对应 Agent",
        },
        ensure_ascii=False,
    )


def _mock_agent_response(tool: str, question: str, ctx: dict[str, str]) -> str:
    """各工具的模拟回复"""
    lang = ctx.get("language", "简体中文")

    if tool == "explain_concept_tool":
        return f"""## 📘 概念讲解（演示模式）

**您的问题：** {question}

### 第一步：定义
在职业教育场景中，该概念指学生在实训中需要掌握的核心知识点。

### 第二步：为什么重要
香港职训局（VTC）课程强调「做中学」，理解此概念有助于通过技能评估。

### 第三步：实训举例
例如在机电维修实训中，正确理解安全操作规程可避免工伤事故。

### 第四步：记忆提示
> 系统已记录您的学习轨迹，后续会结合您的薄弱点「{ctx.get('weak_concepts', '暂无')}」进行巩固。

*（{lang} · Tutor Agent · 演示数据）*"""

    if tool == "generate_quiz_tool":
        return f"""## 📝 练习题（演示模式）

**主题：** {question}

---

**1. [选择题]** 职业教育「双元制」培训的核心特点是什么？
- A. 纯理论授课
- B. 校企合作、工学结合 ✓
- C. 仅在线学习
- D. 无考核标准

**解析：** 双元制强调学校与企业共同培养，学生 alternates 在校学习与岗位实践。

---

**2. [简答题]** 请列举两项实训室安全操作规范。
**参考答案：** ① 穿戴防护用品；② 操作前检查设备；③ 遵守导师指令。

---

**3. [情景题]** 若发现焊接区域通风不足，你应如何处理？
**参考答案：** 立即停止作业、报告导师、开启排风或转移作业点。

*（Quiz Agent · 演示数据 · 共 3 题）*"""

    if tool == "diagnose_learning_gap_tool":
        return f"""## 🔍 学习诊断报告（演示模式）

**分析：** 根据您的问题「{question[:50]}…」，系统识别以下可加强领域：

### 薄弱概念
- 基础理论与应用场景的衔接
- 实训安全规范的细节记忆
- 考核要点的时间管理

### 个性化建议
1. **本周重点：** 复习相关章节并做 5 道同类练习题
2. **实训建议：** 向导师申请 30 分钟跟岗观察
3. **资源推荐：** 使用 VTC 在线学习平台复习模块
4. **自我检测：** 3 天后再次使用「练习生成」功能自测

### 下一步
建议先使用「概念讲解」巩固基础，再用「练习生成」检验效果。

*（Diagnosis Agent · 演示数据）*"""

    return f"""您好！我是 EduAgent 职业教育 AI 学习助手。

您说：「{question}」

我可以帮您：**讲解概念**、**生成练习题**、**诊断学习薄弱点**。
请告诉我您想学习哪个主题，例如：「请解释 OSHA 安全规范」或「给我出 3 道电工题」。

*（General Chat · 演示模式）*"""


# ── Tool Calling 层（模拟 LangChain @tool）────────────────────────────────


def explain_concept_tool(
    client: OpenAI | None,
    model: str,
    question: str,
    memory: SessionMemory,
    is_mock: bool,
) -> str:
    """工具：概念讲解 → Tutor Agent"""
    ctx = memory.get_context_for_prompt()
    system = TUTOR_AGENT_PROMPT.format(**ctx)
    return _call_llm(
        client, model, system, question,
        mock_fn=lambda: _mock_agent_response("explain_concept_tool", question, ctx),
        is_mock=is_mock,
    )


def generate_quiz_tool(
    client: OpenAI | None,
    model: str,
    question: str,
    memory: SessionMemory,
    is_mock: bool,
) -> str:
    """工具：生成练习题 → Quiz Agent"""
    ctx = memory.get_context_for_prompt()
    system = QUIZ_AGENT_PROMPT.format(**ctx)
    return _call_llm(
        client, model, system, question,
        mock_fn=lambda: _mock_agent_response("generate_quiz_tool", question, ctx),
        is_mock=is_mock,
    )


def diagnose_learning_gap_tool(
    client: OpenAI | None,
    model: str,
    question: str,
    memory: SessionMemory,
    is_mock: bool,
) -> str:
    """工具：学习诊断 → Diagnosis Agent"""
    ctx = memory.get_context_for_prompt()
    system = DIAGNOSIS_AGENT_PROMPT.format(**ctx)
    content = _call_llm(
        client, model, system, question,
        mock_fn=lambda: _mock_agent_response("diagnose_learning_gap_tool", question, ctx),
        is_mock=is_mock,
    )
    # 诊断结果中提取薄弱概念写入记忆
    new_weak = extract_weak_concepts_from_diagnosis(content)
    if new_weak:
        memory.add_weak_concepts(new_weak)
    return content


def general_chat_tool(
    client: OpenAI | None,
    model: str,
    question: str,
    memory: SessionMemory,
    is_mock: bool,
) -> str:
    """工具：通用对话"""
    ctx = memory.get_context_for_prompt()
    system = GENERAL_CHAT_PROMPT.format(language=ctx["language"])
    return _call_llm(
        client, model, system, question,
        mock_fn=lambda: _mock_agent_response("general_chat_tool", question, ctx),
        is_mock=is_mock,
    )


# 工具注册表 — 对应 LangChain ToolRegistry / LangGraph tool node
TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "explain_concept_tool": explain_concept_tool,
    "generate_quiz_tool": generate_quiz_tool,
    "diagnose_learning_gap_tool": diagnose_learning_gap_tool,
    "general_chat_tool": general_chat_tool,
}


# ── 主 Agent 编排器（Orchestrator）──────────────────────────────────────────


class EduAgentOrchestrator:
    """
    主编排器：Planning + Tool Calling + Memory Update
    未来可映射为 LangGraph StateGraph 的 supervisor 节点。
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.client, self.model, self.is_mock, self.config = create_llm_client(config)
        self.base_url = self.config.base_url

    def classify_intent(self, question: str) -> dict[str, Any]:
        """Step 1: 意图理解"""
        raw = _call_llm(
            self.client,
            self.model,
            INTENT_CLASSIFICATION_PROMPT,
            question,
            mock_fn=lambda: _mock_classify(question),
            is_mock=self.is_mock,
        )
        return _parse_json(raw)

    def plan(
        self, question: str, intent: str, topic: str, memory: SessionMemory
    ) -> dict[str, Any]:
        """Step 2: 规划 — 选择工具与执行步骤"""
        ctx = memory.get_context_for_prompt()
        prompt = PLANNING_PROMPT.format(
            question=question,
            intent=intent,
            topic=topic,
            weak_concepts=ctx["weak_concepts"],
            language=ctx["language"],
        )
        raw = _call_llm(
            self.client,
            self.model,
            "你是规划模块，只返回 JSON。",
            prompt,
            mock_fn=lambda: _mock_plan(intent, topic),
            is_mock=self.is_mock,
        )
        return _parse_json(raw)

    def run(self, question: str, memory: SessionMemory) -> AgentResponse:
        """
        完整 Agent 工作流：
        理解意图 → 生成计划 → 调用工具 → 更新记忆
        """
        trace = AgentTrace(mock_mode=self.is_mock)

        try:
            # 1. 意图分类
            classification = self.classify_intent(question)
            intent = classification.get("intent", "general_chat")
            trace.intent = intent
            trace.intent_confidence = float(classification.get("confidence", 0.0))
            trace.topic = classification.get("topic", "未知")

            # 2. 规划
            plan = self.plan(question, intent, trace.topic, memory)
            trace.plan_steps = plan.get("steps", [])
            trace.selected_tool = plan.get(
                "selected_tool", INTENT_TO_TOOL.get(intent, "general_chat_tool")
            )
            trace.plan_reasoning = plan.get("reasoning", "")

            # 3. 工具调用 → 对应 Agent
            tool_name = trace.selected_tool
            if tool_name not in TOOL_REGISTRY:
                tool_name = INTENT_TO_TOOL.get(intent, "general_chat_tool")
                trace.selected_tool = tool_name

            tool_fn = TOOL_REGISTRY[tool_name]
            trace.agent_name = TOOLS.get(tool_name, {}).get("agent", "EduAgent")

            content = tool_fn(
                self.client, self.model, question, memory, self.is_mock
            )

            # 4. 更新记忆
            memory.add_question(question, intent, trace.topic)

            return AgentResponse(content=content, trace=trace)

        except Exception as exc:
            trace.error = str(exc)
            fallback = (
                f"⚠️ 处理请求时出错：{exc}\n\n"
                "请检查 API Key 配置与网络连接。"
                + ("（当前为演示模式，部分功能使用模拟数据）" if self.is_mock else "")
            )
            return AgentResponse(content=fallback, trace=trace)
