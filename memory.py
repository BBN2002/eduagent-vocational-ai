"""
EduAgent 记忆模块
模拟 Agent Memory：存储近期问题、薄弱概念、语言偏好。
后续可替换为 LangChain ConversationBufferMemory 或向量数据库。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    """单条对话记忆"""

    question: str
    intent: str
    topic: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


@dataclass
class SessionMemory:
    """
    会话级记忆（Streamlit session_state 中持久化）。
    对应 LangGraph 中的 State 里的 memory 字段。
    """

    recent_questions: list[MemoryEntry] = field(default_factory=list)
    weak_concepts: list[str] = field(default_factory=list)
    preferred_language: str = "简体中文"
    max_recent: int = 8
    max_weak: int = 12

    def add_question(self, question: str, intent: str, topic: str) -> None:
        """记录用户问题"""
        self.recent_questions.insert(
            0, MemoryEntry(question=question, intent=intent, topic=topic)
        )
        self.recent_questions = self.recent_questions[: self.max_recent]

    def add_weak_concepts(self, concepts: list[str]) -> None:
        """合并薄弱概念（去重，保留顺序）"""
        for c in concepts:
            c = c.strip()
            if c and c not in self.weak_concepts:
                self.weak_concepts.append(c)
        self.weak_concepts = self.weak_concepts[: self.max_weak]

    def set_language(self, language: str) -> None:
        self.preferred_language = language

    def get_context_for_prompt(self) -> dict[str, str]:
        """生成供 Agent 使用的上下文字符串"""
        recent = (
            "；".join(e.question[:40] for e in self.recent_questions[:5])
            or "（暂无）"
        )
        weak = "、".join(self.weak_concepts[:8]) or "（暂无记录）"
        return {
            "recent_questions": recent,
            "weak_concepts": weak,
            "language": self.preferred_language,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_questions": [asdict(e) for e in self.recent_questions],
            "weak_concepts": self.weak_concepts,
            "preferred_language": self.preferred_language,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMemory:
        mem = cls(preferred_language=data.get("preferred_language", "简体中文"))
        for item in data.get("recent_questions", []):
            mem.recent_questions.append(MemoryEntry(**item))
        mem.weak_concepts = list(data.get("weak_concepts", []))
        return mem

    def clear(self) -> None:
        """清空记忆（演示用）"""
        self.recent_questions.clear()
        self.weak_concepts.clear()


def extract_weak_concepts_from_diagnosis(text: str) -> list[str]:
    """
    从诊断回复中启发式提取薄弱概念关键词。
    生产环境可用 LLM structured output 替代。
    """
    concepts: list[str] = []
    markers = ["薄弱", "不足", "需要加强", "建议复习", "掌握不够"]
    for line in text.split("\n"):
        line = line.strip().lstrip("-•0123456789. ")
        if any(m in line for m in markers) and 4 < len(line) < 80:
            # 取冒号或顿号前的短语
            for sep in ["：", ":", "—", "-"]:
                if sep in line:
                    line = line.split(sep)[0]
                    break
            if line:
                concepts.append(line[:30])
    return concepts[:5]


def memory_to_json(memory: SessionMemory) -> str:
    return json.dumps(memory.to_dict(), ensure_ascii=False, indent=2)
