from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_openai_base_url, get_settings
from app.schemas.interpreter import InterpretedIntent

JSON_FORMAT_RULE = """
【输出格式硬性要求】
请务必只返回一个 JSON 对象的字符串，不要包含任何 Markdown 代码块标签（如 ```json 或 ```）或任何解释说明文字。
JSON 的键必须恰好为：title, original_title, season, episode, media_type, season_official_name；无法确定的字段使用 null。
除上述 JSON 外不要输出任何其它字符。"""

SYSTEM_PROMPT = """你是影视与动漫意图解析器。根据用户口语化输入，输出结构化字段。

字段要求：
- title：作品标准译名或常用中文名（如 黑执事）。
- original_title：官方原文标题；动画常用日文「黒執事」或英文 Black Butler；不确定则 null。
- season / episode：整数；用户未提则 null。不要臆造集数。
- media_type：必须是 anime、movie、tv 之一。
  - anime：日本动画剧集、OVA/Web 动画连续剧。
  - movie：剧场版、电影。
  - tv：真人电视剧（非动画）。
- season_official_name：当作品按「季」或「篇章」有官方英文称呼时填写，否则 null。

《黑执事》(Black Butler / 黒執事) 特别规则（务必结合语义补全）：
- 用户说「第三季」「第3季」「S3」等且指本作品时：season=3，season_official_name="Book of Circus"（对应「马戏团篇」）。
- 「马戏团篇」「Book of Circus」：season 通常为 3（与中文语境中的第三季一致），season_official_name="Book of Circus"。
- 「寄宿学校篇」：season_official_name="Public School"。
- 「幽鬼城篇」：season_official_name="Book of Murder"。
- 「大西洋之书」篇：season_official_name="Book of the Atlantic"。
若用户只说「黑执事」未提季集，season、episode、season_official_name 均可为 null。

只依据用户输入与常识推断；不确定的字段用 null，不要编造。"""


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _message_text(msg: BaseMessage) -> str:
    if isinstance(msg, AIMessage) or hasattr(msg, "content"):
        c: Any = getattr(msg, "content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts: list[str] = []
            for block in c:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
    return str(msg)


class IntentInterpreter:
    """使用 LangChain + OpenAI 兼容接口将自然语言解析为 `InterpretedIntent`（纯文本 JSON，兼容无 response_format 的网关）。"""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("未设置 OPENAI_API_KEY，IntentInterpreter 无法调用模型。")

        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=get_openai_base_url(),
            temperature=0.2,
        )

    async def interpret(self, user_input: str) -> InterpretedIntent:
        """解析用户自然语言，返回结构化意图。"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n" + JSON_FORMAT_RULE),
            HumanMessage(content=f"用户输入：\n{user_input.strip()}"),
        ]
        raw_msg = await self._llm.ainvoke(messages)
        text = _message_text(raw_msg)
        cleaned = _strip_json_fences(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"模型返回不是合法 JSON: {e}; 原始片段: {text[:500]!r}") from e
        if not isinstance(data, dict):
            raise ValueError("模型返回的 JSON 根类型必须是对象")
        return InterpretedIntent.model_validate(data)


@lru_cache
def get_interpreter() -> IntentInterpreter:
    """进程内单例，避免重复构建 LangChain Runnable。"""
    return IntentInterpreter()
