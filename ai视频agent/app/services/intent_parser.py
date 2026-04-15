import json
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from openai import OpenAI

from app.core.config import get_openai_base_url, get_settings
from app.schemas.intent import ParsedIntent

SYSTEM_PROMPT = """你是影视与动漫检索意图解析器。将用户模糊、口语化的中文或英文输入，解析为严格的 JSON。

字段含义（必须输出这些键）：
- work_title: 作品的标准或常用名称（中文优先；若用户用英文标题则保留官方英文名）。
- season: 整数，标准「第几季」。若用户只说「第三季」则填 3；若无法从上下文判断季数则填 null。
- episode: 整数，第几集；未提及则 null。
- arc_name: 篇章的英文官方称呼（若适用），否则 null。例如《黑执事》(Black Butler) 常见篇章：
  - 「马戏团篇」/ Book of Circus → arc_name: "Book of Circus"
  - 「寄宿学校篇」/ Public School → arc_name: "Public School"
  - 「幽鬼城篇」/ Book of Murder → arc_name: "Book of Murder"
  - 「大西洋之书」/ Book of the Atlantic → arc_name: "Book of the Atlantic"
- arc_name_zh: 对应的中文篇章名（若可识别），否则 null。

规则：
1. 只输出一个 JSON 对象，不要 Markdown 代码块，不要解释文字。
2. 季数与篇章可能同时存在：例如用户明确「第三季第一集」且作品为黑执事时，结合常识填入 season、episode，并在能对应到官方篇章时填写 arc_name。
3. 若信息不足，对应字段用 null，不要猜测具体数字。"""

JSON_SCHEMA_HINT = """输出格式示例：
{"work_title":"黑执事","season":3,"episode":1,"arc_name":"Book of Circus","arc_name_zh":"马戏团篇"}"""


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_json_object(raw: str) -> Dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    return json.loads(cleaned)


def _coerce_intent(data: Dict[str, Any]) -> ParsedIntent:
    return ParsedIntent(
        work_title=str(data.get("work_title") or "").strip() or "未知作品",
        season=_optional_int(data.get("season")),
        episode=_optional_int(data.get("episode")),
        arc_name=_optional_str(data.get("arc_name")),
        arc_name_zh=_optional_str(data.get("arc_name_zh")),
    )


def _optional_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


async def parse_user_intent(query: str) -> ParsedIntent:
    """
    将用户模糊输入解析为结构化意图（作品名、季、集、篇章名）。
    通过环境变量 INTENT_PROVIDER 选择 openai 或 anthropic。
    """
    settings = get_settings()
    provider = (settings.intent_provider or "openai").lower().strip()
    if provider == "anthropic":
        return await _parse_with_anthropic(query)
    return await _parse_with_openai(query)


async def _parse_with_openai(query: str) -> ParsedIntent:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY，无法调用 OpenAI。")

    client = OpenAI(api_key=settings.openai_api_key, base_url=get_openai_base_url())
    user_content = f"用户输入：\n{query}\n\n{JSON_SCHEMA_HINT}"

    completion = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = completion.choices[0].message.content or "{}"
    data = _parse_json_object(raw)
    return _coerce_intent(data)


async def _parse_with_anthropic(query: str) -> ParsedIntent:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY，无法调用 Claude。")

    client = Anthropic(api_key=settings.anthropic_api_key)
    user_content = f"用户输入：\n{query}\n\n{JSON_SCHEMA_HINT}"

    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
    )

    parts: List[str] = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    raw = "".join(parts) or "{}"
    data = _parse_json_object(raw)
    return _coerce_intent(data)
