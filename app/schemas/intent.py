from typing import Optional

from pydantic import BaseModel, Field


class ParsedIntent(BaseModel):
    """用户自然语言解析后的结构化意图。"""

    work_title: str = Field(..., description="作品名（标准/常用译名）")
    season: Optional[int] = Field(
        None,
        description="标准季数（从 1 开始）；若仅篇章、无明确季数可为 null",
    )
    episode: Optional[int] = Field(None, description="集数；若用户未指定可为 null")
    arc_name: Optional[str] = Field(
        None,
        description="篇章/篇名英文或官方称呼，如 Black Butler 的 Book of Circus、Public School",
    )
    arc_name_zh: Optional[str] = Field(
        None,
        description="篇章中文称呼（若有），如 马戏团篇、寄宿学校篇",
    )
