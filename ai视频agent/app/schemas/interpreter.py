from typing import Literal, Optional

from pydantic import BaseModel, Field

MediaType = Literal["anime", "movie", "tv"]


class InterpretedIntent(BaseModel):
    """意图解析器输出的结构化意图（大白话 → JSON）。"""

    title: str = Field(..., description="作品标准译名或常用中文名")
    original_title: Optional[str] = Field(
        None,
        description="原文标题（如日文的「黒執事」或英文官方名 Black Butler）",
    )
    season: Optional[int] = Field(None, ge=1, description="标准季数，未知则为 null")
    episode: Optional[int] = Field(None, ge=1, description="集数，未知则为 null")
    media_type: MediaType = Field(
        ...,
        description="介质类型：anime（电视/Web 动画）/ movie（剧场版）/ tv（真人剧集）",
    )
    season_official_name: Optional[str] = Field(
        None,
        description=(
            "该季或篇章的官方英文名称；如《黑执事》第三季常对应 Book of Circus。"
            "无篇章或无法判断时为 null。"
        ),
    )
