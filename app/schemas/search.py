from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PlatformTag = Literal[
    "bilibili",
    "netflix",
    "youtube",
    "mainstream_other",
    "third_party",
]


class SearchHit(BaseModel):
    """单条搜索结果（已按 PRD 做平台/来源分类）。"""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    platform: PlatformTag = Field(
        ...,
        description="主流平台或第三方；第三方含非官方聚合站等，仅作结构化标注。",
    )
    source: Literal["tavily", "serper"] = "tavily"
    access_note: Optional[str] = Field(
        None,
        description="访问提示，如海外模式下对纯国内站标注可能需要回国 VPN。",
    )
    display_domain: Optional[str] = Field(
        None,
        description="规范化主机名，用于「域名来源」展示。",
    )
    official_recommend_badge: bool = Field(
        False,
        description="B 站或 YouTube：前端展示「官方/推荐」。",
    )
    youtube_recommend_badge: bool = Field(
        False,
        description="兼容旧前端；请优先使用 official_recommend_badge。",
    )
    third_party_ad_caution: bool = Field(
        False,
        description="非 B 站/YouTube 时为 True，前端展示第三方资源与风险提示。",
    )
    bilibili_hot_badge: bool = Field(
        False,
        description="兼容旧前端；请优先使用 official_recommend_badge。",
    )


class SearchResponse(BaseModel):
    """搜索接口返回：候选链接 + 前端提示标记。"""

    hits: List[SearchHit] = Field(
        ...,
        description="默认可直接展示的结果（通常最多 8 条）。",
    )
    more_hits: List[SearchHit] = Field(
        default_factory=list,
        description="匹配度较低或来源一般的补充链接，建议前端折叠展示并提示风险。",
    )
    more_hits_notice: Optional[str] = Field(
        None,
        description="展示 more_hits 时的统一提示文案。",
    )
    show_domestic_vpn_hint: bool = Field(
        False,
        description="为 True 时建议前端提示：海外用户部分链接可能需回国 VPN。",
    )
