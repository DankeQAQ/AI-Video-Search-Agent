from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.interpreter import InterpretedIntent
from app.services.interpreter import get_interpreter

router = APIRouter()


class ParseRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户原始输入")


@router.post("/parse", response_model=InterpretedIntent, summary="自然语言 → 结构化意图")
async def parse_intent(body: ParseRequest) -> InterpretedIntent:
    try:
        return await get_interpreter().interpret(body.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"模型返回无法解析为 JSON: {e}") from e
