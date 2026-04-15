from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="存活探针")
async def health() -> dict[str, str]:
    return {"status": "ok"}
