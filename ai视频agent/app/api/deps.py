from app.core.config import Settings, get_settings


def get_app_settings() -> Settings:
    """FastAPI Depends 用：返回单例配置。"""
    return get_settings()
