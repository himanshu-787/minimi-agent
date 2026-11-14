from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    model_name: str = "claude-sonnet-5-5"
    search_api_key: str = ""

    db_path: str = "data/chat.db"
    request_timeout_s: float = 20.0   # never let a slow call hang forever
    max_agent_steps: int = 4          # stops infinite tool-calling loops


settings = Settings()