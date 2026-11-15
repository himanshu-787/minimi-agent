from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    model_name: str = "qwen/qwen3.8-27b"
    search_api_key: str = ""

    db_path: str = "data/chat.db"
    request_timeout_s: float = 20.0   # never let a slow call hang forever
    max_agent_steps: int = 4          # stops infinite tool-calling loops

    search_max_results: int = 5       # fewer results = faster + cheaper
    max_output_tokens: int = 1024     # cap answer length

    max_history_messages: int = 20          # context window for one conversation
    digest_size: int = 15                   # recent questions from other chats
    display_timezone: str = "Asia/Kolkata"  # what the model and user see

    search_timeout_s: float = 8.0     # per attempt; short so retries still fit in budget
    search_retries: int = 2           # up to 3 attempts total
    search_backoff_s: float = 0.4     # waits 0.4s, then 0.8s
    search_cache_ttl_s: int = 300     # cache lifetime: 5 minutes
    search_cache_max: int = 256       # max cached queries
    llm_max_retries: int = 2          # the Groq SDK retries 429/5xx itself

    rate_limit_per_min: int = 20      # requests per IP per minute


settings = Settings()
