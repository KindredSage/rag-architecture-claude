from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # mTLS paths
    CLIENT_CERT: str        # path to client.crt
    CLIENT_KEY: str         # path to client.key
    CA_CERT: str            # path to ca.crt

    # Backend
    BACKEND_BASE_URL: str   # your LLM backend base URL e.g. https://internal-llm.corp
    TOKEN_URL: str          # token endpoint e.g. https://auth.corp/token

    # Token lifecycle
    TOKEN_EXPIRY_BUFFER: int = 60   # refresh this many seconds before actual expiry

    # Gateway auth — this is what OpenWebUI sends to THIS gateway
    GATEWAY_API_KEY: str = "changeme"

    class Config:
        env_file = ".env"


settings = Settings()
