from pydantic_settings import BaseSettings
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

class Settings(BaseSettings):
    app_name: str 

    model_config = SettingsConfigDict(env_file=env_path)

settings = Settings()

