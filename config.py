import os

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Groq API Key (для LLM)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# OpenAI API Key (для Whisper транскрибации)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
