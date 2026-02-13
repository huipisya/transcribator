"""
Telegram бот-транскрибатор с 3 режимами обработки
"""
import asyncio
import tempfile
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

from config import TELEGRAM_BOT_TOKEN, GROQ_API_KEY, OPENAI_API_KEY

# Хранение настроек пользователей (user_id -> mode)
user_settings: dict[int, str] = {}

# Глобальная инструкция для всех режимов
GLOBAL_INSTRUCTION = """
ВАЖНО: Это транскрипция голосового сообщения. Твоя задача — сделать так, чтобы текст выглядел как написанный от руки, а не как типичная транскрипция. 
Сохрани эмоции, интонацию и живость речи автора. Текст должен быть приятным для чтения, но при этом передавать характер и настроение говорящего.
"""

# Режимы обработки
MODES = {
    "transcribe": {
        "name": "📝 Транскрипция",
        "short": "Транскрипция",
        "description": "Выдаю транскрипцию, как Телеграм премиум, но бесплатно и с верной пунктуацией.",
        "prompt": "Исправь грамматические и пунктуационные ошибки. Сохрани оригинальную структуру, эмоции и живость речи. Текст должен выглядеть как написанный от руки. Верни только исправленный текст."
    },
    "cosmetic": {
        "name": "✨ Косметические изменения",
        "short": "Косметические изменения",
        "description": "Убираю междометия, разделяю на абзацы. Конструктивный тон в стиле инфостиля Ильяхова.",
        "prompt": "Отредактируй текст: убери междометия и слова-паразиты, раздели на абзацы, исправь грамматику. Сохрани эмоции, интонацию и характер автора — текст должен звучать живо, как написанный от руки. Тон — конструктивный. Не пиши слишком формально, не создавай лишней дистанции с читателем, но обходись без панибратства. Чаще используй глаголы, опирайся на «инфостиль» Максима Ильяхова. Верни только отредактированный текст."
    },
    "notes": {
        "name": "📋 Заметки/сообщения",
        "short": "Заметки/сообщения",
        "description": "Конструктивный тон в стиле инфостиля Ильяхова. Подойдёт для отправки сообщения коллеге или текстовой заметки для себя.",
        "prompt": """Преобразуй в качественную структурированную заметку.

КРИТИЧЕСКИ ВАЖНО - ФОРМАТИРОВАНИЕ:
- ИСПОЛЬЗУЙ ТОЛЬКО HTML ТЕГИ: <b>жирный</b> и <i>курсив</i>
- НИКОГДА НЕ ИСПОЛЬЗУЙ Markdown! Запрещено: **текст**, *текст*, __текст__
- Используй • для списков (НЕ *, НЕ -)
- Используй эмодзи: 📌, ✅, 💡, 📝, ⚡
- ВСЕГДА ставь ПРОБЕЛ после эмодзи перед текстом или тегами!
  ✓ Правильно: "✅ <b>Выводы:</b>" 
  ✗ Неправильно: "✅<b>Выводы:</b>" или "✅ **Выводы:**"
- Разделяй абзацы пустой строкой

СТРУКТУРА:
1. 📌 <b>Краткий заголовок заметки</b>
2. <b>Ключевые мысли:</b> (списком с •)
3. ✅ <b>Выводы или действия</b> (если есть)

Тон — конструктивный. Не пиши слишком формально, не создавай лишней дистанции с читателем, но обходись без панибратства. Чаще используй глаголы, опирайся на «инфостиль» Максима Ильяхова.
Верни ТОЛЬКО готовую заметку, используя ТОЛЬКО HTML теги для форматирования."""
    }
}

DEFAULT_MODE = None  # Пока не выбран режим


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Транскрибация аудио через OpenAI Whisper API"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": ("audio.ogg", audio_bytes, "audio/ogg")},
            data={"model": "whisper-1", "language": "ru"}
        )
        response.raise_for_status()
        return response.json()["text"]


async def process_with_llm(text: str, mode: str) -> str:
    """Обработка текста через Groq LLM"""
    system_prompt = GLOBAL_INSTRUCTION + "\n" + MODES[mode]["prompt"]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def get_mode_selection_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора режима"""
    buttons = []
    for mode_id, mode_data in MODES.items():
        buttons.append([InlineKeyboardButton(
            f"— {mode_data['short']}", 
            callback_data=f"select:{mode_id}"
        )])
    return InlineKeyboardMarkup(buttons)


def get_change_mode_keyboard() -> ReplyKeyboardMarkup:
    """Постоянная кнопка смены режима"""
    return ReplyKeyboardMarkup(
        [[KeyboardButton("🔄 Изменить режим")]],
        resize_keyboard=True
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    user_id = update.effective_user.id
    
    # Сбрасываем режим при /start
    if user_id in user_settings:
        del user_settings[user_id]
    
    await update.message.reply_text(
        "👋 Привет! Я расширенный транскрибатор голосовых сообщений с несколькими режимами работы.\n\n"
        "• **Транскрипция** — выдаю транскрипцию, как Телеграм премиум, но бесплатно и с верной пунктуацией.\n\n"
        "• **Косметические изменения** — убираю междометия, разделяю на абзацы и очищаю текст.\n\n"
        "• **Заметки/сообщения** — более официальный и ёмкий тон. Подойдёт для отправки сообщения коллеге или текстовой заметки для себя.\n\n"
        "Выбери режим, в котором хочешь работать сейчас 👇",
        reply_markup=get_mode_selection_keyboard(),
        parse_mode="Markdown"
    )


async def change_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка кнопки 'Изменить режим'"""
    await update.message.reply_text(
        "Выбери новый режим работы 👇",
        reply_markup=get_mode_selection_keyboard()
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на кнопки"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if query.data.startswith("select:"):
        # Выбор режима
        new_mode = query.data.split(":")[1]
        user_settings[user_id] = new_mode
        
        await query.edit_message_text(
            f"✅ Отлично! Режим «{MODES[new_mode]['short']}» выбран.\n\n"
            f"{MODES[new_mode]['description']}\n\n"
            "Теперь отправь мне голосовое сообщение 🎙️",
            parse_mode="Markdown"
        )
        
        # Добавляем постоянную кнопку смены режима
        await context.bot.send_message(
            chat_id=user_id,
            text="Кнопка для смены режима всегда доступна 👇",
            reply_markup=get_change_mode_keyboard()
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосовых сообщений"""
    user_id = update.effective_user.id
    mode = user_settings.get(user_id)
    
    # Если режим не выбран, просим выбрать
    if mode is None:
        await update.message.reply_text(
            "⚠️ Сначала выбери режим работы 👇",
            reply_markup=get_mode_selection_keyboard()
        )
        return
    
    # Отправляем статус
    status_msg = await update.message.reply_text("🎙️ Транскрибирую...")
    
    try:
        # Скачиваем голосовое
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
        
        # Транскрибируем
        await status_msg.edit_text("🎙️ Транскрибирую... ✅\n✍️ Обрабатываю текст...")
        raw_text = await transcribe_audio(bytes(audio_bytes))
        
        # Обрабатываем через LLM
        result = await process_with_llm(raw_text, mode)
        
        # Отправляем результат
        await status_msg.delete()
        # Используем HTML для правильного отображения форматирования
        try:
            await update.message.reply_text(result, parse_mode="HTML")
        except Exception:
            # Если HTML не парсится, отправляем как обычный текст
            await update.message.reply_text(result)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка: {e}")


def main():
    """Запуск бота"""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Regex("^🔄 Изменить режим$"), change_mode))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    print("🤖 Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()
