"""
Telegram бот-транскрибатор с режимами обработки и кастомными промптами
Данные пользователей сохраняются в PostgreSQL (через DATABASE_URL).
"""
import asyncio
import json
import os
import tempfile
import httpx
import psycopg2
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)

from config import TELEGRAM_BOT_TOKEN, GROQ_API_KEY, OPENAI_API_KEY

MAX_CUSTOM_PROMPTS = 3

# --- PostgreSQL ---

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_db():
    """Получить соединение с PostgreSQL"""
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Создать таблицу user_data, если не существует"""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_data (
                    user_id BIGINT PRIMARY KEY,
                    data JSONB NOT NULL DEFAULT '{}'
                )
            """)
            conn.commit()
            # Отладка: показываем содержимое таблицы при старте
            cur.execute("SELECT user_id, data FROM user_data")
            rows = cur.fetchall()
            print(f"📊 БД: найдено {len(rows)} пользователей")
            for user_id, data in rows:
                prompts = data.get("custom_prompts", [])
                mode = data.get("mode", "нет")
                print(f"   👤 {user_id}: режим={mode}, промптов={len(prompts)}")
    finally:
        conn.close()


def load_user_data(user_id: int) -> dict:
    """Загрузить данные пользователя из БД"""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM user_data WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            return row[0] if row else {}
    finally:
        conn.close()


def save_user_data(user_id: int, data: dict):
    """Сохранить данные пользователя в БД"""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_data (user_id, data)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET data = EXCLUDED.data
            """, (user_id, json.dumps(data, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()


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
    "custom_prompt": {
        "name": "🎯 Свой промпт",
        "short": "Свой промпт",
        "description": "Создай свои правила обработки текста (до 3 промптов)."
    }
}


# --- Вспомогательные функции для работы с user_data (PostgreSQL) ---

def get_user_mode(user_id: int) -> str | None:
    """Получить текущий режим пользователя"""
    data = load_user_data(user_id)
    return data.get("mode")


def set_user_mode(user_id: int, mode: str):
    """Установить режим пользователя"""
    data = load_user_data(user_id)
    data["mode"] = mode
    save_user_data(user_id, data)


def clear_user_mode(user_id: int):
    """Сбросить режим пользователя"""
    data = load_user_data(user_id)
    data.pop("mode", None)
    save_user_data(user_id, data)


def get_custom_prompts(user_id: int) -> list[dict]:
    """Получить список кастомных промптов пользователя"""
    data = load_user_data(user_id)
    return data.get("custom_prompts", [])


def add_custom_prompt(user_id: int, name: str, prompt: str) -> int:
    """Добавить кастомный промпт и вернуть его индекс"""
    data = load_user_data(user_id)
    if "custom_prompts" not in data:
        data["custom_prompts"] = []
    data["custom_prompts"].append({"name": name, "prompt": prompt})
    save_user_data(user_id, data)
    return len(data["custom_prompts"]) - 1


def delete_custom_prompt(user_id: int, idx: int) -> bool:
    """Удалить кастомный промпт по индексу. Возвращает True если успешно."""
    data = load_user_data(user_id)
    prompts = data.get("custom_prompts", [])
    if 0 <= idx < len(prompts):
        prompts.pop(idx)
        # Если текущий режим указывал на удалённый или сдвинутый промпт — сбрасываем
        mode = data.get("mode")
        if mode and mode.startswith("custom_prompt:"):
            old_idx = int(mode.split(":")[1])
            if old_idx == idx:
                data.pop("mode", None)
            elif old_idx > idx:
                data["mode"] = f"custom_prompt:{old_idx - 1}"
        save_user_data(user_id, data)
        return True
    return False


def get_pending_action(user_id: int) -> dict | None:
    """Получить pending action пользователя"""
    data = load_user_data(user_id)
    return data.get("pending_action")


def set_pending_action(user_id: int, action: dict):
    """Установить pending action"""
    data = load_user_data(user_id)
    data["pending_action"] = action
    save_user_data(user_id, data)


def clear_pending_action(user_id: int):
    """Очистить pending action"""
    data = load_user_data(user_id)
    data.pop("pending_action", None)
    save_user_data(user_id, data)


# --- Основные функции бота ---

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


async def process_with_llm(text: str, mode: str, user_id: int = None) -> str:
    """Обработка текста через Groq LLM"""
    if mode.startswith("custom_prompt:"):
        # Кастомный промпт — берём из пользовательских данных
        idx = int(mode.split(":")[1])
        prompts = get_custom_prompts(user_id)
        if idx < len(prompts):
            user_prompt = prompts[idx]["prompt"]
        else:
            user_prompt = "Исправь текст."
        system_prompt = GLOBAL_INSTRUCTION + "\n" + user_prompt
    else:
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


def get_custom_prompts_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура выбора кастомного промпта"""
    buttons = []
    prompts = get_custom_prompts(user_id)
    
    for i, p in enumerate(prompts):
        buttons.append([InlineKeyboardButton(
            f"📄 {p['name']}", 
            callback_data=f"use_custom:{i}"
        )])
    
    # Кнопка "Создать новый" — только если меньше максимума
    if len(prompts) < MAX_CUSTOM_PROMPTS:
        buttons.append([InlineKeyboardButton(
            "➕ Создать новый промпт", 
            callback_data="new_custom"
        )])
    
    # Кнопка "Удалить промпт" — только если есть что удалять
    if prompts:
        buttons.append([InlineKeyboardButton(
            "🗑 Удалить промпт", 
            callback_data="delete_custom"
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
    clear_user_mode(user_id)
    # Очищаем pending action
    clear_pending_action(user_id)
    
    await update.message.reply_text(
        "👋 Привет! Я расширенный транскрибатор голосовых сообщений с несколькими режимами работы.\n\n"
        "• **Транскрипция** — выдаю транскрипцию, как Телеграм премиум, но бесплатно и с верной пунктуацией.\n\n"
        "• **Косметические изменения** — убираю междометия, разделяю на абзацы и очищаю текст.\n\n"
        "• **Свой промпт** — создай свои правила обработки текста (до 3 промптов).\n\n"
        "Выбери режим, в котором хочешь работать сейчас 👇",
        reply_markup=get_mode_selection_keyboard(),
        parse_mode="Markdown"
    )


async def change_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка кнопки 'Изменить режим'"""
    user_id = update.effective_user.id
    # Очищаем pending action при смене режима
    clear_pending_action(user_id)
    
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
        new_mode = query.data.split(":")[1]
        
        if new_mode == "custom_prompt":
            # Проверяем, есть ли сохранённые промпты
            prompts = get_custom_prompts(user_id)
            
            if prompts:
                # Показываем список промптов
                await query.edit_message_text(
                    "🎯 Выбери свой промпт или создай новый 👇",
                    reply_markup=get_custom_prompts_keyboard(user_id)
                )
            else:
                # Нет промптов — сразу создаём новый
                set_pending_action(user_id, {"action": "awaiting_name"})
                await query.edit_message_text(
                    "🎯 У тебя пока нет своих промптов. Давай создадим!\n\n"
                    "Напиши **название** для нового промпта:",
                    parse_mode="Markdown"
                )
        else:
            # Обычный режим
            set_user_mode(user_id, new_mode)
            
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
    
    elif query.data.startswith("use_custom:"):
        # Выбор существующего кастомного промпта
        idx = int(query.data.split(":")[1])
        prompts = get_custom_prompts(user_id)
        
        if idx < len(prompts):
            set_user_mode(user_id, f"custom_prompt:{idx}")
            prompt_name = prompts[idx]["name"]
            
            await query.edit_message_text(
                f"✅ Промпт «{prompt_name}» выбран!\n\n"
                "Теперь отправь мне голосовое сообщение 🎙️"
            )
            
            await context.bot.send_message(
                chat_id=user_id,
                text="Кнопка для смены режима всегда доступна 👇",
                reply_markup=get_change_mode_keyboard()
            )
        else:
            await query.edit_message_text("❌ Промпт не найден. Попробуй ещё раз.")
    
    elif query.data == "new_custom":
        # Создание нового кастомного промпта
        prompts = get_custom_prompts(user_id)
        
        if len(prompts) >= MAX_CUSTOM_PROMPTS:
            await query.edit_message_text(
                f"❌ Достигнут лимит ({MAX_CUSTOM_PROMPTS} промпта). "
                "Удали существующий, чтобы создать новый."
            )
            return
        
        set_pending_action(user_id, {"action": "awaiting_name"})
        await query.edit_message_text(
            "📝 Напиши **название** для нового промпта:",
            parse_mode="Markdown"
        )
    
    elif query.data == "delete_custom":
        # Показываем список промптов для удаления
        prompts = get_custom_prompts(user_id)
        
        if not prompts:
            await query.edit_message_text("У тебя нет сохранённых промптов.")
            return
        
        buttons = []
        for i, p in enumerate(prompts):
            buttons.append([InlineKeyboardButton(
                f"🗑 {p['name']}",
                callback_data=f"delete_confirm:{i}"
            )])
        buttons.append([InlineKeyboardButton(
            "↩️ Назад",
            callback_data="select:custom_prompt"
        )])
        
        await query.edit_message_text(
            "Какой промпт удалить? 👇",
            reply_markup=InlineKeyboardMarkup(buttons)
        )
    
    elif query.data.startswith("delete_confirm:"):
        # Удаление конкретного промпта
        idx = int(query.data.split(":")[1])
        prompts = get_custom_prompts(user_id)
        
        if idx < len(prompts):
            deleted_name = prompts[idx]["name"]
            delete_custom_prompt(user_id, idx)
            
            # Возвращаем в меню "Свой промпт"
            remaining_prompts = get_custom_prompts(user_id)
            if remaining_prompts:
                await query.edit_message_text(
                    f"✅ Промпт «{deleted_name}» удалён.\n\n"
                    "Выбери промпт или создай новый 👇",
                    reply_markup=get_custom_prompts_keyboard(user_id)
                )
            else:
                await query.edit_message_text(
                    f"✅ Промпт «{deleted_name}» удалён.\n\n"
                    "У тебя больше нет сохранённых промптов.",
                    reply_markup=get_custom_prompts_keyboard(user_id)
                )
        else:
            await query.edit_message_text("❌ Промпт не найден.")


async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений (для создания кастомных промптов)"""
    user_id = update.effective_user.id
    pending = get_pending_action(user_id)
    
    if not pending:
        return  # Нет ожидающего действия — игнорируем
    
    text = update.message.text.strip()
    
    if pending["action"] == "awaiting_name":
        # Получили название промпта
        set_pending_action(user_id, {"action": "awaiting_prompt", "name": text})
        await update.message.reply_text(
            f"👍 Название: «{text}»\n\n"
            "Теперь напиши **текст промпта** — инструкцию, как обрабатывать текст:",
            parse_mode="Markdown"
        )
    
    elif pending["action"] == "awaiting_prompt":
        # Получили текст промпта
        name = pending["name"]
        
        new_idx = add_custom_prompt(user_id, name, text)
        
        # Устанавливаем новый промпт как активный режим
        set_user_mode(user_id, f"custom_prompt:{new_idx}")
        
        # Очищаем pending action
        clear_pending_action(user_id)
        
        remaining = MAX_CUSTOM_PROMPTS - len(get_custom_prompts(user_id))
        
        await update.message.reply_text(
            f"✅ Промпт «{name}» создан и выбран!\n\n"
            f"Осталось свободных слотов: {remaining}/{MAX_CUSTOM_PROMPTS}\n\n"
            "Теперь отправь мне голосовое сообщение 🎙️",
            reply_markup=get_change_mode_keyboard()
        )


async def send_long_message(message, text: str, parse_mode: str = None):
    """Отправка длинного сообщения частями (лимит Telegram — 4096 символов)"""
    MAX_LENGTH = 4096
    
    if len(text) <= MAX_LENGTH:
        await message.reply_text(text, parse_mode=parse_mode)
        return
    
    # Разбиваем на части
    for i in range(0, len(text), MAX_LENGTH):
        chunk = text[i:i + MAX_LENGTH]
        try:
            await message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            await message.reply_text(chunk)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосовых сообщений"""
    user_id = update.effective_user.id
    mode = get_user_mode(user_id)
    
    # Если режим не выбран, просим выбрать
    if mode is None:
        await update.message.reply_text(
            "⚠️ Сначала выбери режим работы 👇",
            reply_markup=get_mode_selection_keyboard()
        )
        return
    
    # Отправляем статус
    status_msg = await update.message.reply_text("🎙️ Транскрибирую...")
    status_deleted = False
    
    try:
        # Скачиваем голосовое
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
        
        # Транскрибируем
        await status_msg.edit_text("🎙️ Транскрибирую... ✅\n✍️ Обрабатываю текст...")
        raw_text = await transcribe_audio(bytes(audio_bytes))
        
        # Обрабатываем через LLM
        result = await process_with_llm(raw_text, mode, user_id=user_id)
        
        # Отправляем результат
        await status_msg.delete()
        status_deleted = True
        
        # Отправляем (с разбивкой на части если текст длинный)
        try:
            await send_long_message(update.message, result, parse_mode="HTML")
        except Exception:
            await send_long_message(update.message, result)
        
    except Exception as e:
        if not status_deleted:
            try:
                await status_msg.edit_text(f"❌ Ошибка: {e}")
            except Exception:
                await update.message.reply_text(f"❌ Ошибка: {e}")
        else:
            await update.message.reply_text(f"❌ Ошибка: {e}")


def main():
    """Запуск бота"""
    # Инициализируем таблицу в PostgreSQL
    init_db()
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Regex("^🔄 Изменить режим$"), change_mode))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    # Обработка текста для создания кастомных промптов (после всех остальных)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex("^🔄 Изменить режим$"), handle_text_input))
    
    print("🤖 Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()
