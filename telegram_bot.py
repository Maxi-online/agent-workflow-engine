"""Optional text/photo Telegram adapter for the authenticated agent API."""

import base64
import os
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send a question or a photo with a caption.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = {
        "user_id": str(update.effective_user.id),
        "message": update.message.text or update.message.caption or "Describe the supplied image.",
    }
    if update.message.photo:
        photo = update.message.photo[-1]
        if photo.file_size and photo.file_size > 600_000:
            await update.message.reply_text("The photo is too large for this adapter.")
            return
        remote = await photo.get_file()
        raw = await remote.download_as_bytearray()
        if len(raw) > 600_000:
            await update.message.reply_text("The photo is too large for this adapter.")
            return
        data["image"] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    settings = context.application.bot_data
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                settings["api_url"] + "/telegram",
                json=data,
                headers={"Authorization": "Bearer " + settings["api_access"]},
            )
            response.raise_for_status()
            answer = response.json()["answer"]
        for offset in range(0, len(answer), 4000):
            await update.message.reply_text(answer[offset : offset + 4000])
    except (httpx.HTTPError, ValueError, KeyError):
        await update.message.reply_text(
            "The agent request failed. Check the service configuration and retry."
        )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    access = os.getenv("AGENT_API_TOKEN", "")
    if not token or not access:
        raise SystemExit("TELEGRAM_BOT_TOKEN and AGENT_API_TOKEN are required")
    app = Application.builder().token(token).build()
    app.bot_data["api_url"] = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")
    app.bot_data["api_access"] = access
    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, handle_message)
    )
    app.run_polling()


if __name__ == "__main__":
    main()
