import asyncio
import sqlite3
from bot import send_daily_questions, TOKEN
from telegram.ext import ApplicationBuilder

# Fetch participants from the database
def get_participants():
    conn = sqlite3.connect("responses.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM participants")
    participants = [row[0] for row in cursor.fetchall()]
    conn.close()
    return participants

# Print current participants
participants = get_participants()
print("Participants:", participants)

# Trigger the questions manually
async def trigger_questions():
    if participants:
        app = ApplicationBuilder().token(TOKEN).build()
        await app.initialize()
        await send_daily_questions(bot=app.bot)
    else:
        print("❌ No participants found. Make sure they have done /start.")

asyncio.run(trigger_questions())

