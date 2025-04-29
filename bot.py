import logging
import pandas as pd
import random
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import asyncio
import nest_asyncio
nest_asyncio.apply()
import sqlite3



# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Your Telegram Bot Token
TOKEN = '7930761716:AAEYNnMhmq7d_87Bywh3Sa6L8Q3Q0QUwqK4'


def initialize_database():
    conn = sqlite3.connect("responses.db")
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            response TEXT,
            timestamp DATETIME
        )
    ''')

    conn.commit()
    conn.close()
def initialize_participants():
    conn = sqlite3.connect("responses.db")
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS participants (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT
        )
    ''')

    conn.commit()
    conn.close()


# Call this at the start of the main function
initialize_database()
initialize_participants()



# Load questions from Excel
def load_questions_from_excel():
    df = pd.read_excel('Questions_capstoneproject.xlsx')  # Ensure the file is in the same folder as bot.py
    return df['Questions'].dropna().tolist()

# Load all questions
all_questions = load_questions_from_excel()

# Store user-specific data
user_data = {}

# Set of registered participants
participants = set()

# /start command to register users
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username or "N/A"
    first_name = user.first_name or "N/A"
    last_name = user.last_name or "N/A"

    # Save participant in the database
    conn = sqlite3.connect("responses.db")
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR IGNORE INTO participants (user_id, username, first_name, last_name)
        VALUES (?, ?, ?, ?)
    ''', (user_id, username, first_name, last_name))

    conn.commit()
    conn.close()

    await update.message.reply_text("✅ You are now registered to receive your five daily questions!")

    # Generate daily questions for the user
    user_data[user_id] = {
        'current_question_index': 0,
        'daily_questions': generate_daily_questions(),
        'responses': {},
        'last_question_date': datetime.now().date()
    }

    first_question = user_data[user_id]['daily_questions'][0]
    await update.message.reply_text(f"Question 1/5: {first_question}")


# Generate 5 random questions for the day
def generate_daily_questions():
    return random.sample(all_questions, 5)

# Automatically send daily questions to participants
# Automatically send daily questions to participants
async def send_daily_questions(bot=None, context=None):
    today = datetime.now().date()

    # Fetch all participants from the DB
    conn = sqlite3.connect("responses.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM participants")
    user_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    for user_id in user_ids:
        if user_id not in user_data:
            user_data[user_id] = {
                'current_question_index': 0,
                'daily_questions': generate_daily_questions(),
                'responses': {},
                'last_question_date': today
            }

        user_data[user_id]['daily_questions'] = generate_daily_questions()
        user_data[user_id]['current_question_index'] = 0
        user_data[user_id]['last_question_date'] = today

        # Utiliser soit `bot` pour un trigger manuel, soit `context.bot` pour un trigger auto
        bot_instance = bot if bot else context.bot

        await bot_instance.send_message(chat_id=user_id, text="Here are your five questions for today.")
        first_question = user_data[user_id]['daily_questions'][0]
        await bot_instance.send_message(chat_id=user_id, text=f"Question 1/5: {first_question}")

# Send a reminder if the user hasn't completed all questions
async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    today = str(datetime.now().date())

    for user_id in participants:
        if user_id in user_data and len(user_data[user_id]['responses'].get(today, [])) < 5:
            await context.bot.send_message(chat_id=user_id, text="Reminder: You still have questions left to answer today.")

# Handle user responses
# Handle user responses
async def handle_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = update.message.text
    today = str(datetime.now().date())

    # Vérifie que l'utilisateur a bien été enregistré
    if user_id not in user_data:
        await update.message.reply_text("❌ Error: No user data found. Type /start to restart.")
        return

    # Vérifie que les questions du jour existent
    if 'daily_questions' not in user_data[user_id] or not user_data[user_id]['daily_questions']:
        await update.message.reply_text("❌ Error: No questions available. Type /start to restart.")
        return

    # Vérifie que la clé de réponses du jour est bien initialisée
    if today not in user_data[user_id]['responses']:
        user_data[user_id]['responses'][today] = []

    # Récupère l'index actuel de la question
    current_index = user_data[user_id]['current_question_index']

    # Vérifie que l'index ne dépasse pas la liste des questions
    if current_index >= len(user_data[user_id]['daily_questions']):
        await update.message.reply_text("✅ You have answered all today's questions!")
        return

    # Récupère la question actuelle
    question = user_data[user_id]['daily_questions'][current_index]

    # Sauvegarde la réponse
    user_data[user_id]['responses'][today].append(message)

    # Enregistre la réponse dans SQLite
    conn = sqlite3.connect("responses.db")
    cursor = conn.cursor()
    timestamp = datetime.now()

    cursor.execute('''
        INSERT INTO responses (user_id, question, response, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (user_id, question, message, timestamp))

    conn.commit()
    conn.close()

    # Incrémente l'index après avoir enregistré la réponse
    user_data[user_id]['current_question_index'] += 1

    # Envoie la prochaine question s'il en reste
    if user_data[user_id]['current_question_index'] < len(user_data[user_id]['daily_questions']):
        next_question = user_data[user_id]['daily_questions'][user_data[user_id]['current_question_index']]
        await update.message.reply_text(f"Question {user_data[user_id]['current_question_index'] + 1}/5: {next_question}")
    else:
        await update.message.reply_text("✅ Thank you for answering all of today's questions!")



async def main():
    application = ApplicationBuilder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_response))

    # Scheduler for daily questions and reminders
    scheduler = BackgroundScheduler()

    # Send daily questions at 9:00 AM
    scheduler.add_job(
        lambda: asyncio.run(send_daily_questions(ContextTypes.DEFAULT_TYPE)),
        'cron',
        hour=9,
        minute=0
    )

    # Send reminders at 6:00 PM if not all questions are answered
    scheduler.add_job(
        lambda: asyncio.run(send_reminder(ContextTypes.DEFAULT_TYPE)),
        'cron',
        hour=18,
        minute=0
    )

    scheduler.start()

    # Start the bot
    print("The bot is now running. Daily questions will be sent automatically.")
    await application.run_polling()

if __name__ == '__main__':
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    finally:
        if not loop.is_closed():
            loop.close()

