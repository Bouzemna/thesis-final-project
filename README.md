# thesis-final-project
Final project for my Bachelor's thesis on AI and communication for ALS patients

## Overview

This project builds a personalized AI chatbot system for ALS patients using daily question-and-answer data collected via a Telegram bot. The data is cleaned and processed to fine-tune a generative model (Flan-T5) and a retrieval-based system using Mistral-7B. The aim is to create personalized answers that reflect each user's style and personality.

Two main modeling approaches were explored:
1. **Flan-T5 fine-tuning** on expanded question-answer data
2. **Retrieval-based generation (RAG)** using **Mistral-7B-Instruct** and **cosine similarity** (no FAISS or vector database used)

---

## Repository Contents

### Python Scripts

- `bot.py`: Main script that runs the Telegram bot. It sends 5 daily questions to users at 9:00 AM and stores responses in a local SQLite database (`responses.db`).
- `trigger_questions.py`: If the scheduled job in `bot.py` does not run (e.g., the bot wasn’t active at 9 AM), this script manually triggers the questions for the day. This ensures uninterrupted data collection.
- `export_responses.py`: Extracts all response data from `responses.db` and saves it as `responses_export.csv`.
- `clean_responses.py`: Cleans and standardizes text responses for processing. The result is saved as `cleaned_responses.csv`.

### Jupyter Notebooks

- `FlanT5_FineTuned_QA_Chatbot_withresults_final.ipynb`: Fine-tunes Flan-T5 models on original and expanded datasets. Includes evaluation using BLEU, ROUGE, and BERTScore.
- `Mistral_RAG_Chatbot_testing.ipynb`: Retrieves the most semantically similar Q&A using cosine similarity and generates a contextual response using Mistral-7B-Instruct. Evaluated using the same metrics and human judgment.

### Data Files

- `Questions_capstoneproject.xlsx`: The master list of questions used by the Telegram bot.
- `responses_export.csv`: Exported raw responses from the SQLite database, including questions, answers, user IDs, and timestamps.
- `cleaned_responses.csv`: Preprocessed version of the responses, cleaned and normalized for further analysis and modeling.
- `expanded_qa_dataset.csv`: A manually paraphrased dataset where only the **answers** were expanded and reworded.  
  This dataset was created to enrich the training material for fine-tuning the Flan-T5 model by providing more varied examples of user responses, while keeping the original questions unchanged. 
- `responses.db` *(optional)*: The SQLite database file used to store interactions from the Telegram bot. Not needed for modeling unless starting data collection again.

---

