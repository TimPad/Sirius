import asyncio
import aiofiles
import aiofiles.os
import os
import json
from datetime import datetime

import pandas as pd
from openai import OpenAI

# Constants
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"
RESULTS_CSV = "reflection_analysis_results.csv"

async def analyze_reflection_with_deepseek(client: OpenAI, text: str) -> dict:
    """
    Асинхронный вызов DeepSeek API для анализа одной рефлексии.
    """
    error_result = {
        "sentiment_score": 0.0,
        "learning_feedback": "",
        "teamwork_feedback": "",
        "organization_feedback": "",
        "learning_sentiment_score": 0.0,
        "teamwork_sentiment_score": 0.0,
        "organization_sentiment_score": 0.0,
    }
    if not text or not isinstance(text, str) or not text.strip():
        return error_result

    prompt = (
        "Ты — ИИ-ассистент для анализа текстов рефлексии. Проанализируй рефлексию школьника. "
        "Верни JSON с ключами: sentiment_score, learning_feedback, teamwork_feedback, "
        "organization_feedback, learning_sentiment_score, teamwork_sentiment_score, organization_sentiment_score. "
        f"Текст: \"{text}\""
    )

    try:
        # Асинхронный запрос к API
        response = await client.chat.completions.acreate(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        # Гарантируем наличие всех ключей
        for key, default in error_result.items():
            result.setdefault(key, default)
        return result

    except Exception as e:
        print(f"Ошибка при вызове DeepSeek API: {e}")
        return error_result

async def load_previous_results() -> pd.DataFrame:
    """
    Загружает ранее сохранённые результаты, если файл существует.
    """
    if await aiofiles.os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()

async def save_results(df: pd.DataFrame):
    """
    Сохраняет итоговый DataFrame в CSV.
    """
    df.to_csv(RESULTS_CSV, index=False)

async def main(input_excel: str, api_key: str):
    # Подготовка клиента
    client = OpenAI(base_url=DEESEEK_API_URL, api_key=api_key)

    # Загрузка исходных данных
    df = pd.read_excel(input_excel)
    df.columns = [str(col).strip().lower() for col in df.columns]
    df['text'] = df['text'].astype(str).fillna('')

    # Загрузка предыдущих результатов
    prev_df = await load_previous_results()

    # Определяем новые или ещё не обработанные записи
    if not prev_df.empty and 'text' in prev_df.columns:
        processed_texts = set(prev_df['text'])
        new_df = df[~df['text'].isin(processed_texts)].copy()
    else:
        new_df = df.copy()

    # Если нечего обрабатывать, сохраняем и выходим
    if new_df.empty:
        print("Нет новых рефлексий для обработки.")
        return

    # Запускаем анализ асинхронно
    tasks = [analyze_reflection_with_deepseek(client, text)
             for text in new_df['text']]
    results = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(results)

    # Объединяем с new_df
    combined_new = pd.concat([new_df.reset_index(drop=True), results_df], axis=1)

    # Объединяем с ранее сохранёнными
    if not prev_df.empty:
        final_df = pd.concat([prev_df, combined_new], ignore_index=True)
    else:
        final_df = combined_new

    # Сохраняем итог
    await save_results(final_df)
    print(f"Обработано записей: {len(combined_new)}. Всего в файле: {len(final_df)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Async DeepSeek Reflection Analysis")
    parser.add_argument(
        "--input", 
        default=os.environ.get("REFLECTION_INPUT"),
        help="Путь к Excel-файлу с рефлексиями"
    )
    parser.add_argument(
        "--api_key", 
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="API-ключ DeepSeek"
    )
    args = parser.parse_args()

    if not args.input or not args.api_key:
        parser.print_help()
        print("
Пожалуйста, укажите путь к файлу и API-ключ через параметры --input и --api_key 
или установите переменные окружения REFLECTION_INPUT и DEEPSEEK_API_KEY.")
    else:
        asyncio.run(main(args.input, args.api_key))

# requirements.txt
# ----------------
# pandas
# openai
# aiofiles
# asyncio
