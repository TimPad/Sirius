
# 1. Импорты и константы
# ----------------------
import os
import json
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# Константы
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"
ARCHIVE_DIR = "archive"
os.makedirs(ARCHIVE_DIR, exist_ok=True)


# 2. Утилиты
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df['text'] = df.get('text', '').astype(str).fillna('')
    return df

def convert_sentiment_to_10_point(score: float) -> float:
    if not isinstance(score, (int, float)):
        return 5.5
    return (score + 1) * 4.5 + 1

def analyze_reflection_with_deepseek(client: OpenAI, text: str) -> dict:
    """Синхронная функция-обёртка для DeepSeek."""
    base = {
        "sentiment_score": 0.0,
        "learning_feedback": "",
        "teamwork_feedback": "",
        "organization_feedback": "",
        "learning_sentiment_score": 0.0,
        "teamwork_sentiment_score": 0.0,
        "organization_sentiment_score": 0.0,
    }
    if not text.strip():
        return base
    prompt = (
        "Ты — ИИ-ассистент для анализа текстов рефлексии. Верни JSON:\n"
        "…ключи как в базовом словаре…\n"
        f"Текст: \"{text}\""
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        for k,v in base.items():
            result.setdefault(k, v)
        return result
    except Exception:
        return base

async def analyze_async(client: OpenAI, texts: list[str]) -> pd.DataFrame:
    """Асинхронно запускает анализ всех текстов и возвращает DataFrame."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, analyze_reflection_with_deepseek, client, t)
        for t in texts
    ]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


# 3. Главная функция
# ----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий")

    # --- Sidebar: выбор источника и запуск ---
    with st.sidebar.form("data_form"):
        st.header("🗂 Источник данных")
        archive_files = sorted([f for f in os.listdir(ARCHIVE_DIR) if f.endswith('.csv')], reverse=True)
        choice = st.selectbox("Выберите анализ или новый:", ["Новый анализ"] + archive_files)
        if choice == "Новый анализ":
            upload = st.file_uploader("Загрузить Excel (.xlsx)", type="xlsx")
            api_key = st.text_input("API-ключ DeepSeek", type="password")
        run_btn = st.form_submit_button("Запустить анализ")

    if not run_btn:
        st.info("Заполните форму и нажмите «Запустить анализ»")
        return

    # --- Загрузка данных и кеширование по CSV ---
    if choice != "Новый анализ":
        df = pd.read_csv(os.path.join(ARCHIVE_DIR, choice), parse_dates=['data'])
        current_name = choice
    else:
        df = load_data(upload)
        base_name = os.path.splitext(upload.name)[0]
        current_name = f"{base_name}_processed.csv"
        csv_path = os.path.join(ARCHIVE_DIR, current_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=['data'])
        else:
            # новый анализ
            client = OpenAI(base_url=DEESEEK_API_URL, api_key=api_key)
            with st.spinner("Идёт анализ..."):
                results_df = asyncio.run(analyze_async(client, df['text'].tolist()))
                df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                # конвертация
                for c in ['sentiment_score','learning_sentiment_score',
                          'teamwork_sentiment_score','organization_sentiment_score']:
                    df[c.replace('_score','_10_point')] = df[c].apply(convert_sentiment_to_10_point)
                df.to_csv(csv_path, index=False)
                st.success(f"Результаты сохранены в {current_name}")

    # --- Основной dashboard ---
    st.sidebar.header("📊 Фильтры")
    if 'data' in df.columns and not df['data'].isna().all():
        min_d, max_d = df['data'].min().date(), df['data'].max().date()
        if min_d < max_d:
            start, end = st.sidebar.slider("Даты", min_d, max_d, (min_d, max_d))
            mask = df['data'].dt.date.between(start, end)
            df = df[mask]
    if df.empty:
        st.error("Нет данных после фильтрации.")
        return

    # ... здесь вставьте оставшуюся логику визуализации без изменений ...
    # (графики, таблицы, тепловые карты, радары и т.д.)
    st.write(df.head())

if __name__ == "__main__":
    main()
