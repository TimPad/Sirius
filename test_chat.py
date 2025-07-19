import subprocess
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from openai import AsyncOpenAI
import asyncio
import numpy as np
import os
from supabase import create_client, Client

# Константы для DeepSeek API
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"

# ----------------------
# Функции загрузки и инициализации
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(col).strip().lower() if isinstance(col, str) else col for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

@st.cache_resource
def init_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError as e:
        st.error(f"Ошибка: ключ '{e.args[0]}' не найден в секретах Streamlit.")
        return None

# ----------------------
# Существующая DeepSeek анализ функция
# ----------------------
async def analyze_reflection_with_deepseek(client: AsyncOpenAI, text: str) -> dict:
    error_result = {
        "sentiment_score": 0.0,
        "learning_feedback": "N/A",
        "teamwork_feedback": "N/A",
        "organization_feedback": "N/A",
        "learning_sentiment_score": 0.0,
        "teamwork_sentiment_score": 0.0,
        "organization_sentiment_score": 0.0,
    }
    if not text or not isinstance(text, str) or not text.strip():
        return error_result
    prompt = (
        "Ты — ИИ-ассистент для анализа текстов рефлексии. Проанализируй рефлексию школьника. "
        "Твоя задача — вернуть JSON-объект со следующими ключами:\n"
        "1. 'sentiment_score': общая тональность текста, число от -1.0 (негатив) до 1.0 (позитив).\n"
        "2. 'learning_feedback': краткая выжимка (1-2 предложения) из текста об оценке учебного процесса.\n"
        "3. 'teamwork_feedback': краткая выжимка (1-2 предложения) об оценке работы в команде.\n"
        "4. 'organization_feedback': краткая выжимка (1-2 предложения) об оценке организационных и досуговых моментов.\n"
        "5. 'learning_sentiment_score': тональность ТОЛЬКО части про учёбу (от -1.0 до 1.0).\n"
        "6. 'teamwork_sentiment_score': тональность ТОЛЬКО части про команду (от -1.0 до 1.0).\n"
        "7. 'organization_sentiment_score': тональность ТОЛЬКО части про организацию (от -1.0 до 1.0).\n\n"
        f"Текст для анализа: \"{text}\""
    )
    try:
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        for key, val in error_result.items():
            if key not in result:
                result[key] = val
        return result
    except Exception as e:
        print(f"Error processing text: '{text[:50]}...'. Error: {e}")
        st.error(f"Ошибка при вызове DeepSeek API: {e}")
        return error_result

# ----------------------
# Конвертация тональности
# ----------------------
def convert_sentiment_to_10_point(score: float) -> float:
    if not isinstance(score, (int, float)):
        return 5.5
    return (score + 1) * 4.5 + 1

# ----------------------
# Новые функции генерации номинаций и рефлексий
# ----------------------
@st.cache_data
async def generate_nominations(df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    grouped = df.groupby('username')['text'].apply(lambda texts: ' '.join(texts)).reset_index()
    async def gen(item):
        username, text = item
        prompt = (
            f"Ты — ИИ-ассистент, анализирующий рефлексии школьников с морской научно-технической проектной смены. "
            f"На основе рефлексий: \"{text}\", присвой шуточную номинацию в морской тематике и дай краткое обоснование (1-2 предложения), "
            f"почему она подходит. Верни JSON-объект: {{'nomination': str, 'justification': str}}."
        )
        try:
            resp = await client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            return {'username': username,
                    'nomination': data.get('nomination', 'Морской Исследователь'),
                    'justification': data.get('justification', 'За активное участие в проекте!')}
        except:
            return {'username': username,
                    'nomination': 'Морской Исследователь',
                    'justification': 'За активное участие в проекте!'}
    items = list(zip(grouped['username'], grouped['text']))
    results = await asyncio.gather(*[gen(item) for item in items])
    return pd.DataFrame(results)

@st.cache_data
async def generate_friendly_reflections(df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    grouped = df.groupby('username')['text'].apply(lambda texts: ' '.join(texts)).reset_index()
    async def gen(item):
        username, text = item
        prompt = (
            f"Ты — ИИ-ассистент, суммирующий рефлексии школьников с морской научно-технической проектной смены. "
            f"На основе рефлексий: \"{text}\", создай дружелюбное, шуточное резюме (2-3 предложения) и позитивное напутствие (1-2 предложения). "
            f"Верни JSON-объект: {{'reflection': str, 'encouragement': str}}."
        )
        try:
            resp = await client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                'username': username,
                'reflection': data.get('reflection', 'Ты отлично справляешься с проектами!'),
                'encouragement': data.get('encouragement', 'Продолжай в том же духе!')
            }
        except:
            return {
                'username': username,
                'reflection': 'Ты отлично справляешься с проектами!',
                'encouragement': 'Продолжай в том же духе!'
            }
    items = list(zip(grouped['username'], grouped['text']))
    results = await asyncio.gather(*[gen(item) for item in items])
    return pd.DataFrame(results)

# ----------------------
# Основная функция
# ----------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    # ... предыдущая часть main без изменений до фильтров ...
    # После определения filtered_df и перед отображением основных блоков:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎉 Дополнительные функции")
    # Кнопка для номинаций
    if st.sidebar.button("Сгенерировать шуточные номинации"):
        if 'nominations_df' not in st.session_state:
            with st.spinner("Генерация номинаций..."):
                nomin_df = asyncio.run(generate_nominations(filtered_df, client))
                st.session_state['nominations_df'] = nomin_df
        st.subheader("🏆 Шуточные номинации участников")
        st.dataframe(st.session_state['nominations_df'])

    # Кнопка для дружелюбных рефлексий
    if st.sidebar.button("Сгенерировать дружелюбные рефлексии"):
        if 'friendly_reflections_df' not in st.session_state:
            with st.spinner("Генерация дружелюбных рефлексий..."):
                refl_df = asyncio.run(generate_friendly_reflections(filtered_df, client))
                st.session_state['friendly_reflections_df'] = refl_df
        st.subheader("🌟 Дружелюбные рефлексии участников")
        st.dataframe(st.session_state['friendly_reflections_df'])

    # ... остальной код main без изменений ...

if __name__ == "__main__":
    main()
