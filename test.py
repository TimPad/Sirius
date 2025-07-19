# 1. Установка зависимостей
# Убедитесь, что все зависимости установлены. Выполните в терминале:
# pip install streamlit pandas plotly supabase openai openpyxl
# ----------------------
import subprocess
import sys
import io # для работы с файлами в памяти
import os
import json
import asyncio
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from openai import AsyncOpenAI
from supabase import create_client, Client


# ----------------------
# 2. Константы
# ----------------------
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"


# ----------------------
# 3. Вспомогательные функции
# ----------------------

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Загружает и предварительно обрабатывает данные из Excel-файла."""
    df = pd.read_excel(file_path)
    df.columns = [str(col).strip().lower() if isinstance(col, str) else col for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

def to_excel(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в байты Excel-файла для скачивания."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Характеристики')
    return output.getvalue()

def run_async(coro):
    """Безопасно запускает асинхронную корутину в окружении Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def convert_sentiment_to_10_point(score: float) -> float:
    """Конвертирует тональность из диапазона [-1, 1] в [1, 10]."""
    if not isinstance(score, (int, float)):
        return 5.5
    return (score + 1) * 4.5 + 1


# ----------------------
# 4. Функции для работы с API (Supabase и LLM)
# ----------------------

@st.cache_resource
def init_supabase_client():
    """Инициализирует и возвращает клиент Supabase."""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError as e:
        st.error(f"Ошибка: ключ '{e.args[0]}' не найден в секретах Streamlit.")
        return None
    except Exception as e:
        st.error(f"Ошибка подключения к Supabase: {e}")
        return None

@st.cache_data(ttl=300)
def get_report_list_from_supabase(_supabase: Client) -> list:
    """Получает список уникальных имен отчетов из Supabase."""
    try:
        response = _supabase.table('reports').select('report_name', count='exact').execute()
        return sorted(list(set(item['report_name'] for item in response.data)), reverse=True) if response.data else []
    except Exception as e:
        st.error(f"Ошибка при получении списка отчетов из Supabase: {e}")
        return []

@st.cache_data
def load_report_from_supabase(_supabase: Client, report_name: str) -> pd.DataFrame:
    """Загружает все данные для указанного отчета из Supabase в DataFrame."""
    try:
        response = _supabase.table('reports').select('*').eq('report_name', report_name).execute()
        df = pd.DataFrame(response.data)
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
        if not df.empty:
            df = df.drop(columns=['id', 'created_at', 'report_name'], errors='ignore')
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке отчета '{report_name}': {e}")
        return pd.DataFrame()

async def analyze_reflection_with_deepseek(client: AsyncOpenAI, text: str) -> dict:
    """Асинхронно анализирует текст рефлексии для получения структурированных данных."""
    # ... (код этой функции без изменений)
    error_result = {"sentiment_score": 0.0, "learning_feedback": "N/A", "teamwork_feedback": "N/A", "organization_feedback": "N/A", "learning_sentiment_score": 0.0, "teamwork_sentiment_score": 0.0, "organization_sentiment_score": 0.0}
    if not text or not isinstance(text, str) or not text.strip(): return error_result
    prompt = ("Ты — ИИ-ассистент... (полный текст вашего промпта)") # Оставил сокращенно для читаемости
    try:
        response = await client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}], temperature=0.2, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        for key, value in error_result.items(): result.setdefault(key, value)
        return result
    except Exception as e:
        print(f"Error processing text: '{text[:50]}...'. Error: {e}")
        return error_result

async def generate_nomination_with_llm(client: AsyncOpenAI, reflections: str) -> str:
    """Генерирует шуточную номинацию на основе рефлексий участника."""
    if not reflections or not reflections.strip():
        reflections = "Участник был слишком погружен в великие научные дела и не оставил рефлексий."
    prompt = ( "Ты — креативный копирайтер для детского научного лагеря... (полный текст вашего промпта)" ) # Оставил сокращенно
    try:
        response = await client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=50)
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"Error generating nomination: {e}")
        return "Мастер Неопределенности"

async def generate_character_description_with_llm(client: AsyncOpenAI, name: str, reflections: str) -> str:
    """Генерирует шуточную, но добрую характеристику на участника."""
    is_empty_reflection = not reflections or not reflections.strip()
    if is_empty_reflection:
        reflections = "Рефлексии отсутствуют..." # Полный текст вашего "пустого" случая
    prompt = ( f"Ты — добрый и мудрый наставник... (полный текст вашего промпта с именем {name})" ) # Оставил сокращенно
    try:
        response = await client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=400)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating description: {e}")
        return f"К сожалению, во время генерации характеристики для {name} произошла космическая аномалия. Но мы точно знаем, что {name} — замечательный человек, и желаем ему огромных успехов во всех начинаниях!"


# ----------------------
# 5. Основная логика и дашборд на Streamlit
# ----------------------
def main():
    st.set_page_config(layout="wide", page_title="Анализ рефлексий")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    with st.expander("ℹ️ О проекте: что это и как пользоваться?", expanded=False):
        st.markdown(""" ... (ваш текст о проекте) ... """)

    # --- Инициализация клиентов ---
    supabase = init_supabase_client()
    if not supabase:
        st.stop()
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        client = AsyncOpenAI(base_url=DEESEEK_API_URL, api_key=api_key)
    except KeyError:
        st.error("Ошибка конфигурации: отсутствует ключ `DEEPSEEK_API_KEY` в секретах Streamlit.")
        st.stop()

    # --- Загрузка данных ---
    st.sidebar.header("🗂️ Источник данных")
    report_files = get_report_list_from_supabase(supabase)
    data_source_options = ["Новый анализ"] + report_files
    selected_source = st.sidebar.selectbox("Выберите отчет или начните новый анализ:", data_source_options)

    df = None
    if selected_source != "Новый анализ":
        df = load_report_from_supabase(supabase, selected_source)
        st.session_state['current_file_name'] = selected_source
    else:
        uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл:", type="xlsx")
        if uploaded_file:
            st.session_state['current_file_name'] = uploaded_file.name
            df = load_data(uploaded_file)
            df['text'] = df['text'].astype(str).fillna('')

    if df is None:
        st.info("Пожалуйста, загрузите файл или выберите отчет в боковой панели.")
        st.stop()

    # --- Обработка и кэширование данных ---
    session_key = f"df_processed_{st.session_state.get('current_file_name', 'default')}"
    if session_key not in st.session_state:
        if selected_source == "Новый анализ":
            with st.spinner('Выполняется анализ рефлексий... Это займет некоторое время.'):
                tasks = [analyze_reflection_with_deepseek(client, text) for text in df['text']]
                results = run_async(asyncio.gather(*tasks))
                results_df = pd.DataFrame(results)
                df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

        for col in ['sentiment_score', 'learning_sentiment_score', 'teamwork_sentiment_score', 'organization_sentiment_score']:
            if col in df.columns:
                df[col.replace('_score', '_10_point')] = df[col].apply(convert_sentiment_to_10_point)
        st.session_state[session_key] = df
    
    df = st.session_state[session_key]

    # --- Фильтры ---
    st.sidebar.header("📊 Фильтры")
    filtered_df = df.copy()
    if 'data' in filtered_df.columns and not filtered_df['data'].dropna().empty:
        min_date, max_date = filtered_df['data'].min().date(), filtered_df['data'].max().date()
        if min_date != max_date:
            start_date, end_date = st.sidebar.slider("Диапазон дат:", min_date, max_date, (min_date, max_date))
            mask = (filtered_df['data'].dt.date >= start_date) & (filtered_df['data'].dt.date <= end_date)
            filtered_df = filtered_df.loc[mask]

    if filtered_df.empty:
        st.warning("Нет данных для отображения по выбранным фильтрам.")
        st.stop()

    # --- БЛОК МАССОВОЙ ГЕНЕРАЦИИ И СКАЧИВАНИЯ ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎓 Выпускные материалы")
    if st.sidebar.button("Сгенерировать характеристики для всех"):
        unique_students = filtered_df['username'].unique()
        
        async def generate_all_creative_content():
            tasks = []
            for student_name in unique_students:
                student_reflections = " ".join(filtered_df[filtered_df['username'] == student_name]['text'].dropna().astype(str))
                tasks.append(generate_nomination_with_llm(client, student_reflections))
                tasks.append(generate_character_description_with_llm(client, student_name, student_reflections))
            
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_data = []
            for i, student_name in enumerate(unique_students):
                nomination, description = all_results[i*2], all_results[i*2 + 1]
                final_data.append({
                    "Ученик": student_name,
                    "Номинация": "Ошибка" if isinstance(nomination, Exception) else nomination,
                    "Характеристика": f"Ошибка: {description}" if isinstance(description, Exception) else description,
                })
            return pd.DataFrame(final_data)

        with st.sidebar.spinner(f"Генерация контента для {len(unique_students)} учеников..."):
            creative_df = run_async(generate_all_creative_content())
            if not creative_df.empty:
                st.session_state['downloadable_excel'] = to_excel(creative_df)
                st.session_state['excel_filename'] = f"Характеристики_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                st.sidebar.success("Готово! Файл можно скачать.")
            else:
                st.sidebar.error("Не удалось сгенерировать данные.")

    if 'downloadable_excel' in st.session_state:
        st.sidebar.download_button(
            label="📥 Скачать файл Excel",
            data=st.session_state['downloadable_excel'],
            file_name=st.session_state['excel_filename'],
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    # --- ОСНОВНОЙ КОНТЕНТ ДАШБОРДА ---
    st.header("Общая динамика и групповой анализ")
    # ... (весь ваш код для отрисовки общих графиков и тепловой карты)

    st.header("Анализ по отдельным учащимся")
    student_list = sorted(filtered_df['username'].unique())
    if student_list:
        student = st.selectbox("Выберите ученика:", student_list)
        if student:
            student_df = filtered_df[filtered_df['username'] == student].sort_values('data')
            # ... (ваш код для отрисовки индивидуальных графиков ученика)

            # --- БЛОК ИНДИВИДУАЛЬНОЙ ГЕНЕРАЦИИ ---
            st.markdown("---")
            st.subheader(f"✨ Креативная характеристика для {student}")
            full_reflection_text = " ".join(student_df['text'].dropna().astype(str))
            if st.button(f"Сгенерировать для {student}"):
                with st.spinner("Магия творится..."):
                    async def get_content():
                        return await asyncio.gather(
                            generate_nomination_with_llm(client, full_reflection_text),
                            generate_character_description_with_llm(client, student, full_reflection_text)
                        )
                    nomination, description = run_async(get_content())
                    st.session_state[f'nomination_{student}'] = nomination
                    st.session_state[f'description_{student}'] = description

            if f'nomination_{student}' in st.session_state:
                st.success(f"**Номинация:** {st.session_state[f'nomination_{student}']}")
            if f'description_{student}' in st.session_state:
                st.markdown(st.session_state[f'description_{student}'])
            
            # --- Таблица с деталями ---
            st.markdown("---")
            st.subheader("Детальная таблица рефлексий")
            # ... (ваш код для expander и st.dataframe)

    # ... (ваш код для зоны риска и т.д.)


# ----------------------
# 6. Точка входа в приложение
# ----------------------
if __name__ == "__main__":
    main()
