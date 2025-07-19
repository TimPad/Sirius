# 1. Установка зависимостей (без изменений)
# ----------------------
import subprocess
import sys


# ----------------------
# 2. Импорты и константы
# ----------------------
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

### НОВОЕ: Импорты для работы с Supabase
from supabase import create_client, Client

# Константы для API DeepSeek
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"

# ----------------------
# 3. Функция загрузки данных (без изменений)
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    # Приводим названия колонок к нижнему регистру и убираем пробелы
    df.columns = [str(col).strip().lower() if isinstance(col, str) else col for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

# ----------------------
# 4. БЛОК ПРЕДОБРАБОТКИ ТЕКСТА (NLP) - УДАЛЕН
# ----------------------

# ----------------------
# 5. DeepSeek API анализ (без изменений)
# ----------------------
async def analyze_reflection_with_deepseek(client: AsyncOpenAI, text: str) -> dict:
    """
    Асинхронно анализирует текст рефлексии с помощью DeepSeek API.
    """
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
        "5. 'learning_sentiment_score': тональность ТОЛЬКО части про учёбу (от -1.0 до 1.0). Если не упоминается, верни 0.0.\n"
        "6. 'teamwork_sentiment_score': тональность ТОЛЬКО части про команду (от -1.0 до 1.0). Если не упоминается, верни 0.0.\n"
        "7. 'organization_sentiment_score': тональность ТОЛЬКО части про организацию (от -1.0 до 1.0). Если не упоминается, верни 0.0.\n\n"
        "Если какой-то аспект в тексте не упоминается, для ключей feedback оставь пустую строку, а для ключей sentiment_score верни 0.0.\n\n"
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
        for key, value in error_result.items():
            if key not in result:
                result[key] = value
        return result

    except Exception as e:
        print(f"Error processing text: '{text[:50]}...'. Error: {e}")
        st.error(f"Ошибка при вызове DeepSeek API: {e}")
        return error_result

# ----------------------
# НОВЫЕ ФУНКЦИИ ГЕНЕРАЦИИ (без изменений)
# ----------------------
async def _get_one_nomination(client: AsyncOpenAI, username: str, text: str) -> dict:
    prompt = (
        "Ты — ИИ-ассистент, анализирующий рефлексии школьников с морской научно-технической проектной смены. "
        f"На основе рефлексий участника по имени {username}: \"{text}\", присвой шуточную номинацию в морской тематике "
        "(например, 'Капитан Гениальности', 'Инженер Глубин') и дай краткое обоснование (1-2 предложения), "
        "почему она подходит, основываясь на содержании рефлексий. Номинация и обоснование должны быть позитивными, "
        "подходящими для школьников и связанными с морской/научно-технической тематикой. "
        "Верни JSON-объект: {\"nomination\": str, \"justification\": str}."
    )
    default_result = {"nomination": "Морской Исследователь", "justification": "За активное участие в проекте!"}
    try:
        response = await client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}], temperature=0.7, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        return result if 'nomination' in result and 'justification' in result else default_result
    except Exception as e:
        print(f"Error generating nomination for {username}: {e}")
        return default_result

async def _generate_nominations_async(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    user_reflections = _df.groupby('username')['text'].apply(lambda texts: ' '.join(texts.astype(str).str.strip())).reset_index()
    tasks = [_get_one_nomination(client, row['username'], row['text']) for _, row in user_reflections.iterrows()]
    results = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(results)
    final_df = pd.concat([user_reflections[['username']], results_df], axis=1).rename(columns={'username': 'ФИО', 'nomination': 'Номинация', 'justification': 'Обоснование'})
    return final_df

@st.cache_data(show_spinner=False, hash_funcs={AsyncOpenAI: lambda _: None})
def get_cached_nominations(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    return asyncio.run(_generate_nominations_async(_df, client))

async def _get_one_friendly_reflection(client: AsyncOpenAI, username: str, text: str) -> dict:
    prompt = (
        "Ты — ИИ-ассистент, суммирующий рефлексии школьников с морской научно-технической проектной смены. "
        f"На основе рефлексий участника по имени {username}: \"{text}\", создай дружелюбное, шуточное резюме его рефлексий (2-3 предложения) "
        "и позитивное напутствие (1-2 предложения) для мотивации. Тон должен быть позитивным, не обидным, "
        "подходящим для школьников, с учетом морской/научно-технической тематики (например, используй слова 'курс', 'плавание', 'горизонты'). "
        "Верни JSON-объект: {\"reflection\": str, \"encouragement\": str}."
    )
    default_result = {"reflection": "Ты отлично справляешься с проектами!", "encouragement": "Продолжай в том же духе и покоряй новые горизонты!"}
    try:
        response = await client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}], temperature=0.7, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        return result if 'reflection' in result and 'encouragement' in result else default_result
    except Exception as e:
        print(f"Error generating friendly reflection for {username}: {e}")
        return default_result

async def _generate_friendly_reflections_async(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    user_reflections = _df.groupby('username')['text'].apply(lambda texts: ' '.join(texts.astype(str).str.strip())).reset_index()
    tasks = [_get_one_friendly_reflection(client, row['username'], row['text']) for _, row in user_reflections.iterrows()]
    results = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(results)
    final_df = pd.concat([user_reflections[['username']], results_df], axis=1).rename(columns={'username': 'ФИО', 'reflection': 'Рефлексия', 'encouragement': 'Пожелание'})
    return final_df

@st.cache_data(show_spinner=False, hash_funcs={AsyncOpenAI: lambda _: None})
def get_cached_friendly_reflections(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    return asyncio.run(_generate_friendly_reflections_async(_df, client))

# ----------------------
# 6. Конвертация тональности (без изменений)
# ----------------------
def convert_sentiment_to_10_point(score: float) -> float:
    if not isinstance(score, (int, float)):
        return 5.5
    return (score + 1) * 4.5 + 1

# ----------------------
# 7. Функции для работы с Supabase (без изменений)
# ----------------------
@st.cache_resource
def init_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except KeyError as e:
        st.error(f"Ошибка: ключ '{e.args[0]}' не найден в секретах Streamlit. Пожалуйста, добавьте его в настройки.")
        return None
    except Exception as e:
        st.error(f"Ошибка подключения к Supabase: {e}")
        return None

@st.cache_data(ttl=300)
def get_report_list_from_supabase(_supabase: Client) -> list:
    try:
        response = _supabase.table('reports').select('report_name', count='exact').execute()
        return sorted(list(set(item['report_name'] for item in response.data)), reverse=True) if response.data else []
    except Exception as e:
        st.error(f"Ошибка при получении списка отчетов из Supabase: {e}")
        return []

@st.cache_data
def load_report_from_supabase(_supabase: Client, report_name: str) -> pd.DataFrame:
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

# ----------------------
# 8. Основная логика и дашборд на Streamlit (ИЗМЕНЕНО)
# ----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    with st.expander("ℹ️ О проекте: что это и как пользоваться?", expanded=False):
        st.markdown("""
        ... (текст без изменений) ...
        """)

    supabase = init_supabase_client()
    if not supabase:
        st.stop()

    st.sidebar.header("🗂️ Источник данных")
    report_files = get_report_list_from_supabase(supabase)
    data_source_options = ["Новый анализ"] + report_files
    selected_source = st.sidebar.selectbox("Выберите отчет из архива или начните новый анализ:", data_source_options)

    # --- Инициализация флагов отображения в session_state при смене источника данных ---
    # Это нужно, чтобы при загрузке нового файла старые сгенерированные таблицы не отображались
    current_file_key = f"current_file_{selected_source}"
    if 'last_file_key' not in st.session_state or st.session_state.last_file_key != current_file_key:
        st.session_state.show_nominations = False
        st.session_state.show_reflections = False
        st.session_state.last_file_key = current_file_key
        
    df = None
    uploaded_file = None
    if selected_source != "Новый анализ":
        st.sidebar.success(f"Загружен отчет из архива: {selected_source}")
        df = load_report_from_supabase(supabase, selected_source)
        st.session_state['current_file_name'] = selected_source
    else:
        st.sidebar.header("📄 Загрузка для нового анализа")
        uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл с рефлексиями", type="xlsx")
        if uploaded_file:
            st.session_state['current_file_name'] = uploaded_file.name
            current_file_key = f"current_file_{uploaded_file.name}"
            if 'last_file_key' not in st.session_state or st.session_state.last_file_key != current_file_key:
                st.session_state.show_nominations = False
                st.session_state.show_reflections = False
                st.session_state.last_file_key = current_file_key
            df = load_data(uploaded_file)
            df['text'] = df['text'].astype(str).fillna('')

    if df is None:
        st.info("Пожалуйста, загрузите файл для нового анализа или выберите готовый отчет в боковой панели.")
        return

    client = None
    if selected_source == "Новый анализ":
        try:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            client = AsyncOpenAI(base_url=DEESEEK_API_URL, api_key=api_key)
        except KeyError:
            st.sidebar.error("API-ключ DeepSeek не найден.")
            st.stop()

    session_key = f"df_processed_{st.session_state.get('current_file_name', 'default')}"
    if session_key not in st.session_state:
        # ... (логика обработки и сохранения df в session_state без изменений) ...
        if selected_source == "Новый анализ" and client:
            with st.spinner('Выполняется анализ рефлексий... Это займет некоторое время.'):
                tasks = [analyze_reflection_with_deepseek(client, text) for text in df['text']]
                async def gather_tasks(): return await asyncio.gather(*tasks)
                results = asyncio.run(gather_tasks())
                results_df = pd.DataFrame(results)
                df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        
        for col in ['sentiment_score', 'learning_sentiment_score', 'teamwork_sentiment_score', 'organization_sentiment_score']:
            if col in df.columns:
                 df[col.replace('_score', '_10_point')] = df[col].apply(convert_sentiment_to_10_point)
        st.session_state[session_key] = df
    else:
        df = st.session_state[session_key]

    if selected_source == "Новый анализ" and uploaded_file:
        # ... (логика сохранения в архив без изменений) ...
        st.sidebar.header("💾 Сохранение")
        if st.sidebar.button("Сохранить в архив"):
            # ...
            pass # Код сохранения

    if df.empty:
        st.error("Нет данных для отображения.")
        return
        
    filtered_df = df.copy()

    st.sidebar.header("📊 Фильтры")
    # ... (логика фильтрации без изменений) ...
    if 'data' in filtered_df.columns and not filtered_df['data'].dropna().empty:
        #...
        pass
    
    if filtered_df.empty:
        st.error("Нет данных для отображения по выбранным фильтрам.")
        return

    # --- ИЗМЕНЕННЫЙ БЛОК: Управление отображением через флаги ---
    st.sidebar.header("🎉 Дополнительные функции")
    if st.sidebar.button("Сгенерировать шуточные номинации"):
        st.session_state.show_nominations = True

    if st.sidebar.button("Сгенерировать дружелюбные рефлексии"):
        st.session_state.show_reflections = True
        
    # Показываем кнопку "Скрыть", только если есть что скрывать
    if st.session_state.get('show_nominations', False) or st.session_state.get('show_reflections', False):
        if st.sidebar.button("Скрыть дополнительные таблицы"):
            st.session_state.show_nominations = False
            st.session_state.show_reflections = False


    # --- 1. ВСЕГДА ОТОБРАЖАЕМ ОСНОВНОЙ ДАШБОРД ---
    st.header("Общая динамика и групповой анализ")
    # ... (весь код для графиков и тепловой карты дашборда) ...
    daily_groups = filtered_df.groupby(filtered_df['data'].dt.date)
    # ... и так далее
    st.header("Анализ по отдельным учащимся")
    # ... (весь код для анализа по учащимся, радара и таблицы) ...
    st.header("Анализ \"Зоны риска\": участники с повторяющимся негативом")
    # ... (весь код для анализа зон риска) ...


    # --- 2. УСЛОВНО ОТОБРАЖАЕМ ДОПОЛНИТЕЛЬНЫЕ ТАБЛИЦЫ ---
    # Блок для номинаций
    if st.session_state.get('show_nominations', False):
        st.header("🏆 Шуточные номинации участников")
        nominations_key = f"nominations_{session_key}"
        
        if client is None:
            st.error("Генерация недоступна при просмотре отчета из архива. Выберите 'Новый анализ' в боковой панели.")
        else:
            if nominations_key not in st.session_state:
                with st.spinner("Создаем номинации... Это может занять несколько минут..."):
                    nominations_df = get_cached_nominations(filtered_df, client)
                    st.session_state[nominations_key] = nominations_df
            
            st.dataframe(st.session_state[nominations_key], use_container_width=True)

    # Блок для рефлексий
    if st.session_state.get('show_reflections', False):
        st.header("🌟 Дружелюбные рефлексии и напутствия")
        reflections_key = f"friendly_reflections_{session_key}"

        if client is None:
            st.error("Генерация недоступна при просмотре отчета из архива. Выберите 'Новый анализ' в боковой панели.")
        else:
            if reflections_key not in st.session_state:
                with st.spinner("Пишем дружеские послания... Это может занять несколько минут..."):
                    reflections_df = get_cached_friendly_reflections(filtered_df, client)
                    st.session_state[reflections_key] = reflections_df

            df_to_display = st.session_state[reflections_key].copy()
            df_to_display['Рефлексия и напутствие'] = df_to_display['Рефлексия'] + '\n\n**Пожелание:** ' + df_to_display['Пожелание']
            st.dataframe(df_to_display[['ФИО', 'Рефлексия и напутствие']], use_container_width=True)


if __name__ == "__main__":
    main()
