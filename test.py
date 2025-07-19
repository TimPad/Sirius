# 1. Установка зависимостей
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
from supabase import create_client, Client

# Константы для API DeepSeek
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"

# ----------------------
# 3. Функция загрузки данных
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(col).strip().lower() if isinstance(col, str) else col for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

# ----------------------
# 4. DeepSeek API анализ
# ----------------------
async def analyze_reflection_with_deepseek(client: AsyncOpenAI, text: str) -> dict:
    error_result = {
        "sentiment_score": 0.0, "learning_feedback": "N/A", "teamwork_feedback": "N/A",
        "organization_feedback": "N/A", "learning_sentiment_score": 0.0,
        "teamwork_sentiment_score": 0.0, "organization_sentiment_score": 0.0,
    }
    if not text or not isinstance(text, str) or not text.strip():
        return error_result
    prompt = (
        "Ты — ИИ-ассистент для анализа текстов рефлексии. Проанализируй рефлексию школьника. "
        "Твоя задача — вернуть JSON-объект со следующими ключами:\n"
        "1. 'sentiment_score': общая тональность текста, число от -1.0 до 1.0.\n"
        "2. 'learning_feedback': краткая выжимка (1-2 предложения) об оценке учебного процесса.\n"
        "3. 'teamwork_feedback': краткая выжимка (1-2 предложения) о работе в команде.\n"
        "4. 'organization_feedback': краткая выжимка (1-2 предложения) об организации и досуге.\n"
        "5. 'learning_sentiment_score': тональность ТОЛЬКО части про учёбу (от -1.0 до 1.0). Если не упоминается, верни 0.0.\n"
        "6. 'teamwork_sentiment_score': тональность ТОЛЬКО части про команду (от -1.0 до 1.0). Если не упоминается, верни 0.0.\n"
        "7. 'organization_sentiment_score': тональность ТОЛЬКО части про организацию (от -1.0 до 1.0). Если не упоминается, верни 0.0.\n\n"
        f"Текст для анализа: \"{text}\""
    )
    try:
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}],
            temperature=0.2, response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        for key, value in error_result.items():
            if key not in result: result[key] = value
        return result
    except Exception as e:
        print(f"Error processing text: '{text[:50]}...'. Error: {e}")
        return error_result

# ----------------------
# 5. Новые функции генерации
# ----------------------
async def _get_one_nomination(client: AsyncOpenAI, username: str, text: str) -> dict:
    prompt = (
        "Ты — ИИ-ассистент, анализирующий рефлексии школьников с морской научно-технической проектной смены. "
        f"На основе рефлексий участника по имени {username}: \"{text}\", присвой шуточную номинацию в морской тематике "
        "(например, 'Капитан Гениальности', 'Инженер Глубин') и дай краткое обоснование (1-2 предложения). "
        "Номинация и обоснование должны быть позитивными и подходящими для школьников. "
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
    return pd.concat([user_reflections[['username']], results_df], axis=1).rename(columns={'username': 'ФИО', 'nomination': 'Номинация', 'justification': 'Обоснование'})

@st.cache_data(show_spinner=False, hash_funcs={AsyncOpenAI: lambda _: None})
def get_cached_nominations(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    return asyncio.run(_generate_nominations_async(_df, client))

async def _get_one_friendly_reflection(client: AsyncOpenAI, username: str, text: str) -> dict:
    prompt = (
        "Ты — ИИ-ассистент, суммирующий рефлексии школьников с морской научно-технической проектной смены. "
        f"На основе рефлексий участника по имени {username}: \"{text}\", создай дружелюбное, шуточное резюме (2-3 предложения) "
        "и позитивное напутствие (1-2 предложения). Тон должен быть позитивным, не обидным, с учетом морской тематики. "
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
    return pd.concat([user_reflections[['username']], results_df], axis=1).rename(columns={'username': 'ФИО', 'reflection': 'Рефлексия', 'encouragement': 'Пожелание'})

@st.cache_data(show_spinner=False, hash_funcs={AsyncOpenAI: lambda _: None})
def get_cached_friendly_reflections(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    return asyncio.run(_generate_friendly_reflections_async(_df, client))

# ----------------------
# 6. Вспомогательные функции
# ----------------------
def convert_sentiment_to_10_point(score: float) -> float:
    return (score + 1) * 4.5 + 1 if isinstance(score, (int, float)) else 5.5

@st.cache_resource
def init_supabase_client():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
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
        return df.drop(columns=['id', 'created_at', 'report_name'], errors='ignore') if not df.empty else df
    except Exception as e:
        st.error(f"Ошибка при загрузке отчета '{report_name}': {e}")
        return pd.DataFrame()

# ----------------------
# 7. Основная логика и дашборд
# ----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    with st.expander("ℹ️ О проекте: что это и как пользоваться?", expanded=False):
        st.markdown("""...""") # Скрыл для краткости

    supabase = init_supabase_client()
    if not supabase: st.stop()

    st.sidebar.header("🗂️ Источник данных")
    report_files = get_report_list_from_supabase(supabase)
    data_source_options = ["Новый анализ"] + report_files
    selected_source = st.sidebar.selectbox("Выберите отчет:", data_source_options)

    df = None
    uploaded_file = None
    
    if selected_source == "Новый анализ":
        uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл:", type="xlsx")
        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state['current_file_name'] = uploaded_file.name
    else:
        df = load_report_from_supabase(supabase, selected_source)
        st.session_state['current_file_name'] = selected_source

    # Сброс флагов при смене источника данных
    file_key = st.session_state.get('current_file_name')
    if 'last_file_key' not in st.session_state or st.session_state.last_file_key != file_key:
        st.session_state.show_nominations = False
        st.session_state.show_reflections = False
        st.session_state.last_file_key = file_key

    if df is None:
        st.info("Пожалуйста, загрузите файл или выберите отчет.")
        return

    client = AsyncOpenAI(base_url=DEESEEK_API_URL, api_key=st.secrets["DEEPSEEK_API_KEY"]) if selected_source == "Новый анализ" else None

    session_key = f"df_processed_{st.session_state.get('current_file_name', 'default')}"
    if session_key not in st.session_state:
        if selected_source == "Новый анализ" and client:
            with st.spinner('Выполняется анализ рефлексий...'):
                async def gather_tasks(): return await asyncio.gather(*[analyze_reflection_with_deepseek(client, text) for text in df['text']])
                results = asyncio.run(gather_tasks())
                df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)], axis=1)
        
        for col in ['sentiment_score', 'learning_sentiment_score', 'teamwork_sentiment_score', 'organization_sentiment_score']:
            if col in df.columns:
                 df[col.replace('_score', '_10_point')] = df[col].apply(convert_sentiment_to_10_point)
        st.session_state[session_key] = df
    else:
        df = st.session_state[session_key]
    
    # --- ВОССТАНОВЛЕННЫЙ БЛОК СОХРАНЕНИЯ ---
    if selected_source == "Новый анализ" and uploaded_file:
        st.sidebar.header("💾 Сохранение")
        if st.sidebar.button("Сохранить в архив"):
            with st.spinner("Сохранение отчета в облако..."):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                base_filename = os.path.splitext(uploaded_file.name)[0]
                report_filename = f"{base_filename}_processed_{timestamp}"

                # Сохраняем только основной анализ, без номинаций
                df_to_save = st.session_state[session_key].copy()
                df_to_save['report_name'] = report_filename
                
                if 'data' in df_to_save.columns:
                    df_to_save['data'] = pd.to_datetime(df_to_save['data']).dt.strftime('%Y-%m-%dT%H:%M:%S')
                
                data_to_upload = df_to_save.replace({pd.NaT: None, np.nan: None}).to_dict(orient='records')
                
                try:
                    supabase.table('reports').upsert(data_to_upload, on_conflict='username,data').execute()
                    st.sidebar.success(f"Анализ сохранен как:\n**{report_filename}**")
                    st.cache_data.clear() # Очищаем кэш для обновления списка файлов
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Ошибка сохранения в Supabase: {e}")

    if df.empty:
        st.warning("Нет данных для отображения.")
        return
        
    filtered_df = df.copy()

    st.sidebar.header("📊 Фильтры")
    if 'data' in filtered_df.columns and not filtered_df['data'].dropna().empty:
        min_date, max_date = filtered_df['data'].min().date(), filtered_df['data'].max().date()
        if min_date != max_date:
            start_date, end_date = st.sidebar.slider("Диапазон дат:", min_date, max_date, (min_date, max_date))
            filtered_df = filtered_df.loc[(filtered_df['data'].dt.date >= start_date) & (filtered_df['data'].dt.date <= end_date)]
    if filtered_df.empty:
        st.error("Нет данных по выбранным фильтрам.")
        return

    st.sidebar.header("🎉 Дополнительные функции")
    if client:
        if st.sidebar.button("Сгенерировать шуточные номинации"): st.session_state.show_nominations = True
        if st.sidebar.button("Сгенерировать дружелюбные рефлексии"): st.session_state.show_reflections = True
    
    if st.session_state.get('show_nominations') or st.session_state.get('show_reflections'):
        if st.sidebar.button("Скрыть доп. таблицы", type="primary"):
            st.session_state.show_nominations = False
            st.session_state.show_reflections = False
            st.rerun()

    # --- 1. ВСЕГДА ОТОБРАЖАЕМ ОСНОВНОЙ ДАШБОРД ---
    st.header("Общая динамика и групповой анализ")
    # ... (код отображения графиков и таблиц)
    
    st.header("Анализ по отдельным учащимся")
    # ... (код отображения графиков и таблиц)
    
    st.header("Анализ \"Зоны риска\"")
    # ... (код отображения графиков и таблиц)


    # --- 2. УСЛОВНО ОТОБРАЖАЕМ ДОПОЛНИТЕЛЬНЫЕ ТАБЛИЦЫ ---
    if st.session_state.get('show_nominations'):
        st.header("🏆 Шуточные номинации участников")
        nominations_key = f"nominations_{session_key}"
        if nominations_key not in st.session_state:
            with st.spinner("Создаем номинации..."):
                st.session_state[nominations_key] = get_cached_nominations(filtered_df, client)
        st.dataframe(st.session_state[nominations_key], use_container_width=True)

    if st.session_state.get('show_reflections'):
        st.header("🌟 Дружелюбные рефлексии и напутствия")
        reflections_key = f"reflections_{session_key}"
        if reflections_key not in st.session_state:
            with st.spinner("Пишем дружеские послания..."):
                st.session_state[reflections_key] = get_cached_friendly_reflections(filtered_df, client)
        df_to_display = st.session_state[reflections_key].copy()
        df_to_display['Рефлексия и напутствие'] = df_to_display['Рефлексия'] + '\n\n**Пожелание:** ' + df_to_display['Пожелание']
        st.dataframe(df_to_display[['ФИО', 'Рефлексия и напутствие']], use_container_width=True)

if __name__ == "__main__":
    main()
