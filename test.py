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
# НОВЫЕ ФУНКЦИИ ГЕНЕРАЦИИ
# ----------------------

async def _get_one_nomination(client: AsyncOpenAI, username: str, text: str) -> dict:
    """Вспомогательная асинхронная функция для генерации одной номинации."""
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
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        if 'nomination' not in result or 'justification' not in result:
             return default_result
        return result
    except Exception as e:
        print(f"Error generating nomination for {username}: {e}")
        return default_result

@st.cache_data(show_spinner=False)
async def generate_nominations(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    """Генерирует шуточные номинации для каждого участника на основе всех его рефлексий."""
    user_reflections = _df.groupby('username')['text'].apply(lambda texts: ' '.join(texts.astype(str).str.strip())).reset_index()
    
    tasks = [
        _get_one_nomination(client, row['username'], row['text']) 
        for index, row in user_reflections.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)
    
    results_df = pd.DataFrame(results)
    final_df = pd.concat([user_reflections[['username']], results_df], axis=1)
    final_df = final_df.rename(columns={'username': 'ФИО', 'nomination': 'Номинация', 'justification': 'Обоснование'})
    return final_df

async def _get_one_friendly_reflection(client: AsyncOpenAI, username: str, text: str) -> dict:
    """Вспомогательная асинхронная функция для генерации одной дружелюбной рефлексии."""
    prompt = (
        "Ты — ИИ-ассистент, суммирующий рефлексии школьников с морской научно-технической проектной смены. "
        f"На основе рефлексий участника по имени {username}: \"{text}\", создай дружелюбное, шуточное резюме его рефлексий (2-3 предложения) "
        "и позитивное напутствие (1-2 предложения) для мотивации. Тон должен быть позитивным, не обидным, "
        "подходящим для школьников, с учетом морской/научно-технической тематики (например, используй слова 'курс', 'плавание', 'горизонты'). "
        "Верни JSON-объект: {\"reflection\": str, \"encouragement\": str}."
    )
    default_result = {"reflection": "Ты отлично справляешься с проектами!", "encouragement": "Продолжай в том же духе и покоряй новые горизонты!"}
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        if 'reflection' not in result or 'encouragement' not in result:
             return default_result
        return result
    except Exception as e:
        print(f"Error generating friendly reflection for {username}: {e}")
        return default_result

@st.cache_data(show_spinner=False)
async def generate_friendly_reflections(_df: pd.DataFrame, client: AsyncOpenAI) -> pd.DataFrame:
    """Генерирует дружелюбные рефлексии и напутствия для каждого участника."""
    user_reflections = _df.groupby('username')['text'].apply(lambda texts: ' '.join(texts.astype(str).str.strip())).reset_index()

    tasks = [
        _get_one_friendly_reflection(client, row['username'], row['text'])
        for index, row in user_reflections.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)
    
    results_df = pd.DataFrame(results)
    final_df = pd.concat([user_reflections[['username']], results_df], axis=1)
    final_df = final_df.rename(columns={'username': 'ФИО', 'reflection': 'Рефлексия', 'encouragement': 'Пожелание'})
    return final_df


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
    """Инициализирует и возвращает клиент Supabase."""
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
    """Получает список уникальных имен отчетов из Supabase."""
    try:
        response = _supabase.table('reports').select('report_name', count='exact').execute()
        if response.data:
            unique_names = sorted(list(set(item['report_name'] for item in response.data)), reverse=True)
            return unique_names
        return []
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


# ----------------------
# 8. Основная логика и дашборд на Streamlit (ИЗМЕНЕНО)
# ----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    with st.expander("ℹ️ О проекте: что это и как пользоваться?", expanded=False):
        st.markdown("""
        **Цель дашборда** — помочь педагогам и кураторам быстро оценить эмоциональное состояние группы, выявить общие тенденции и определить учащихся, требующих особого внимания, на основе их письменных рефлексий.

        **Как это работает?**
        1.  Для **нового анализа** загрузите Excel-файл с текстами рефлексий.
        2.  Чтобы посмотреть **старый отчет**, выберите его из выпадающего списка. Данные загрузятся из облачного архива.
        3.  При новом анализе искусственный интеллект (DeepSeek) анализирует каждый текст.
        4.  После анализа вы можете **сохранить результат в архив**, нажав соответствующую кнопку. Отчет станет доступен для выбора при следующем запуске.
        5.  В **дополнительных функциях** можно сгенерировать шуточные номинации и персональные рефлексии для участников.
        """)

    supabase = init_supabase_client()
    if not supabase:
        st.stop()

    st.sidebar.header("🗂️ Источник данных")
    report_files = get_report_list_from_supabase(supabase)
    data_source_options = ["Новый анализ"] + report_files
    selected_source = st.sidebar.selectbox("Выберите отчет из архива или начните новый анализ:", data_source_options)

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
            st.sidebar.error("API-ключ DeepSeek не найден в настройках приложения.")
            st.error("Ошибка конфигурации: отсутствует ключ `DEEPSEEK_API_KEY`. "
                     "Пожалуйста, добавьте его в настройки секретов в Streamlit Cloud.")
            st.stop()

    session_key = f"df_processed_{st.session_state.get('current_file_name', 'default')}"

    if session_key not in st.session_state:
        if selected_source == "Новый анализ" and client:
            with st.spinner('Выполняется анализ рефлексий... Это займет некоторое время.'):
                tasks = [analyze_reflection_with_deepseek(client, text) for text in df['text']]
                async def gather_tasks():
                    return await asyncio.gather(*tasks)
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
        st.sidebar.header("💾 Сохранение")
        if st.sidebar.button("Сохранить в архив"):
            with st.spinner("Сохранение отчета в облако..."):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                base_filename = os.path.splitext(uploaded_file.name)[0]
                report_filename = f"{base_filename}_processed_{timestamp}"
                processed_df_to_save = st.session_state[session_key].copy()
                processed_df_to_save['report_name'] = report_filename
                if 'data' in processed_df_to_save.columns:
                    processed_df_to_save['data'] = pd.to_datetime(processed_df_to_save['data']).dt.strftime('%Y-%m-%dT%H:%M:%S')
                df_for_upload = processed_df_to_save.replace({pd.NaT: None, np.nan: None})
                df_for_upload = df_for_upload.drop(columns=['id'], errors='ignore')
                data_to_upload = df_for_upload.to_dict(orient='records')
                try:
                    supabase.table('reports').upsert(data_to_upload, on_conflict='username,data').execute()
                    st.sidebar.success(f"Анализ сохранен как:\n**{report_filename}**\nДубликаты проигнорированы.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Ошибка сохранения в Supabase: {e}")
    if df.empty:
        st.error("Нет данных для отображения. Пожалуйста, измените фильтры или загрузите другой файл.")
        return
        
    filtered_df = df.copy()

    st.sidebar.header("📊 Фильтры")
    if 'data' in filtered_df.columns and not filtered_df['data'].dropna().empty:
        min_date = filtered_df['data'].min().date()
        max_date = filtered_df['data'].max().date()
        if min_date != max_date:
            date_range = st.sidebar.slider(
                "Выберите диапазон дат:",
                min_value=min_date, max_value=max_date,
                value=(min_date, max_date), format="DD.MM.YYYY"
            )
            start_date, end_date = date_range
            mask = (filtered_df['data'].dt.date >= start_date) & (filtered_df['data'].dt.date <= end_date)
            filtered_df = filtered_df.loc[mask]
        else:
             st.sidebar.info("В данных только один день, фильтр по дате неактивен.")
    else:
        st.sidebar.warning("В файле отсутствуют корректные даты для фильтрации.")

    if filtered_df.empty:
        st.error("Нет данных для отображения по выбранным фильтрам.")
        return

    # --- НОВЫЙ БЛОК: Управление отображением ---
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'dashboard'

    # Кнопки для переключения вида в боковой панели
    st.sidebar.header("🎉 Дополнительные функции")
    if st.sidebar.button("Сгенерировать шуточные номинации"):
        st.session_state.view_mode = 'nominations'
        st.rerun()

    if st.sidebar.button("Сгенерировать дружелюбные рефлексии"):
        st.session_state.view_mode = 'reflections'
        st.rerun()

    # Кнопка возврата на главный дашборд
    if st.session_state.view_mode != 'dashboard':
        if st.sidebar.button("◀️ Вернуться к основному дашборду"):
            st.session_state.view_mode = 'dashboard'
            st.rerun()
            
    # --- Основной контент в зависимости от режима ---
    if st.session_state.view_mode == 'dashboard':
        # --- Начало оригинального блока отображения дашборда ---
        st.header("Общая динамика и групповой анализ")
        daily_groups = filtered_df.groupby(filtered_df['data'].dt.date)
        agg_dict = {
            'avg_emotion': ('emotion', 'mean'),
            'avg_sentiment_10_point': ('sentiment_10_point', 'mean'),
            'avg_learning_sentiment': ('learning_sentiment_10_point', 'mean'),
            'avg_teamwork_sentiment': ('teamwork_sentiment_10_point', 'mean'),
            'avg_organization_sentiment': ('organization_sentiment_10_point', 'mean')
        }
        # Исключаем агрегации для колонок, которых нет
        valid_agg_dict = {k: v for k, v in agg_dict.items() if v[0] in filtered_df.columns}
        if valid_agg_dict:
            daily_df = daily_groups.agg(**valid_agg_dict).reset_index()
            daily_df.rename(columns={'data': 'Дата'}, inplace=True)

            if not daily_df.empty:
                daily_df.sort_values('Дата', inplace=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Общая тональность vs. Самооценка")
                    fig = px.line(
                        daily_df, x='Дата', y=['avg_sentiment_10_point', 'avg_emotion'],
                        labels={'value': 'Оценка (1-10)', 'variable': 'Метрика'},
                        title='Сравнение тональности и самооценки'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("Детализация тональности по аспектам")
                    fig_details = px.line(
                        daily_df, x='Дата', y=['avg_learning_sentiment', 'avg_teamwork_sentiment', 'avg_organization_sentiment'],
                        labels={'value': 'Оценка (1-10)', 'variable': 'Аспект'},
                        title='Динамика тональности по аспектам'
                    )
                    new_names = {'avg_learning_sentiment': 'Учёба', 'avg_teamwork_sentiment': 'Команда', 'avg_organization_sentiment': 'Организация'}
                    fig_details.for_each_trace(lambda t: t.update(name = new_names.get(t.name, t.name)))
                    st.plotly_chart(fig_details, use_container_width=True)

        st.subheader("Тепловая карта тональности группы")
        if 'sentiment_10_point' in filtered_df.columns:
            heatmap_data = filtered_df.pivot_table(
                index='username',
                columns=filtered_df['data'].dt.date,
                values='sentiment_10_point',
                aggfunc='mean'
            )
            if not heatmap_data.empty:
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Дата", y="Ученик", color="Тональность"),
                    title="Общая тональность (1-10) по дням для каждого ученика",
                    color_continuous_scale='RdYlGn',
                    aspect="auto"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Недостаточно данных для построения тепловой карты.")
        else:
            st.info("Данные для тепловой карты (sentiment_10_point) отсутствуют.")

        st.header("Анализ по отдельным учащимся")
        student_list = sorted(filtered_df['username'].unique())
        if student_list:
            student = st.selectbox("Выберите ученика:", student_list)
            if student:
                student_df = filtered_df[filtered_df['username'] == student].sort_values('data')
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.subheader(f"Динамика оценок для {student}")
                    fig2 = px.line(
                        student_df, x='data', y=['sentiment_10_point', 'emotion'],
                        labels={'value': 'Оценка (1-10)', 'data': 'Дата'},
                        title=f'Тональность vs. Самооценка'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                with col2:
                    st.subheader(f"Профиль ученика")
                    categories = ['Самооценка', 'Учёба', 'Команда', 'Организация']
                    values = [
                        student_df['emotion'].mean(),
                        student_df['learning_sentiment_10_point'].mean(),
                        student_df['teamwork_sentiment_10_point'].mean(),
                        student_df['organization_sentiment_10_point'].mean()
                    ]
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Средняя оценка'))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[1, 10])),
                        showlegend=False,
                        title=f"Средние оценки для {student}",
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.subheader("Детальная таблица рефлексий")
                display_columns = [
                    'data', 'text', 'emotion', 'sentiment_10_point',
                    'learning_sentiment_10_point', 'teamwork_sentiment_10_point', 'organization_sentiment_10_point',
                    'learning_feedback', 'teamwork_feedback', 'organization_feedback'
                ]
                st.dataframe(student_df[[col for col in display_columns if col in student_df.columns]])

        if st.sidebar.checkbox("Показать полную таблицу с отфильтрованными результатами"):
            st.header("Полная таблица данных")
            st.dataframe(filtered_df)

        st.header("Анализ \"Зоны риска\": участники с повторяющимся негативом")
        if 'sentiment_score' in filtered_df.columns:
            negative_reflections = filtered_df[filtered_df['sentiment_score'] < 0]
            if not negative_reflections.empty:
                negative_counts = negative_reflections.groupby('username').size().reset_index(name='negative_count')
                at_risk_users = negative_counts[negative_counts['negative_count'] > 1].sort_values('negative_count', ascending=False)
                if not at_risk_users.empty:
                    st.warning("Внимание! Выявлены участники с многократной негативной тональностью в рефлексиях за выбранный период.")
                    st.dataframe(at_risk_users)
                else:
                    st.success("За выбранный период не выявлено участников, у которых тональность рефлексий была бы негативной более одного раза.")
            else:
                st.success("Негативных рефлексий за выбранный период не найдено.")
        # --- Конец оригинального блока отображения дашборда ---

    elif st.session_state.view_mode == 'nominations':
        st.header("🏆 Шуточные номинации участников")
        nominations_key = f"nominations_{session_key}"
        
        if client is None:
            st.error("Генерация недоступна при просмотре отчета из архива. Выберите 'Новый анализ' в боковой панели.")
        else:
            if nominations_key not in st.session_state:
                with st.spinner("Создаем номинации... Это может занять несколько минут..."):
                    nominations_df = asyncio.run(generate_nominations(filtered_df, client))
                    st.session_state[nominations_key] = nominations_df
            
            st.dataframe(st.session_state[nominations_key], use_container_width=True)

    elif st.session_state.view_mode == 'reflections':
        st.header("🌟 Дружелюбные рефлексии и напутствия")
        reflections_key = f"friendly_reflections_{session_key}"

        if client is None:
            st.error("Генерация недоступна при просмотре отчета из архива. Выберите 'Новый анализ' в боковой панели.")
        else:
            if reflections_key not in st.session_state:
                with st.spinner("Пишем дружеские послания... Это может занять несколько минут..."):
                    reflections_df = asyncio.run(generate_friendly_reflections(filtered_df, client))
                    st.session_state[reflections_key] = reflections_df

            df_to_display = st.session_state[reflections_key].copy()
            df_to_display['Рефлексия и напутствие'] = df_to_display['Рефлексия'] + '\n\n**Пожелание:** ' + df_to_display['Пожелание']
            st.dataframe(df_to_display[['ФИО', 'Рефлексия и напутствие']], use_container_width=True)

if __name__ == "__main__":
    main()
