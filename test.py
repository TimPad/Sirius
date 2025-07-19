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
async def _get_one_nomination(client: AsyncOpenAI, username: str, text: str, style: str, examples: str) -> dict:
    prompt = (
        f"Ты — креативный ИИ-ассистент. Твоя задача — проанализировать рефлексии школьника и придумать для него шуточную номинацию.\n"
        f"Стиль номинации: {style}.\n"
        f"Вот примеры для вдохновения:\n{examples}\n\n"
        f"На основе рефлексий участника по имени {username}: \"{text}\", присвой ему уникальную шуточную номинацию в заданном стиле и дай краткое (1-2 предложения) позитивное обоснование обезличенное. "
        "Верни JSON-объект: {\"nomination\": str, \"justification\": str}."
    )
    default_result = {"nomination": "Морской Исследователь", "justification": "За активное участие в проекте!"}
    try:
        response = await client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=[{"role": "user", "content": prompt}], temperature=0.8, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        return result if 'nomination' in result and 'justification' in result else default_result
    except Exception as e:
        print(f"Error generating nomination for {username}: {e}")
        return default_result

async def _generate_nominations_async(_df: pd.DataFrame, client: AsyncOpenAI, style: str, examples: str) -> pd.DataFrame:
    user_reflections = _df.groupby('username')['text'].apply(lambda texts: ' '.join(texts.astype(str).str.strip())).reset_index()
    tasks = [_get_one_nomination(client, row['username'], row['text'], style, examples) for _, row in user_reflections.iterrows()]
    results = await asyncio.gather(*tasks)
    results_df = pd.DataFrame(results)
    return pd.concat([user_reflections[['username']], results_df], axis=1).rename(columns={'username': 'ФИО', 'nomination': 'Номинация', 'justification': 'Обоснование'})

@st.cache_data(show_spinner=False, hash_funcs={AsyncOpenAI: lambda _: None})
def get_cached_nominations(_df: pd.DataFrame, client: AsyncOpenAI, style: str, examples: str) -> pd.DataFrame:
    return asyncio.run(_generate_nominations_async(_df, client, style, examples))

async def _get_one_friendly_reflection(client: AsyncOpenAI, username: str, text: str) -> dict:
    prompt = (
        "Ты — ИИ-ассистент, суммирующий рефлексии школьников с морской научно-технической проектной смены. "
        f"На основе рефлексий участника по имени {username}: \"{text}\", создай дружелюбное, шуточную харакетристику (2-3 абзаца) с инсайтами из рефлексий "
        "и позитивное напутствие (1 абзац). Тон должен быть позитивным, не обидным, мотивировать на дальнейшее развитие в учебе, проектой деятельности и в жизни с учетом морской тематики. "
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
        st.markdown("""...""") # Скрыто для краткости

    supabase = init_supabase_client()
    if not supabase: st.stop()
    
    client = None
    try:
        client = AsyncOpenAI(base_url=DEESEEK_API_URL, api_key=st.secrets["DEEPSEEK_API_KEY"])
    except KeyError:
        st.sidebar.warning("API-ключ DeepSeek не найден. Функции генерации будут недоступны.")
    except Exception as e:
        st.sidebar.error(f"Ошибка инициализации API: {e}")

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

    file_key = st.session_state.get('current_file_name')
    if 'last_file_key' not in st.session_state or st.session_state.last_file_key != file_key:
        st.session_state.show_nominations = False
        st.session_state.show_reflections = False
        st.session_state.last_file_key = file_key

    if df is None:
        st.info("Пожалуйста, загрузите файл или выберите отчет.")
        return

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
    
    if selected_source == "Новый анализ" and uploaded_file:
        st.sidebar.header("💾 Сохранение")
        if st.sidebar.button("Сохранить в архив"):
            with st.spinner("Сохранение отчета в облако..."):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                base_filename = os.path.splitext(uploaded_file.name)[0]
                report_filename = f"{base_filename}_processed_{timestamp}"
                df_to_save = st.session_state[session_key].copy()
                df_to_save['report_name'] = report_filename
                if 'data' in df_to_save.columns:
                    df_to_save['data'] = pd.to_datetime(df_to_save['data']).dt.strftime('%Y-%m-%dT%H:%M:%S')
                data_to_upload = df_to_save.replace({pd.NaT: None, np.nan: None}).to_dict(orient='records')
                try:
                    supabase.table('reports').upsert(data_to_upload, on_conflict='username,data').execute()
                    st.sidebar.success(f"Анализ сохранен как:\n**{report_filename}**")
                    st.cache_data.clear()
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
    
    nomination_style = st.sidebar.text_input("Задайте стиль номинаций:", "Морская научно-техническая тематика")
    nomination_examples = st.sidebar.text_area("Примеры номинаций (каждый с новой строки):", "Капитан Гениальности\nИнженер Глубин\nАдмирал Идей")
    if st.sidebar.button("Сгенерировать шуточные номинации"): 
        st.session_state.show_nominations = True
        st.rerun()

    st.sidebar.markdown("---") 

    reflection_style = st.sidebar.text_input("Задайте стиль характеристик:", "Дружелюбный и мотивирующий, с морскими метафорами")
    reflection_examples = st.sidebar.text_area("Примеры характеристик (помогут задать тон):", "Этот юнга показал себя настоящим морским волком в решении задач, не боялся штормов критики и всегда держал курс на успех. Его вклад в проект подобен маяку, освещающему путь всей команде.")
    if st.sidebar.button("Сгенерировать дружелюбные характеристики"): 
        st.session_state.show_reflections = True
        st.rerun()
    
    if st.session_state.get('show_nominations') or st.session_state.get('show_reflections'):
        if st.sidebar.button("Скрыть доп. таблицы", type="primary"):
            st.session_state.show_nominations = False
            st.session_state.show_reflections = False
            st.rerun()

    # --- 1. ВСЕГДА ОТОБРАЖАЕМ ОСНОВНОЙ ДАШБОРД ---
    st.header("Общая динамика и групповой анализ")

    daily_groups = filtered_df.groupby(filtered_df['data'].dt.date)
    agg_dict = {'avg_emotion': ('emotion', 'mean'), 'avg_sentiment_10_point': ('sentiment_10_point', 'mean'), 'avg_learning_sentiment': ('learning_sentiment_10_point', 'mean'), 'avg_teamwork_sentiment': ('teamwork_sentiment_10_point', 'mean'), 'avg_organization_sentiment': ('organization_sentiment_10_point', 'mean')}
    valid_agg_dict = {k: v for k, v in agg_dict.items() if v[0] in filtered_df.columns}
    if valid_agg_dict:
        daily_df = daily_groups.agg(**valid_agg_dict).reset_index().rename(columns={'data': 'Дата'}).sort_values('Дата')
        if not daily_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                if 'avg_sentiment_10_point' in daily_df.columns and 'avg_emotion' in daily_df.columns:
                    st.plotly_chart(px.line(daily_df, x='Дата', y=['avg_sentiment_10_point', 'avg_emotion'], title='Тональность vs. Самооценка'), use_container_width=True)
            with c2: 
                aspect_cols = ['avg_learning_sentiment', 'avg_teamwork_sentiment', 'avg_organization_sentiment']
                if all(c in daily_df.columns for c in aspect_cols):
                    fig = px.line(daily_df, x='Дата', y=aspect_cols, title='Динамика по аспектам')
                    fig.for_each_trace(lambda t: t.update(name = {'avg_learning_sentiment': 'Учёба', 'avg_teamwork_sentiment': 'Команда', 'avg_organization_sentiment': 'Организация'}.get(t.name, t.name)))
                    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Тепловая карта тональности группы")
    if 'sentiment_10_point' in filtered_df.columns:
        heatmap_data = filtered_df.pivot_table(index='username', columns=filtered_df['data'].dt.date, values='sentiment_10_point', aggfunc='mean')
        if not heatmap_data.empty: st.plotly_chart(px.imshow(heatmap_data, labels=dict(x="Дата", y="Ученик", color="Тональность"), color_continuous_scale='RdYlGn', aspect="auto"), use_container_width=True)
        else: st.info("Недостаточно данных для тепловой карты.")

    st.header("Анализ по отдельным учащимся")
    student_list = sorted(filtered_df['username'].unique())
    if student_list:
        if student := st.selectbox("Выберите ученика:", student_list):
            student_df = filtered_df[filtered_df['username'] == student].sort_values('data')
            c1, c2 = st.columns([3, 2])
            with c1:
                if 'sentiment_10_point' in student_df.columns and 'emotion' in student_df.columns:
                    st.plotly_chart(px.line(student_df, x='data', y=['sentiment_10_point', 'emotion'], title=f'Тональность vs. Самооценка'), use_container_width=True)
            with c2:
                radar_cols = ['emotion', 'learning_sentiment_10_point', 'teamwork_sentiment_10_point', 'organization_sentiment_10_point']
                if all(c in student_df.columns for c in radar_cols):
                    radar_values = [student_df[col].mean() for col in radar_cols]
                    fig_radar = go.Figure(data=go.Scatterpolar(r=radar_values, theta=['Самооценка', 'Учёба', 'Команда', 'Организация'], fill='toself'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 10])), title=f"Средние оценки для {student}")
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            display_cols = ['data', 'text', 'emotion', 'sentiment_10_point', 'learning_sentiment_10_point', 'teamwork_sentiment_10_point', 'organization_sentiment_10_point', 'learning_feedback', 'teamwork_feedback', 'organization_feedback']
            st.dataframe(student_df[[col for col in display_cols if col in student_df.columns]])

    st.header("Анализ \"Зоны риска\"")
    if 'sentiment_score' in filtered_df.columns:
        risk_users = filtered_df[filtered_df['sentiment_score'] < 0].groupby('username').size().reset_index(name='negative_count').query('negative_count > 1').sort_values('negative_count', ascending=False)
        if not risk_users.empty:
            st.warning("Выявлены участники с многократной негативной тональностью:")
            st.dataframe(risk_users)
        else: st.success("Участников с повторяющимся негативом не выявлено.")

    # --- ИЗМЕНЕНИЕ: ДОБАВЛЕН БЛОК С ОБЩЕЙ СВОДНОЙ ТАБЛИЦЕЙ ---
    st.header("Общая сводная таблица рефлексий")
    st.markdown("Здесь представлены все отфильтрованные записи с результатами анализа. Таблицу можно сортировать, нажимая на заголовки столбцов.")
    
    # Определяем порядок и состав столбцов для вывода
    summary_display_cols = [
        'username', 'data', 'text', 'emotion', 
        'sentiment_10_point', 'learning_sentiment_10_point', 
        'teamwork_sentiment_10_point', 'organization_sentiment_10_point',
        'learning_feedback', 'teamwork_feedback', 'organization_feedback'
    ]
    # Отбираем только те столбцы, которые есть в датафрейме, чтобы избежать ошибок
    available_cols = [col for col in summary_display_cols if col in filtered_df.columns]
    
    if available_cols:
        st.dataframe(filtered_df[available_cols], use_container_width=True)
    else:
        st.info("Нет обработанных данных для отображения в сводной таблице.")
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # --- 2. УСЛОВНО ОТОБРАЖАЕМ ДОПОЛНИТЕЛЬНЫЕ ТАБЛИЦЫ ---
    if st.session_state.get('show_nominations'):
        if not client:
            st.error("Генерация номинаций невозможна: API-ключ не настроен.")
        else:
            st.header("🏆 Шуточные номинации участников")
            nominations_key = f"nominations_{session_key}_{hash(nomination_style)}_{hash(nomination_examples)}"
            if nominations_key not in st.session_state:
                with st.spinner("Создаем номинации по вашему стилю..."):
                    st.session_state[nominations_key] = get_cached_nominations(filtered_df, client, nomination_style, nomination_examples)
            st.dataframe(st.session_state[nominations_key], use_container_width=True)

    if st.session_state.get('show_reflections'):
        if not client:
            st.error("Генерация характеристик невозможна: API-ключ не настроен.")
        else:
            st.header("🌟 Дружелюбные характеристики и напутствия")
            reflections_key = f"reflections_{session_key}_{hash(reflection_style)}_{hash(reflection_examples)}"
            if reflections_key not in st.session_state:
                with st.spinner("Пишем дружеские послания в заданном стиле..."):
                    st.session_state[reflections_key] = get_cached_friendly_reflections(filtered_df, client, reflection_style, reflection_examples)
            
            df_to_display = st.session_state[reflections_key].copy()
            df_to_display['Рефлексия и напутствие'] = df_to_display['Рефлексия'] + '\n\n**Пожелание:** ' + df_to_display['Пожелание']
            st.dataframe(df_to_display[['ФИО', 'Рефлексия и напутствие']], use_container_width=True)

if __name__ == "__main__":
    main()
