# 1. Установка зависимостей при необходимости
# ----------------------
import subprocess
import sys

required_packages = [
    "pandas", "natasha", "streamlit", "requests", "plotly", "openpyxl", "openai", "wordcloud"
]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ----------------------
# 2. Импорты и константы
# ----------------------
import pandas as pd
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger)
from natasha import Doc
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
from openai import OpenAI
import numpy as np

# Константы для API DeepSeek
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"

# ----------------------
# 3. Функция загрузки данных (без изменений)
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

# ----------------------
# 4. Предобработка текста (NLP) (без изменений)
# ----------------------
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

@st.cache_data
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = ''.join(ch for ch in txt if ch.isalpha() or ch.isspace())
    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = [token.lemma for token in doc.tokens]
    stop_words = set(["и", "в", "на", "с", "не", "что", "как", "по"])
    lemmas = [lemma for lemma in lemmas if lemma not in stop_words]
    return ' '.join(lemmas)

# ----------------------
# 5. DeepSeek API анализ (без изменений)
# ----------------------
def analyze_reflection_with_deepseek(client: OpenAI, text: str) -> dict:
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
        response = client.chat.completions.create(
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
        st.error(f"Ошибка при вызове DeepSeek API: {e}")
        return error_result

# ----------------------
# 6. Конвертация тональности (без изменений)
# ----------------------
def convert_sentiment_to_10_point(score: float) -> float:
    if not isinstance(score, (int, float)):
        return 5.5
    return (score + 1) * 4.5 + 1

# ----------------------
# 7. Основная логика и дашборд на Streamlit
# ----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    # --- БЛОК АВТОРИЗАЦИИ С ВЫБОРОМ МЕТОДА ---
    api_key = None
    st.sidebar.header("🔐 Настройки API")
    auth_method = st.sidebar.radio(
        "Как предоставить API-ключ?",
        ("Ввести вручную", "Загрузить из файла (.txt)")
    )

    if auth_method == "Ввести вручную":
        api_key_input = st.sidebar.text_input(
            "Ваш API-ключ DeepSeek:",
            type="password",
            help="Ключ не будет сохранен и используется только для текущей сессии."
        )
        if api_key_input:
            api_key = api_key_input

    elif auth_method == "Загрузить из файла (.txt)":
        key_file = st.sidebar.file_uploader(
            "Выберите .txt файл с ключом",
            type=["txt"],
            help="Файл должен содержать только ваш API-ключ."
        )
        if key_file is not None:
            try:
                # Читаем содержимое файла, декодируем в строку и убираем лишние пробелы/переносы
                key_from_file = key_file.getvalue().decode("utf-8").strip()
                if key_from_file:
                    api_key = key_from_file
                    st.sidebar.success("Ключ из файла успешно загружен.")
                else:
                    st.sidebar.warning("Загруженный файл пуст.")
            except Exception as e:
                st.sidebar.error(f"Ошибка чтения файла: {e}")

    # Проверяем, получен ли ключ, и останавливаем выполнение, если нет
    if not api_key:
        st.info("Пожалуйста, предоставьте API-ключ в боковой панели для запуска анализа.")
        st.stop() # Gracefully stop the script execution

    # Если ключ предоставлен, инициализируем клиент
    client = OpenAI(base_url=DEESEEK_API_URL, api_key=api_key)

    # --- КОНЕЦ БЛОКА АВТОРИЗАЦИИ ---

    st.sidebar.header("📄 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл с рефлексиями", type="xlsx")
    if not uploaded_file:
        st.info("Пожалуйста, загрузите Excel-файл с рефлексиями, чтобы начать.")
        return

    # Остальная часть скрипта остается такой же, как в предыдущей версии
    df = load_data(uploaded_file)
    df['text'] = df['text'].astype(str).fillna('')

    # Кэширование результатов, чтобы избежать повторного анализа при смене фильтров
    session_key = f"df_processed_{uploaded_file.name}"
    if session_key not in st.session_state:
        with st.spinner('Выполняется комплексный анализ рефлексий... Это может занять время.'):
            results = [analyze_reflection_with_deepseek(client, text) for text in df['text']]
            results_df = pd.DataFrame(results)
            df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
            st.session_state[session_key] = df
    else:
        df = st.session_state[session_key]

    for col in ['sentiment_score', 'learning_sentiment_score', 'teamwork_sentiment_score', 'organization_sentiment_score']:
        df[col.replace('_score', '_10_point')] = df[col].apply(convert_sentiment_to_10_point)

    # --- ФИЛЬТР ПО ДАТАМ ---
    st.sidebar.header("📊 Фильтры")
    if not df['data'].dropna().empty:
        min_date = df['data'].min().date()
        max_date = df['data'].max().date()

        date_range = st.sidebar.slider(
            "Выберите диапазон дат:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD.MM.YYYY"
        )

        start_date, end_date = date_range
        mask = (df['data'].dt.date >= start_date) & (df['data'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        st.sidebar.warning("В файле отсутствуют корректные даты для фильтрации.")
        filtered_df = df

    if filtered_df.empty:
        st.error("Нет данных для выбранного диапазона дат. Пожалуйста, измените фильтр.")
        return

    # --- Раздел 1: Общая динамика и групповой анализ ---
    st.header("Общая динамика и групповой анализ")
    daily_groups = filtered_df.groupby(filtered_df['data'].dt.date)
    daily_df = daily_groups.agg(
        avg_emotion=('emotion', np.mean),
        avg_sentiment_10_point=('sentiment_10_point', np.mean),
        avg_learning_sentiment=('learning_sentiment_10_point', np.mean),
        avg_teamwork_sentiment=('teamwork_sentiment_10_point', np.mean),
        avg_organization_sentiment=('organization_sentiment_10_point', np.mean)
    ).reset_index()
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
            fig_details.for_each_trace(lambda t: t.update(name = new_names[t.name]))
            st.plotly_chart(fig_details, use_container_width=True)

    # --- Тепловая карта ---
    st.subheader("Тепловая карта тональности группы")
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

    # --- Анализ по отдельным учащимся ---
    st.header("Анализ по отдельным учащимся")
    student_list = sorted(filtered_df['username'].unique())
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

        ### НАЧАЛО ИЗМЕНЕНИЙ 1: ОПИСАНИЕ ТАБЛИЦЫ ###
        with st.expander("Показать описание столбцов таблицы"):
            st.markdown("""
            - **data**: Дата написания рефлексии.
            - **text**: Исходный текст рефлексии.
            - **emotion**: Самооценка эмоционального состояния, поставленная учеником (от 1 до 10).
            - **sentiment_10_point**: Общая тональность текста, оцененная ИИ и переведенная в 10-балльную шкалу.
            - **learning_sentiment_10_point**: Тональность части текста, касающейся *учебного процесса* (1-10).
            - **teamwork_sentiment_10_point**: Тональность части текста, касающейся *работы в команде* (1-10).
            - **organization_sentiment_10_point**: Тональность части текста, касающейся *организации и досуга* (1-10).
            - **learning_feedback**: Краткая выжимка от ИИ по учебному процессу.
            - **teamwork_feedback**: Краткая выжимка от ИИ по работе в команде.
            - **organization_feedback**: Краткая выжимка от ИИ по организации и досугу.
            """)
        ### КОНЕЦ ИЗМЕНЕНИЙ 1 ###

        st.dataframe(student_df[['data', 'text', 'emotion', 'sentiment_10_point',
                                  'learning_sentiment_10_point', 'teamwork_sentiment_10_point', 'organization_sentiment_10_point',
                                  'learning_feedback', 'teamwork_feedback', 'organization_feedback']])

    # --- Полная таблица ---
    if st.sidebar.checkbox("Показать полную таблицу с отфильтрованными результатами"):
        st.header("Полная таблица данных")
        st.dataframe(filtered_df)

    ### НАЧАЛО ИЗМЕНЕНИЙ 2: ТАБЛИЦА "ЗОНЫ РИСКА" ###
    st.header("Анализ \"Зоны риска\": участники с повторяющимся негативом")

    # Определяем негативные рефлексии (используем исходную шкалу от -1 до 1, где < 0 - негатив)
    negative_reflections = filtered_df[filtered_df['sentiment_score'] < 0]

    # Считаем количество негативных рефлексий для каждого участника
    if not negative_reflections.empty:
        negative_counts = negative_reflections.groupby('username').size().reset_index(name='negative_count')
        # Фильтруем тех, у кого негатив проявился больше одного раза
        at_risk_users = negative_counts[negative_counts['negative_count'] > 1].sort_values('negative_count', ascending=False)
    else:
        at_risk_users = pd.DataFrame(columns=['username', 'negative_count']) # Пустой DataFrame, если нет негатива

    if not at_risk_users.empty:
        st.warning("Внимание! Выявлены участники с многократной негативной тональностью в рефлексиях за выбранный период.")

        # Создаем HTML-таблицу с красными рамками для лучшей визуализации
        html_table = """
        <style>
            .risk-table {
                border-collapse: collapse; width: 100%;
                border: 2px solid #E57373; /* Светло-красная рамка */
            }
            .risk-table th, .risk-table td {
                border: 1px solid #E57373; padding: 10px; text-align: left;
            }
            .risk-table th {
                background-color: #FFEBEE; /* Очень светлый красный фон для заголовков */
                color: #B71C1C; /* Темно-красный цвет текста */
                font-weight: bold;
            }
        </style>
        <table class="risk-table">
            <thead>
                <tr>
                    <th>Участник</th>
                    <th>Количество негативных рефлексий</th>
                </tr>
            </thead>
            <tbody>
        """
        for index, row in at_risk_users.iterrows():
            html_table += f"<tr><td>{row['username']}</td><td>{row['negative_count']}</td></tr>"

        html_table += "</tbody></table>"

        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.success("За выбранный период не выявлено участников, у которых тональность рефлексий была бы негативной более одного раза.")
    ### КОНЕЦ ИЗМЕНЕНИЙ 2 ###


if __name__ == "__main__":
    main()