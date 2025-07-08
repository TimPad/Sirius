# 1. Установка зависимостей при необходимости
# ----------------------
import subprocess
import sys

required_packages = [
    "pandas", "natasha", "streamlit", "requests", "plotly", "openpyxl"
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
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
                     NewsSyntaxParser, NewsNERTagger, Doc)
import streamlit as st
import requests
import plotly.express as px
from datetime import datetime

# Константы для API DeepSeek
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"

# ----------------------
# 3. Функция загрузки данных
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Загружает данные из Excel-файла и приводит колонку data к типу datetime.
    """
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

# ----------------------
# 4. Предобработка текста (NLP)
# ----------------------
# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

@st.cache_data
def preprocess_text(text: str) -> str:
    """
    Приведение текста к нижнему регистру, удаление пунктуации, цифр,
    удаление стоп-слов и лемматизация с помощью Natasha.
    """
    # Нижний регистр
    txt = text.lower()
    # Удаление цифр и пунктуации
    txt = ''.join(ch for ch in txt if ch.isalpha() or ch.isspace())
    # Лемматизация Natasha
    doc = Doc(txt)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    # Собираем леммы
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)
    # Удаление стоп-слов (простейший набор)
    stop_words = set(["и", "в", "на", "с", "не", "что", "как", "по"])
    lemmas = [lemma for lemma in lemmas if lemma not in stop_words]
    return ' '.join(lemmas)

import pandas as pd
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
                     NewsSyntaxParser, NewsNERTagger, Doc)
import streamlit as st

import plotly.express as px
from datetime import datetime
import json
from openai import OpenAI # ИЗМЕНЕНО: Импортируем OpenAI

# ----------------------
# 5. ИЗМЕНЕНО: DeepSeek API анализ рефлексии с использованием библиотеки OpenAI
# ----------------------

def analyze_reflection_with_deepseek(client: OpenAI, text: str) -> dict:
    """
    Посылает запрос к DeepSeek API для комплексного анализа текста.
    Использует библиотеку OpenAI и требует JSON-ответ.
    """
    # Запасной результат на случай ошибки
    error_result = {
        "sentiment_score": 0.0,
        "learning_feedback": "N/A",
        "teamwork_feedback": "N/A",
        "organization_feedback": "N/A"
    }
    if not text or not text.strip():
        return error_result

    prompt = (
        "Ты — ИИ-ассистент для анализа текстов рефлексии. Проанализируй рефлексию школьника. "
        "Твоя задача — вернуть JSON-объект со следующими ключами:\n"
        "1. 'sentiment_score': общая тональность текста, число от -1.0 (негатив) до 1.0 (позитив).\n"
        "2. 'learning_feedback': краткая выжимка (1-2 предложения) из текста об оценке учебного процесса.\n"
        "3. 'teamwork_feedback': краткая выжимка (1-2 предложения) об оценке работы в команде.\n"
        "4. 'organization_feedback': краткая выжимка (1-2 предложения) об оценке организационных и досуговых моментов.\n\n"
        "Если какой-то аспект в тексте не упоминается, оставь для соответствующего ключа пустую строку.\n\n"
        f"Текст для анализа: \"{text}\""
    )
    
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"} # Гарантирует ответ в формате JSON
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        # Проверяем, что все ключи на месте
        for key in error_result.keys():
            if key not in result:
                result[key] = error_result[key] # Добавляем недостающий ключ
        return result

    except Exception as e:
        st.error(f"Ошибка при вызове DeepSeek API: {e}")
        return error_result

# ----------------------
# 6. Конвертация тональности в шкалу 1-10 (без изменений)
# ----------------------
def convert_sentiment_to_10_point(score: float) -> float:
    return (score + 1) * 4.5 + 1

# ----------------------
# 7. ИЗМЕНЕНО: Основная логика и дашборд на Streamlit
# ----------------------
def main():
    st.title("Анализ рефлексий учащихся")

    api_key = st.text_input("Введите API-ключ DeepSeek:", type="password")
    if not api_key:
        st.warning("API-ключ необходим для запуска анализа.")
        return

    # ИЗМЕНЕНО: Создаем клиент один раз с ключом пользователя
    client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=api_key
    )

    uploaded_file = st.file_uploader("Загрузите Excel-файл с рефлексиями", type="xlsx")
    if not uploaded_file:
        return

    df = load_data(uploaded_file)
    df['lemmatized_text'] = df['text'].apply(preprocess_text)
    
    # Кэшируем результаты, чтобы не перезапускать анализ при каждом действии
    if 'df_processed' not in st.session_state:
        with st.spinner('Выполняется комплексный анализ рефлексий... Это может занять время.'):
            results = [analyze_reflection_with_deepseek(client, text) for text in df['text']]
            results_df = pd.DataFrame(results)
            df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
            st.session_state['df_processed'] = df
    else:
        df = st.session_state['df_processed']

    df['sentiment_10_point'] = df['sentiment_score'].apply(convert_sentiment_to_10_point)
    
    # Остальная часть функции main остается без изменений, так как формат данных сохранен
    # ... (код для группового анализа и отрисовки графиков) ...
    # ... (этот код полностью совпадает с кодом из исходного вопроса) ...

    # Групповой анализ по дням
    daily_summary = []
    # Группируем по дате, а не datetime, чтобы избежать проблем с временем
    daily_groups = df.groupby(df['data'].dt.date)

    with st.spinner('Выполняется групповой анализ по дням...'):
        for date, group in daily_groups:
            agg_text = ' '.join(group['lemmatized_text'])
            res = analyze_reflection_with_deepseek(client, agg_text)
            daily_summary.append({
                'data': pd.to_datetime(date), # Конвертируем дату обратно в datetime для графика
                'daily_sentiment_score': res['sentiment_score'],
                'daily_learning_feedback': res['learning_feedback'],
                'daily_teamwork_feedback': res['teamwork_feedback'],
                'daily_organization_feedback': res['organization_feedback'],
                'avg_emotion': group['emotion'].mean(),
                'avg_sentiment_10_point': group['sentiment_10_point'].mean()
            })
    daily_df = pd.DataFrame(daily_summary)
    if not daily_df.empty:
        daily_df.sort_values('data', inplace=True)

    # Раздел 1: Общая динамика
    st.header("Общая динамика по дням")
    fig = px.line(
        daily_df,
        x='data',
        y=['avg_sentiment_10_point', 'avg_emotion'],
        labels={'value': 'Значение', 'data': 'Дата'},
        title='Тональность и самооценка по дням'
    )
    st.plotly_chart(fig)
    st.dataframe(daily_df)

    # Раздел 2: Анализ по отдельным учащимся
    st.header("Анализ по отдельным учащимся")
    student = st.selectbox("Выберите ученика:", df['username'].unique())
    student_df = df[df['username'] == student].sort_values('data')
    fig2 = px.line(
        student_df,
        x='data',
        y=['sentiment_10_point', 'emotion'],
        labels={'value': 'Значение', 'data': 'Дата'},
        title=f'Динамика для {student}'
    )
    st.plotly_chart(fig2)
    st.dataframe(student_df[['data', 'text', 'sentiment_10_point', 'emotion',
                              'learning_feedback', 'teamwork_feedback', 'organization_feedback']])

    # Раздел 3: Полная таблица
    if st.checkbox("Показать полную таблицу с результатами"):
        st.dataframe(df)


if __name__ == "__main__":
    main()