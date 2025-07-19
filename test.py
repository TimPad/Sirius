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
import io # --- НОВЫЙ ИМПОРТ --- для работы с файлами в памяти

### НОВОЕ: Импорты для работы с Supabase
from supabase import create_client, Client

# Константы для API DeepSeek
DEESEEK_API_URL = "https://api.studio.nebius.ai/v1/"

# Убедитесь, что openpyxl установлен, он нужен для .to_excel()
# pip install openpyxl

# ----------------------
# 3. Функция загрузки данных (без изменений)
# ----------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(col).strip().lower() if isinstance(col, str) else col for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    return df

# ----------------------
# 4. БЛОК ПРЕДОБРАБОТКИ ТЕКСТА (NLP) - УДАЛЕН
# ----------------------

# --- Весь код до функции main() остается без изменений ---
# --- Я пропущу его для краткости и перейду сразу к main(), куда мы добавим логику ---

# ... (все функции, включая analyze_reflection_with_deepseek, generate_nomination_with_llm,
# generate_character_description_with_llm, и функции для Supabase остаются здесь без изменений) ...

# --- НОВЫЙ БЛОК: Функция для конвертации DataFrame в Excel ---
def to_excel(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в байты Excel-файла."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Характеристики')
    processed_data = output.getvalue()
    return processed_data

# ----------------------
# 8. Основная логика и дашборд на Streamlit
# ----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Интерактивный дашборд для анализа рефлексий учащихся")

    # ... (весь код до раздела "Анализ по отдельным учащимся" остается без изменений) ...
    # ... (я пропущу его для краткости, чтобы показать только изменения) ...

    # Начало функции main()
    with st.expander("ℹ️ О проекте: что это и как пользоваться?", expanded=False):
        st.markdown("""
        **Цель дашборда** — помочь педагогам и кураторам быстро оценить эмоциональное состояние группы, выявить общие тенденции и определить учащихся, требующих особого внимания, на основе их письменных рефлексий.

        **Как это работает?**
        1.  Для **нового анализа** загрузите Excel-файл с текстами рефлексий.
        2.  Чтобы посмотреть **старый отчет**, выберите его из выпадающего списка. Данные загрузятся из облачного архива.
        3.  При новом анализе искусственный интеллект (DeepSeek) анализирует каждый текст.
        4.  После анализа вы можете **сохранить результат в архив**, нажав соответствующую кнопку. Отчет станет доступен для выбора при следующем запуске.
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
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        client = AsyncOpenAI(base_url=DEESEEK_API_URL, api_key=api_key)
    except KeyError:
        st.sidebar.error("API-ключ DeepSeek не найден в настройках приложения.")
        st.error("Ошибка конфигурации: отсутствует ключ `DEEPSEEK_API_KEY`. Пожалуйста, добавьте его в настройки секретов в Streamlit Cloud.")
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
            # ... логика сохранения ...
            pass # Оставляем без изменений

    if df.empty:
        st.error("Нет данных для отображения. Пожалуйста, измените фильтры или загрузите другой файл.")
        return
        
    filtered_df = df.copy()

    st.sidebar.header("📊 Фильтры")
    if 'data' in filtered_df.columns and not filtered_df['data'].dropna().empty:
        min_date, max_date = filtered_df['data'].min().date(), filtered_df['data'].max().date()
        if min_date != max_date:
            start_date, end_date = st.sidebar.slider("Выберите диапазон дат:", min_date, max_date, (min_date, max_date), format="DD.MM.YYYY")
            mask = (filtered_df['data'].dt.date >= start_date) & (filtered_df['data'].dt.date <= end_date)
            filtered_df = filtered_df.loc[mask]
        else:
             st.sidebar.info("В данных только один день, фильтр по дате неактивен.")
    else:
        st.sidebar.warning("В файле отсутствуют корректные даты для фильтрации.")

    if filtered_df.empty:
        st.error("Нет данных для отображения по выбранным фильтрам.")
        return

    # --- НОВЫЙ БЛОК: Массовая генерация и скачивание ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎓 Выпускные материалы")

    if st.sidebar.button("Сгенерировать характеристики для всех"):
        unique_students = filtered_df['username'].unique()
        
        async def generate_all_creative_content():
            tasks = []
            for student_name in unique_students:
                student_reflections = " ".join(filtered_df[filtered_df['username'] == student_name]['text'].dropna().astype(str))
                # Создаем две задачи для каждого студента
                nomination_task = generate_nomination_with_llm(client, student_reflections)
                description_task = generate_character_description_with_llm(client, student_name, student_reflections)
                tasks.extend([nomination_task, description_task])
            
            # Запускаем все задачи параллельно
            all_results = await asyncio.gather(*tasks)
            
            # Собираем результаты в структурированный список
            final_data = []
            for i, student_name in enumerate(unique_students):
                final_data.append({
                    "Ученик": student_name,
                    "Номинация": all_results[i*2],
                    "Характеристика": all_results[i*2 + 1]
                })
            return final_data

        with st.sidebar.spinner(f"Генерация контента для {len(unique_students)} учеников... Это может занять несколько минут."):
            try:
                creative_data_list = asyncio.run(generate_all_creative_content())
                creative_df = pd.DataFrame(creative_data_list)
                
                # Сохраняем готовый Excel-файл в сессию
                st.session_state['downloadable_excel'] = to_excel(creative_df)
                st.session_state['excel_filename'] = f"Характеристики_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                
                st.sidebar.success("Готово! Теперь можно скачать файл.")
                # st.rerun() # Необязательно, но может быть полезно для мгновенного обновления UI
            except Exception as e:
                st.sidebar.error(f"Произошла ошибка: {e}")

    # Кнопка скачивания появляется только после генерации файла
    if 'downloadable_excel' in st.session_state:
        st.sidebar.download_button(
            label="📥 Скачать файл Excel",
            data=st.session_state['downloadable_excel'],
            file_name=st.session_state['excel_filename'],
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    # --- КОНЕЦ НОВОГО БЛОКА ---

    st.header("Общая динамика и групповой анализ")
    # ... (код для графиков без изменений) ...
    
    st.header("Анализ по отдельным учащимся")
    student_list = sorted(filtered_df['username'].unique())
    if student_list:
        student = st.selectbox("Выберите ученика:", student_list)
        if student:
            # ... (вся логика для отображения данных одного студента остается без изменений) ...
            # ... (включая кнопку для индивидуальной генерации) ...
            pass # Оставляем как было

    # ... (оставшаяся часть кода main() без изменений) ...
    
if __name__ == "__main__":
    # Добавляем функции в глобальную область видимости для доступности
    # Этот блок не является частью исходного кода, но полезен для понимания,
    # что функции должны быть определены до их вызова в main()
    # (в нашем случае они уже определены выше, так что все в порядке)
    # Определим их здесь пустыми заглушками для ясности
    async def generate_nomination_with_llm(client, reflections): pass
    async def generate_character_description_with_llm(client, name, reflections): pass
    
    # Запускаем основной код
    # main() # Этот вызов уже есть в вашем коде, я просто показываю его место
