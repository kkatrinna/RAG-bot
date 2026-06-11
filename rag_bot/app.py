import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "akdengi/saiga-llama3-8b"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "etl_db"),
    "user": os.getenv("POSTGRES_USER", "etl_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "etl_password")
}


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.sidebar.warning(f"Нет подключения к БД: {e}")
        return None


def query_database(question: str) -> tuple:

    question_lower = question.lower()

    if any(word in question_lower for word in ["температура", "погода", "weather", "temperature"]):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT city, temperature, humidity, wind_speed, recorded_at 
                        FROM weather_metrics 
                        ORDER BY recorded_at DESC 
                        LIMIT 10
                    """)
                    rows = cur.fetchall()
                    conn.close()

                    if rows:
                        df = pd.DataFrame(rows)
                        # Создаём график
                        fig = px.line(df, x='recorded_at', y='temperature', color='city',
                                      title='Температура по городам')

                        response = "**Последние данные о погоде:**\n\n"
                        for row in rows[:5]:
                            response += f"- {row['city']}: {row['temperature']}°C, влажность {row['humidity']}%\n"

                        return response, fig
                    else:
                        return " В базе данных пока нет информации о погоде. Дождитесь загрузки данных.", None
            except Exception as e:
                return f"Ошибка запроса: {e}", None

    elif any(word in question_lower for word in ["биткоин", "bitcoin", "крипто", "crypto", "цена"]):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT symbol, price_usd, market_cap_usd, volume_24h, recorded_at 
                        FROM crypto_prices 
                        ORDER BY recorded_at DESC 
                        LIMIT 10
                    """)
                    rows = cur.fetchall()
                    conn.close()

                    if rows:
                        df = pd.DataFrame(rows)
                        fig = go.Figure()
                        for symbol in df['symbol'].unique():
                            symbol_df = df[df['symbol'] == symbol]
                            fig.add_trace(go.Scatter(x=symbol_df['recorded_at'], y=symbol_df['price_usd'],
                                                     mode='lines+markers', name=symbol.upper()))
                        fig.update_layout(title='Цены криптовалют (USD)', xaxis_title='Время', yaxis_title='Цена')

                        response = "**Последние цены криптовалют:**\n\n"
                        for row in rows[:5]:
                            response += f"- {row['symbol'].upper()}: ${row['price_usd']:,.0f}\n"

                        return response, fig
                    else:
                        return " В базе данных пока нет информации о криптовалютах. Дождитесь загрузки данных.", None
            except Exception as e:
                return f"Ошибка запроса: {e}", None

    elif any(word in question_lower for word in ["статистика", "сколько записей", "объем данных"]):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as count FROM weather_metrics")
                    weather_count = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) as count FROM crypto_prices")
                    crypto_count = cur.fetchone()[0]

                    conn.close()

                    response = f""" **Статистика базы данных:**

- Записей о погоде: {weather_count}
- Записей о криптовалютах: {crypto_count}
- Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Данные обновляются автоматически каждые 30 минут через ETL-пайплайн (Prefect)."""
                    return response, None
            except Exception as e:
                return f"Ошибка: {e}", None

    return None, None


st.set_page_config(
    page_title="RAG Бот + ETL Analytics",
    layout="wide"
)

st.title("RAG Бот + ETL Analytics")
st.markdown("""
**Возможности:**
- Задавайте вопросы по загруженным документам (PDF/TXT)
- Спрашивайте о погоде и криптовалютах (данные из ETL)
- Смотрите интерактивные графики
""")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Загрузка документов")

    uploaded_files = st.file_uploader(
        "Загрузите PDF или TXT файлы",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Индексировать документы", use_container_width=True):
        if uploaded_files:
            with st.spinner("Индексация документов..."):
                documents = []

                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False,
                                                     suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        if uploaded_file.name.endswith('.pdf'):
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                        else:
                            loader = TextLoader(tmp_path, encoding='utf-8')
                            docs = loader.load()

                        for doc in docs:
                            doc.metadata['source'] = uploaded_file.name

                        documents.extend(docs)
                    except Exception as e:
                        st.error(f"Ошибка загрузки {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

                if not documents:
                    st.error("Не удалось загрузить документы")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)

                embeddings = OllamaEmbeddings(
                    base_url=OLLAMA_BASE_URL,
                    model=EMBEDDING_MODEL
                )

                try:
                    test_embedding = embeddings.embed_query("тест")
                    st.success(f"Эмбеддинги работают (размер: {len(test_embedding)})")
                except Exception as e:
                    st.error(f"Ошибка эмбеддингов: {e}")
                    st.stop()

                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

                st.success(f"Индексировано {len(documents)} документов, {len(chunks)} чанков")
                st.session_state.messages = []
        else:
            st.error("Пожалуйста, выберите файлы для загрузки")

    st.divider()

    st.header("ETL Статус")
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) as weather_count FROM weather_metrics
                """)
                weather_count = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*) as crypto_count FROM crypto_prices
                """)
                crypto_count = cur.fetchone()[0]

                st.metric("Записей о погоде", weather_count)
                st.metric("Записей о крипте", crypto_count)
        except:
            st.warning("Таблицы ещё не созданы. Запустите ETL пайплайн.")
        finally:
            conn.close()
    else:
        st.warning("PostgreSQL не подключен. ETL данные недоступны.")

    st.divider()

    st.header("Ollama Статус")
    try:
        import requests

        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("Ollama запущен")
            models = response.json().get("models", [])
            if models:
                with st.expander("Доступные модели"):
                    for model in models:
                        st.code(model["name"])
        else:
            st.error("Ollama не отвечает")
    except:
        st.error("Ollama не запущен")

if st.session_state.vector_store is None and not any(m["role"] == "assistant" for m in st.session_state.messages):
    st.info(
        "**Начните работу:**\n\n1. Загрузите документы в боковой панели\n2. Или задайте вопрос о данных ETL (погода/крипта)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "figure" in message and message["figure"]:
            st.plotly_chart(message["figure"], use_container_width=True)

        if "sources" in message and message["sources"]:
            with st.expander("📚 Источники"):
                for source in message["sources"]:
                    st.text(source[:300])

if prompt := st.chat_input("Задайте вопрос по документации или спросите о погоде/крипте..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Анализирую запрос..."):
            try:
                db_answer, figure = query_database(prompt)

                if db_answer:
                    st.markdown(db_answer)
                    if figure:
                        st.plotly_chart(figure, use_container_width=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": db_answer,
                        "figure": figure,
                        "sources": []
                    })

                elif st.session_state.vector_store is not None:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 4}
                    )

                    llm = Ollama(
                        base_url=OLLAMA_BASE_URL,
                        model=LLM_MODEL,
                        temperature=0.3
                    )

                    prompt_template = """
                    Ты — полезный ассистент, который отвечает на вопросы, основываясь ТОЛЬКО на предоставленном контексте.

                    Контекст из документации:
                    {context}

                    Вопрос пользователя: {question}

                    Правила:
                    1. Если ответа нет в контексте — скажи, что информации нет в документации
                    2. В конце каждого утверждения указывай источник в формате: [источник: имя_файла]
                    3. Будь конкретным и точным

                    Ответ:
                    """

                    PROMPT = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": PROMPT},
                        return_source_documents=True
                    )

                    result = qa_chain({"query": prompt})
                    answer = result["result"]
                    source_docs = result["source_documents"]

                    sources_info = []
                    for i, doc in enumerate(source_docs, 1):
                        source_name = doc.metadata.get("source", "неизвестно")
                        page = doc.metadata.get("page", "")
                        content_preview = doc.page_content[:200].replace("\n", " ")

                        if page:
                            source_info = f"{i}. {source_name} (стр. {page})\n   {content_preview}..."
                        else:
                            source_info = f"{i}. {source_name}\n   {content_preview}..."

                        sources_info.append(source_info)

                    st.markdown(answer)

                    if sources_info:
                        with st.expander("Показать источники (цитаты из документации)"):
                            for source in sources_info:
                                st.text(source)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_info,
                        "figure": None
                    })

                else:
                    st.info(
                        "Загрузите документы в боковой панели, чтобы я мог отвечать на вопросы по ним.\n\nИли спросите о погоде/криптовалютах!")

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
                st.info(
                    "Убедитесь, что:\n1. Ollama запущен\n2. Модели загружены\n3. PostgreSQL работает (для ETL данных)")