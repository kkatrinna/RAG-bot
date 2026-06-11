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

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "akdengi/saiga-llama3-8b"

st.set_page_config(
    page_title="RAG Бот по документации",
    layout="wide"
)

st.title("RAG Бот для вопросов по документации")
st.markdown("Загрузите документы и задавайте вопросы с цитированием источников.")

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

    if st.button("Индексировать документы"):
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
                    print(f"Эмбеддинги работают, размерность: {len(test_embedding)}")
                except Exception as e:
                    st.error(f"Ошибка эмбеддингов: {e}")
                    st.stop()

                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

                st.success(f"Индексировано {len(documents)} документов, {len(chunks)} чанков")
                st.session_state.messages = []  # Очищаем историю
        else:
            st.error("Пожалуйста, выберите файлы для загрузки")

    st.divider()

    st.header("Статус")
    try:
        import requests

        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("Ollama запущен")
            models = response.json().get("models", [])
            if models:
                st.write("Доступные модели:")
                for model in models:
                    st.code(model["name"])
        else:
            st.error("Ollama не отвечает")
    except:
        st.error("Ollama не запущен. Запустите: `ollama serve`")

if st.session_state.vector_store is None:
    st.info("Загрузите документы в боковой панели, чтобы начать")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Источники"):
                    for source in message["sources"]:
                        st.text(source[:300])

    if prompt := st.chat_input("Задайте вопрос по документации..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ в документации..."):
                try:
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
                        with st.expander("📚 Показать источники (цитаты из документации)"):
                            for source in sources_info:
                                st.text(source)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_info
                    })

                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
                    st.info("Убедитесь, что Ollama запущен и модели загружены")