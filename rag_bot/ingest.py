import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

DOCUMENTS_DIR = "documents"
CHROMA_PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"


def load_documents():
    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Создана папка {DOCUMENTS_DIR}. Положите туда PDF или txt файлы.")
        return documents

    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)

        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"✓ Загружен PDF: {filename}")
        elif filename.endswith('.txt') or filename.endswith('.md'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            print(f"✓ Загружен текст: {filename}")

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} чанков")
    return chunks


def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"Векторная БД сохранена в {CHROMA_PERSIST_DIR}")
    return vector_store


def main():
    print("Начинаем загрузку документов...")

    docs = load_documents()
    if not docs:
        print("Нет документов в папке 'documents'. Добавьте PDF или txt файлы.")
        return

    chunks = split_documents(docs)

    print("Готово! Можно запускать 'streamlit run app.py'")


if __name__ == "__main__":
    main()