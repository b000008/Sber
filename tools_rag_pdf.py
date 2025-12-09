import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader # Or UnstructuredPDFLoader
from langchain_chroma import Chroma
from tools_llms import EmbeddingsGiga

load_dotenv(".env")
TEST = os.getenv("TEST")
print("итог tools_rag_pdf TEST :", TEST)

DATA_PATH =  "../SberRag/state_db/"  # os.getenv("PATH") #
# DATA_PATH = "../state_db/" # "../memoryGraf/state_db/"
PDF_FILENAME =  "Python-ed2.pdf" #  "Python-ed2.pdf" # # База данных ChromaDB  os.getenv("PDF_FILENAME")
CHROMA_PATH = os.getenv("CHROMA_PATH") # "chroma_db" # База данных ChromaDB

# # ################### загрузка документов #####################
def load_documents():
    """Загружает документы из указанного пути данных."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # half = len(documents) // 2  # Делим пополам, округляем до целого,  учитывыаем окно llm
    # halves = documents[:10]
    print(f"Loaded {len(documents)} page(s) from {pdf_path}")
    return documents

# # ################### Разделение документов #####################
def split_documents(documents):
    """Разделяет документы на более chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    # print(f"chunks {all_splits} ")
    return all_splits

# Инициализация модели Embedding ##############
def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Инициализирует или загружает хранилище векторов Chroma."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    # print(f"Векторное хранилище инициализировано/загружено из: {persist_directory}")
    return vectorstore


##########  Загрузка документа pdf_create()
def pdf_create():
    documents = load_documents()  # загрузили документы
    doc_chunks = split_documents(documents)  # разбили на чанки
    vector_store = get_vector_store(EmbeddingsGiga)
    # Создаём векторное хранилище из документов
    vector_store.add_documents(documents=doc_chunks)


##########  Поиск по созданной безе документа pdf_search( query)
def pdf_search( query ):
    # documents = load_documents()  # загрузили документы
    # doc_chunks = split_documents(documents)  # разбили на чанки
    vector_store = get_vector_store(EmbeddingsGiga)
    # # Создаём векторное хранилище из документов
    # vector_obj.add_documents(documents=doc_chunks)

    similarity_search = vector_store.similarity_search_by_vector(
        embedding=EmbeddingsGiga.embed_query(query), k=3
    )
    # content = []
    # for doc in similarity_search:
    #     content.append(doc.page_content)
    # print(f"* Результат поиска {similarity_search} ")
    return  [doc.page_content for doc in similarity_search]


# if __name__ == "__main__":
#     pdf_create()
#     results = pdf_search(" для чего используют Python")
#     print(f"* Результат поиска {results} ")