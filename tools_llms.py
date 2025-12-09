# Каталог Sber
import os
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

load_dotenv(".env")
# credentials = os.getenv("GIGACHAT_CREDENTIALS")
# print("credentials: ",credentials)
#  настрока работы гигачат

# Инициализация GigaChat
Giga=GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False
)

# Инициализация эмбеддингов
EmbeddingsGiga=GigaChatEmbeddings(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False
)