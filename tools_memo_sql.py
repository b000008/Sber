import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver # сервер checkpointer

from dotenv import load_dotenv
load_dotenv(".env")


def MemorySql():
    """
    создает обьект для взаимодействия с базой данных  - sqlite3
        db_path: Имя файла для сохранения
    """
    db_path = os.getenv("DB_PATH")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return memory


if __name__ == "__main__":
    MemorySql()
    print(f"* Результат MemorySql ")