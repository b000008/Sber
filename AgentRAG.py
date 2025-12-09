# from typing_extensions import Literal
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
import json

# ############  подкачка   ###############
from tools_state import AgentState # memoryGraf
from tools_llms import Giga # Загрузка LLM GigaChain
from tools_png_graph import Gen_png_graph # Загрузка сохранение графика в файл

from tools_rag_pdf import pdf_create, pdf_search # Локальная память
from tools_prompts import CLASS_PROMPT, BEGINNER_PROMPT, ADVANCED_PROMPT   # Загрузка промтов

# memory = MemorySql

##############################
import os
from dotenv import load_dotenv
load_dotenv(".env")
TEST = os.getenv("TEST")
print("итог Agent_rag load_dotenv TEST :", TEST)

db_path = os.getenv("DB_PATH")
#

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver # сервер checkpointer
conn = sqlite3.connect(db_path, check_same_thread=False)
MemorySql = SqliteSaver(conn)
# memory = MemorySql
# ############   ###############


#  Интегрировать поиск по памяти
def retrieve_memory(state: AgentState) -> AgentState:
    """ Чтение прошлого диалога из базы"""
    # print("retrieve_memory :\n")
    summary = state.get("summary", "")
    state["memory"] = summary
    state["user_input"] = state["messages"][-1].content
    state["classification"] = "Beginner"

    # print("retrieve_memory читаем 'memory':", state["memory"])
    # print(f"retrieve_memory читаем 'user_input' :{ state["user_input"]} читаем 'classification': {state["classification"]}")
    return state

#  Анализировать контекст диалога
def analyze_context(state: AgentState)-> AgentState:
    """ Анализировать контекст диалога """
    user_input = state["messages"][-1].content
    state["user_input"] = user_input
    summary = state.get("summary", "")
    # response = state["messages"]
    # print(f"analyze_context читаем user_input : {user_input} читаем summary : {summary}")
    # print("analyze_context читаем messages :",  state["messages"])
    if 'график' in user_input.lower():
        # Генерируем визуализацию по запросу и сохраняем - graph_example.png
        Gen_png_graph(graph, name_photo="graph_example.png")
        print("Граф сохранён как graph_example.png")
        # return END

    if 'загрузи документ' in user_input.lower():
        # Загрузка документа pdf_create()
        print("Загрузка документа в RAG")
        pdf_create()
        print("Документы загружены в RAG")
        # return END

    if summary:
        prompt_message = CLASS_PROMPT  # + SUMMARY_PROMPT
        summary_message = prompt_message.format(history=summary)
        system_message = [SystemMessage(content=summary_message)] + state["messages"]
        # print("analyze_context читаем state[messages] :", system_message)
    else:
        system_message = state["messages"]
        # print("analyze_context читаем state[messages]  без промта:", system_message)


    response = Giga.invoke(system_message)
        # state["messages"] = response
    state["response"] = response
    # print("analyze_context response :", response)
    if response.content.lower() in ["Advanced"]:
        state["classification"] = "Advanced"
        # print("analyze_context state[classification]Advanced  :", state["classification"])
    else:
        state["classification"] = "Beginner"
        # print("analyze_context state[classification]Beginner  :", state["classification"])
        # state["messages"] = response
    return state

# Выбрать режим диалога на основе контекста
def select_mode(state: AgentState)-> AgentState:
    """Выбрать режим диалога на основе контекста"""
    # print("select_mode :\n")
    summary = state.get("summary", "")
    user_input = state["user_input"]
    analyze_context = state["response"]
    # print("select_mode analyze_context :", analyze_context)
    if analyze_context.content.lower() in ["Advanced"]:
        state["classification"] = "Advanced"
        state["prompt_message"] = ADVANCED_PROMPT.format(history=summary)
        # print("analyze_context state[classification]Advanced  :", state["classification"])
    else:
        state["classification"] = "Beginner"
        pdf_search_chanc = pdf_search(user_input)
        documents = pdf_search_chanc if pdf_search_chanc != [] else "Пусто"
        state["prompt_message"] = BEGINNER_PROMPT.format(documents=documents, history=summary)
        # print("analyze_context state[classification]Beginner  :", state["classification"])
    return  state


# Сгенерировать ответ с использованием RAG
def generate_response(state: AgentState):
    """Сгенерировать ответ с использованием RAG"""
    # print("generate_response state[prompt_message]  :", state["prompt_message"])
    # print("generate_response state[messages]  :", state["messages"])
    # print("generate_response state[user_input]  :", state["user_input"])
    # print("generate_response state[response]  :", state["response"])
    # Формируем промпт с контекстом
    messages= [
        SystemMessage(content=state["prompt_message"]),
        HumanMessage(content=state["user_input"])
    ]
    response = Giga.invoke(messages)
    # print("generate_response читаем response :\n", response)
    return  {"messages": response}

# Определите, следует ли завершить или подвести итоги разговора

# Здесь создается summary чтобы сохранить историю ответ
def summarize_conversation(state: AgentState):
    """Создать summary"""
    # print("summarize_conversation :\n")
    summary = state["memory"]
    if summary:
        summary_message = (
            f"Это краткое содержание разговора на сегодняшний день: {summary}\n\n"
            "Расширь краткое содержание, приняв во внимание новые сообщения выше"
        )
    else:
        summary_message = "Создай краткое изложение беседы, приведённой выше:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = Giga.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Определите, следует ли завершить или подвести итоги разговора
def update_memory(state: AgentState):
    """Обновить векторную базу с новым взаимодействием"""
    # print("update_memory :\n")
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END

############ граф  ###############
# Определить ноды
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_memory", retrieve_memory) #  Интегрировать поиск по памяти
workflow.add_node("analyze_context", analyze_context) #  Анализировать контекст диалога
workflow.add_node("select_mode", select_mode) # Выбрать режим диалога на основе контекста
workflow.add_node("generate_response", generate_response) # Сгенерировать ответ с использованием RAG
workflow.add_node("summarize_conversation", summarize_conversation) # Здесь создается summary чтобы пересохранить историю ответ
workflow.add_node("update_memory", update_memory) # Обновить векторную базу с новым взаимодействием"""


#длбавляем возможность выбора следующего узла
workflow.add_edge(START, "retrieve_memory")
workflow.add_edge("retrieve_memory", "analyze_context")
workflow.add_edge("analyze_context", "select_mode")
workflow.add_edge("select_mode", "generate_response")
# workflow.add_edge("generate_response", "update_memory")
# workflow.add_edge("update_memory", "summarize_conversation")
workflow.add_conditional_edges("generate_response", update_memory)
workflow.add_edge("summarize_conversation", END)

#длбавляем возможность выбора следующего узла


# Компилируем график
graph = workflow.compile(checkpointer=MemorySql)

# ############ - Исполнение  ###############

def run_dialog():
    conversation_history = []
    config = {"configurable": {"thread_id": "1"}}
    print("ИИ-агент с долговременной памятью.")
    print("Для выхода напишите 'exit'.\n")
    while True:
        user_input = input("Enter: ")
        if user_input.lower() in ["exit", "выход", "quit"]:
            print("Агент: До встречи!")
            # MemoryText(conversation_history)
            break
        input_message = HumanMessage(content=user_input)
        output = graph.invoke({"messages": [input_message]}, config)
        # print("output:", output)
        for m in output['messages'][-1:]:
            m.pretty_print()
        print('-' * 6 + " Конец ответа " + '-' * 6 + '\n')

if __name__ == "__main__":
    run_dialog()


