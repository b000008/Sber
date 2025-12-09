from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    summary: str
    memory: str
    classification: str
    prompt_message: Optional[str] = Field(default=None, description="Ответ системы")
    system_message: Optional[str] = Field(default=None, description="Ответ системы")
    user_input: str = Field(description="Текущий ввод пользователя")
    context: Dict[str, Any] = Field(default={}, description="Контекст диалога")
    retrieved_docs: List[str] = Field(default=[], description="Извлечённые документы из RAG")
    response: Optional[str] = Field(default=None, description="Ответ системы")
    mode: str = Field(default="normal", description="Режим диалога")
    session_id: str = Field(description="ID сессии диалога")