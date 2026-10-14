from typing import Annotated, NotRequired
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DocChatState(TypedDict):


    messages: Annotated[list[BaseMessage], add_messages] 

    files: NotRequired[list[str]]           

    documents: NotRequired[list[Document]]  

    doc_grade: NotRequired[str]            
    
    retry_count: NotRequired[int]           