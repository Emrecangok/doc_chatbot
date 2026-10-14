from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage,HumanMessage,RemoveMessage,AIMessage,ToolMessage
from langchain_core.documents import Document
from langchain.tools import tool


from llm import OpenAILLM
from state import DocChatState
from config import Config
from rag import index_file,list_sources,document_overview,format_document_list


import os
from typing import Literal,Optional
from pydantic import BaseModel




_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
DOC_NOT_FOUND = "Document not found" 
SEARCH_TOOL_NAME = "search_document"



def last_tool_messages(messages) -> list[ToolMessage]:
    out = []

    for message in reversed(messages):
        if not isinstance(message,ToolMessage):
            break
        out.append(message)
    return out[::-1]

def last_search_args(messages) -> dict:
    for message in reversed(messages):
        if isinstance(message,AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                if call["name"] == SEARCH_TOOL_NAME:
                    return call["args"]

    return {}

def _extract_text(content) -> str:

    if isinstance(content,str):
        return content
    if isinstance(content,list):
        return "".join(
         b.get("text","") for b in content if isinstance(b,dict) and b.get("type") == "text"
        )
    return str(content)

def route_after_tools(state:DocChatState) -> Literal["grade_docs","agent"]:
        search_messages = [m for m in last_tool_messages(state["messages"]) if m.name == SEARCH_TOOL_NAME]

        if not search_messages:
            return "agent"

        if all(str(m.content).startswith(DOC_NOT_FOUND) for m in search_messages):
            return "agent"

        return "grade_docs"



def last_question(messages) -> str:
    for m in reversed(messages):
        if isinstance(m,HumanMessage):
            return m.content

def _with_summary(system_prompt: str, state:DocChatState) ->str:

    summary = state.get("summary")
    if not summary:
        return system_prompt
    return f"{system_prompt}\n\n<conversation_summary>\n{summary}\n</conversation_summary"

def _load_prompts(filename:str) -> str:
    with open(os.path.join(_PROMPTS_DIR,filename),"r",encoding="utf-8") as f:
        return f.read()


def format_docs(docs : str) -> str:
    retrive_docs = []

    for doc in docs:
        source = doc.metadata.get("source")
        page_no = doc.metadata.get("page")
        content =  doc.page_content
        retrive_docs.append(f"source :{source}  page_no :{page_no} \n content:{content}") 
    return "\n\n --- \n\n".join(retrive_docs)


def make_tool(vector_store:Chroma):

    retriver = vector_store.as_retriever(search_kwargs={"k":Config.RETRIVER_K})

    @tool
    def list_documents() -> str:
        """Lists the documents the user has uploaded. Use it when the user asks which
        documents are available or how many documents were uploaded."""
        names = list_sources(vector_store=vector_store)
        if not names:
            return "No documents have been uploaded yet"
        return "\n".join(f"- {name}" for name in names)

    @tool(response_format="content_and_artifact")
    def search_document(query:str,document_names : Optional[list[str]] = None):
        """Searches the uploaded documents. Use it for every question about document
        content; never answer from your own knowledge.

        Args:
            query: Short search query with the key terms of the question.
            document_names: EXACT document names copied from the 'Uploaded documents'
                list in the system prompt. Leave empty to search ALL documents."""

        search_kwargs = {"k":Config.RETRIVER_K}

        if document_names:
            available = list_sources(vector_store=vector_store)
            unknown = [name for name in document_names if name not in available]
            if unknown:
                return (f"{DOC_NOT_FOUND}: {','.join(unknown)}. "
                        f"Available docs: {','.join(available)}",[])


        
            
            search_kwargs["filter"] = {"source": {"$in": document_names}}


        docs = vector_store.similarity_search(query,**search_kwargs)
        return format_docs(docs=docs),docs

    return list_documents,search_document

        

        
        
        

class GradeResult(BaseModel):
    score  : Literal["relevant","not_relevant"]
    reasoning : str


class DocChatNodes():

    def __init__(self,vector_store:Chroma):
        self.vector_store = vector_store
        self.llm = OpenAILLM().get_llm()
        self.system_prompt = _load_prompts("system_prompt.txt")
        self.grade_prompt = _load_prompts("grade_prompt.txt")
        self.rewrite_prompt = _load_prompts("rewrite_prompt.txt")
        self.generate_prompt = _load_prompts("generate_prompt.txt")

        list_documents,search_document = make_tool(vector_store)
        self.tools = [list_documents,search_document]
        self.search_tool_name = search_document.name
        self.model_with_tools = self.llm.bind_tools(self.tools)
        self.grader = self.llm.with_structured_output(GradeResult)

    def agent_node(self,state:DocChatState):

        overview = document_overview(self.vector_store)

        system_text = (_with_summary(self.system_prompt,state)+ f"\n\n## Uploaded Documents({len(overview)}) \n" +
                       format_document_list(overview))
        system = SystemMessage(content=system_text)
        messages = [system] + list(state["messages"])
        response = self.model_with_tools.invoke(messages)

        update = {"messages": [response]}
        if isinstance(state["messages"][-1], HumanMessage):
            update["retry_count"] = 0
        return update

   
        
        
    def ingest_node(self,state:DocChatState):
        for path in state.get("files",[]):
            index_file(path)
        return {"files":[]}

    def grade_docs_node(self,state:DocChatState):

        search_messages = [m for m in last_tool_messages(state["messages"]) if m.name ==SEARCH_TOOL_NAME]

        docs = [doc for m in search_messages for doc in (m.artifact or [])]

        if not docs:
            return {"documents":[],"doc_grade":"not_relevant"}

        result =  self.grader.invoke([
            SystemMessage(content=self.grade_prompt),
            HumanMessage(content=f" Question,query: {last_question(state['messages'])}\n\n Documents:{format_docs(docs)}")
        ])

        return {"documents": docs if result.score == "relevant" else [], "doc_grade": result.score}




    def rewrite_query_node(self,state:DocChatState):

        question = last_question(state["messages"])

        response =self.llm.invoke( [
            SystemMessage(content=self.rewrite_prompt),
            HumanMessage(content=f"original question {question}")
        ])

        better_query = _extract_text(response.content).strip().strip('"')


        args = {"query":better_query}
        previous_args = last_search_args(state["messages"])

        if previous_args.get("document_names"):
            args["document_names"] = previous_args["document_names"]

        retry_count = state.get("retry_count",0) + 1

        tool_call_message = AIMessage(
            content="",
            tool_calls=[{
                "name": SEARCH_TOOL_NAME,
                "args": args,
                "id": f"rewrite-{retry_count}",
            }],
        )

        return {"messages":[tool_call_message],"retry_count":retry_count}


    def fallback_generate_node(self,state:DocChatState):
        return {"messages": [AIMessage(content="I couldn't find this information in the uploaded documents.")]}

    def generate_node(self,state:DocChatState):

        question = last_question(state["messages"])
        response = self.llm.invoke([
            SystemMessage(content=self.generate_prompt),
            HumanMessage(content=f"Question:{question} \n\n Documentes \n {format_docs(state['documents'])}"),
        ])
        return {"messages":response}
       
    def trim_messages_node(self,state:DocChatState):
        messages = state["messages"]

        if len(messages) < Config.TRIM_TRIGGER:
            return {}
        cut = len(messages) - Config.TRIM_KEEP

        while cut > 0 and not isinstance(messages[cut],HumanMessage):
            cut -= 1
        return {"messages": [RemoveMessage(id=message.id) for message in messages[:cut]]}

def route_start(state:DocChatState) -> Literal["ingest","agent"]:
        return "ingest" if state.get("files") else "agent"


def route_after_grade(state: DocChatState) -> Literal["generate", "rewrite", "fallback"]:
        if state.get("doc_grade") == "relevant":
            return "generate"
        if state.get("retry_count", 0) >= Config.MAX_QUERY_RETIRES:
            return "fallback"
        return "rewrite"