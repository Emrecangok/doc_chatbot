import os
from dotenv import load_dotenv


load_dotenv()



class Config:
    API_KEY             = os.getenv("OPENAI_API_KEY")
    MODEL_NAME          = os.getenv("OPENAI_MODEL_NAME")
    TEMPERATURE         = os.getenv("OPENAI_TEMPERATURE")
    MAX_RETRIES         = os.getenv("OPENAI_MAX_RETRIES",2)
    EMBEDDING_MODEL     = os.getenv("OPENAI_EMBEDDING_MODEL")
    RETRIVER_K          = int(os.getenv("RAG_RETRIVER_K",4))
    TRIM_TRIGGER       = int(os.getenv("TRIM_TRIGGER",12))
    TRIM_KEEP           = int(os.getenv("TRIM_KEEP",6))
    MAX_QUERY_RETIRES   = int(os.getenv("RAG_MAX_QUERY_RETRIES",2))
    


