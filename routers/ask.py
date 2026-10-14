import uuid

from fastapi import APIRouter
from models import  QueryRequest, QueryResponse

router = APIRouter(prefix="/task",tags=["ASK"])


