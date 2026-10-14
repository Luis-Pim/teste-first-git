from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.assistant import agent
from app.core.config import get_settings
from app.core.db import get_db
from app.routers.portfolio import PID, ref_date

router = APIRouter(prefix="/api", tags=["assistente"])


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class ChatIn(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=40)


@router.get("/assistant/status")
def status():
    s = get_settings()
    return {"configured": bool(s.openai_api_key), "model": s.openai_model, "tools": [t["name"] for t in agent.TOOLS]}


@router.post("/assistant/chat")
def chat(body: ChatIn, pid: int = PID, today: date = Depends(ref_date), db: Session = Depends(get_db)):
    s = get_settings()
    try:
        return agent.run(db, pid, today, [m.model_dump() for m in body.messages],
                         api_key=s.openai_api_key, model=s.openai_model, base_url=s.openai_base_url)
    except agent.AssistantUnavailable as e:
        raise HTTPException(503, str(e)) from e
