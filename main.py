from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
from openai import OpenAI          # still use the openai package

load_dotenv()

app = FastAPI(title="TaskFlow AI Service (Groq)", version="1.0.0")

# ---------- Groq client (OpenAI-compatible) ----------
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# ----------------------------------------------------


class TaskIn(BaseModel):
    title: str
    description: str = ""
    status: str
    priority: str
    due_date: Optional[str] = None


class ProjectSummaryRequest(BaseModel):
    project_name: str
    project_description: str = ""
    project_status: str
    tasks: List[TaskIn] = Field(default_factory=list)


class ProjectSummaryResponse(BaseModel):
    summary: str
    key_insights: List[str]
    recommendations: List[str]
    progress_percentage: int


@app.get("/health")
def health():
    return {"status": "ok", "provider": "groq", "model": MODEL}


@app.post("/summarize-project", response_model=ProjectSummaryResponse)
def summarize_project(payload: ProjectSummaryRequest):
    if not payload.tasks and not payload.project_description.strip():
        raise HTTPException(status_code=400, detail="No content to summarize")

    tasks_text = "\n".join(
        f"- [{t.status}] ({t.priority}) {t.title}"
        + (f" — due {t.due_date}" if t.due_date else "")
        + (f"\n  {t.description}" if t.description else "")
        for t in payload.tasks
    ) or "No tasks yet."

    prompt = f"""
You are a helpful project management assistant for TaskFlow.

Project: {payload.project_name}
Status: {payload.project_status}
Description: {payload.project_description or "None"}

Tasks:
{tasks_text}

Return ONLY valid JSON with these exact keys:
- summary: 2-4 sentence overview of the current state
- key_insights: array of 3-5 short insights (strings)
- recommendations: array of 2-4 concrete next steps (strings)
- progress_percentage: integer 0-100 estimating completion

Be practical. Do not invent tasks that do not exist.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You always reply with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        data = json.loads(response.choices[0].message.content)
        return ProjectSummaryResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")