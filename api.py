
from pathlib import Path
from uuid import uuid4
from pypdf.errors import PdfReadError
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import app as agent_app
from src.ingestion import ingest_pdf
from src.logger import get_logger

logger = get_logger("ai_research_agent")

api = FastAPI(title="AI Research Agent API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


class ChatResponse(BaseModel):
    answer: str
    category: str


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    state = {
        "user_query": question,
        "llm_result": "",
        "needs_calculation": False,
        "is_calculator_query": False,
    }

    try:
        result = agent_app.invoke(state)

        return ChatResponse(
            answer=str(result.get("llm_result", "No answer returned.")),
            category=str(result.get("query_category", "unknown")),
        )

    except Exception as exc:
        logger.exception("Chat API request failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "The agent could not process your question. "
                "Check the server logs."
            ),
        ) from exc


@api.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF must be 20 MB or smaller.",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename).name
    file_path = upload_dir / f"{uuid4().hex}_{filename}"

    try:
        file_path.write_bytes(content)

        result = ingest_pdf(str(file_path))

        logger.info(
            "PDF uploaded and indexed: %s (%d pages, %d chunks)",
            filename,
            result["pages"],
            result["chunks"],
        )

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": filename,
            "pages": result["pages"],
            "chunks": result["chunks"],
        }

    except Exception as exc:
        logger.exception("PDF upload or ingestion failed.")

        if isinstance(exc, (PdfReadError, ValueError)):
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable PDF file.",
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="PDF ingestion failed. Check the server logs.",
        ) from exc
    finally:
        await file.close()
