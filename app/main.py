from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from app.scoring_service import W2VGOPScoringService

app = FastAPI(
    title="W2V GOP Scoring Service",
    description="Upload audio + transcript text and get per-phone pronunciation scores.",
    version="1.0.0",
)

service = W2VGOPScoringService()
startup_error: Optional[str] = None


def _raise_request_error(exc: Exception) -> None:
    reason = str(exc)
    if reason.lower().startswith("audio rejected"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "audio_rejected",
                "reject_reason": reason,
                "threshold_profile_version": str(service.config.threshold_profile_version),
            },
        ) from exc

    if "asr transcript is empty" in reason.lower():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "asr_transcript_empty",
                "reject_reason": reason,
                "threshold_profile_version": str(service.config.threshold_profile_version),
            },
        ) from exc

    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=reason) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=reason) from exc
    raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.on_event("startup")
def startup_event() -> None:
    global startup_error
    try:
        service.load()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)


@app.get("/")
def root() -> dict:
    return {
        "service": "w2v-gop",
        "endpoints": ["/score", "/score/conversation/summary", "/health"],
        "status": "ready" if service.ready and startup_error is None else "not_ready",
    }


@app.get("/health")
def health() -> dict:
    payload = service.health()
    if startup_error is not None:
        payload["startup_error"] = startup_error
    return payload


@app.post("/score")
async def score_endpoint(
    text: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=f"Service not ready: {startup_error}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio file is empty")

    try:
        return service.score(text=text, audio_bytes=audio_bytes, filename=audio.filename or "upload")
    except Exception as exc:
        _raise_request_error(exc)


@app.post("/score/conversation/summary")
async def score_conversation_summary_endpoint(
    request: Request,
) -> dict:
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=f"Service not ready: {startup_error}")

    form = await request.form()

    include_turn_details_raw = str(form.get("include_turn_details", "true")).strip().lower()
    include_turn_details = include_turn_details_raw in {"1", "true", "yes", "on"}

    texts_raw = [str(x) for x in (form.getlist("texts") or form.getlist("texts[]"))]
    texts_json = form.get("texts_json")

    parsed_texts: List[str] = texts_raw
    if (len(parsed_texts) == 0) and (texts_json is not None):
        try:
            arr = json.loads(str(texts_json))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"texts_json must be valid JSON array: {exc}") from exc
        if not isinstance(arr, list):
            raise HTTPException(status_code=400, detail="texts_json must be a JSON array")
        parsed_texts = [str(t) for t in arr]

    parsed_audios = form.getlist("audios") or form.getlist("audios[]")
    parsed_hr_prompts = [str(x) for x in (form.getlist("hr_prompts") or form.getlist("hr_prompts[]"))]

    if len(parsed_texts) == 0:
        raise HTTPException(status_code=400, detail="texts must contain at least one item")
    if len(parsed_audios) == 0:
        raise HTTPException(status_code=400, detail="audios must contain at least one file")
    if len(parsed_texts) != len(parsed_audios):
        raise HTTPException(
            status_code=400,
            detail=f"texts/audios length mismatch: texts={len(parsed_texts)}, audios={len(parsed_audios)}",
        )
    if (len(parsed_hr_prompts) > 0) and (len(parsed_hr_prompts) != len(parsed_texts)):
        raise HTTPException(
            status_code=400,
            detail=f"hr_prompts length mismatch: hr_prompts={len(parsed_hr_prompts)}, turns={len(parsed_texts)}",
        )

    turns = []
    for i, up in enumerate(parsed_audios):
        if not hasattr(up, "read"):
            raise HTTPException(status_code=400, detail=f"audios[{i}] is not a file upload")
        audio_bytes = await up.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail=f"audio file at turn {i + 1} is empty")
        filename = getattr(up, "filename", None) or f"turn_{i + 1}.wav"
        turns.append(
            {
                "text": parsed_texts[i],
                "audio_bytes": audio_bytes,
                "filename": filename,
                "hr_prompt": parsed_hr_prompts[i] if len(parsed_hr_prompts) == len(parsed_texts) else "",
            }
        )

    try:
        return service.score_conversation_summary(
            turns=turns,
            include_turn_details=include_turn_details,
        )
    except Exception as exc:
        _raise_request_error(exc)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5005"))
    uvicorn.run("app.main:app", host=host, port=port)
