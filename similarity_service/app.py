from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .audio import AudioDecodingError, AudioTooLongError, PathAccessError
from .service import SimilarityService


class ScoreRolloutsRequest(BaseModel):
    request_id: str | None = None
    ref_audio_path: str
    generated_audio_paths: list[str]


class ScoreRolloutsResponse(BaseModel):
    request_id: str | None
    backend: str
    ref_cache_hit: bool
    ref_bucket_seconds: int
    generated_bucket_seconds: list[int]
    scores: list[float]


def create_app(service: SimilarityService) -> FastAPI:
    app = FastAPI(title="WavLM Large + ECAPA TensorRT Similarity Service")
    app.state.similarity_service = service

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "backend": service.backend.name,
        }

    @app.get("/readyz")
    async def readyz():
        if not service.is_ready:
            raise HTTPException(status_code=503, detail="backend is not ready")
        return {
            "status": "ready",
            "backend": service.backend.name,
        }

    @app.post("/score_rollouts", response_model=ScoreRolloutsResponse)
    async def score_rollouts(request: ScoreRolloutsRequest):
        if not request.generated_audio_paths:
            raise HTTPException(status_code=400, detail="generated_audio_paths must not be empty")
        if len(request.generated_audio_paths) > 16:
            raise HTTPException(status_code=400, detail="generated_audio_paths must contain at most 16 paths")
        try:
            result = service.score_rollouts(
                ref_audio_path=request.ref_audio_path,
                generated_audio_paths=request.generated_audio_paths,
                request_id=request.request_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PathAccessError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (AudioDecodingError, AudioTooLongError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ScoreRolloutsResponse(
            request_id=result.request_id,
            backend=result.backend,
            ref_cache_hit=result.ref_cache_hit,
            ref_bucket_seconds=result.ref_bucket_seconds,
            generated_bucket_seconds=result.generated_bucket_seconds,
            scores=result.scores,
        )

    return app
