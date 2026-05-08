"""
Fabric AI — FastAPI Backend

Endpoints:
  GET  /api/status                  → current enhancement methods + keys configured
  POST /api/set-token               → set Replicate token at runtime
  POST /api/set-anthropic-key       → set Anthropic key at runtime
  POST /api/enhance                 → Step 1: upload raw image → enhanced image
  POST /api/detect-swatches         → Step 2: detect fabric swatches via Claude Vision
  POST /api/compose                 → Step 3: compose final branded image
  GET  /api/image/{kind}/{filename} → serve images
  GET  /                            → serve frontend
"""

import os
import uuid
import logging
import shutil
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enhancer import FabricImageEnhancer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Directories ────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
UPLOADS_DIR  = BASE_DIR / "uploads"
ENHANCED_DIR = BASE_DIR / "enhanced"
COMPOSED_DIR = BASE_DIR / "composed"
FRONTEND_DIR = BASE_DIR / "frontend"

for d in [UPLOADS_DIR, ENHANCED_DIR, COMPOSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 50

ENV_FILE = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass


def save_env_key(key: str, value: str):
    """Persist a key=value pair to .env so it survives server restarts."""
    lines = []
    found = False
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {key} to {ENV_FILE}")

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="Fabric AI", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── State (runtime) ────────────────────────────────────────────
enhancer = FabricImageEnhancer()
anthropic_key: str = os.environ.get("ANTHROPIC_API_KEY", "")


# ── Helpers ────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    idx = FRONTEND_DIR / "index.html"
    return FileResponse(str(idx)) if idx.exists() else JSONResponse({"status": "API running"})


@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "sr_method": enhancer.sr_method,
        "replicate_enabled": enhancer.replicate_available,
        "anthropic_enabled": bool(anthropic_key),
    }


# ── Token endpoints ────────────────────────────────────────────
class TokenPayload(BaseModel):
    token: str

@app.post("/api/set-token")
def set_replicate_token(payload: TokenPayload):
    global enhancer
    token = payload.token.strip()
    if not token:
        raise HTTPException(400, "Token cannot be empty")
    save_env_key("REPLICATE_API_TOKEN", token)
    enhancer = FabricImageEnhancer(replicate_token=token)
    return {"ok": True, "sr_method": enhancer.sr_method,
            "replicate_enabled": enhancer.replicate_available}


@app.post("/api/set-anthropic-key")
def set_anthropic_key(payload: TokenPayload):
    global anthropic_key
    key = payload.token.strip()
    if not key:
        raise HTTPException(400, "Key cannot be empty")
    anthropic_key = key
    save_env_key("ANTHROPIC_API_KEY", key)
    logger.info("Anthropic API key saved to .env")
    return {"ok": True, "anthropic_enabled": True}


# ── Step 1: Enhance ────────────────────────────────────────────
@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'")

    job_id        = str(uuid.uuid4())[:8]
    safe_name     = f"{job_id}_raw{ext}"
    raw_path      = UPLOADS_DIR  / safe_name
    enhanced_name = f"{job_id}_enhanced.jpg"
    enhanced_path = ENHANCED_DIR / enhanced_name

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = raw_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raw_path.unlink()
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB)")

    logger.info(f"[{job_id}] Enhancing {file.filename} ({size_mb:.2f} MB)")
    try:
        stats = enhancer.enhance(str(raw_path), str(enhanced_path))
    except Exception as e:
        logger.exception(f"[{job_id}] Enhancement failed")
        raise HTTPException(500, f"Enhancement failed: {e}")

    return JSONResponse({
        "job_id":            job_id,
        "raw_url":           f"/api/image/raw/{safe_name}",
        "enhanced_url":      f"/api/image/enhanced/{enhanced_name}",
        "enhanced_filename": enhanced_name,
        "stats": {
            **stats,
            "raw_size_kb":      round(raw_path.stat().st_size / 1024),
            "enhanced_size_kb": round(enhanced_path.stat().st_size / 1024),
        },
    })


# ── Step 2: Detect swatches ────────────────────────────────────
class DetectPayload(BaseModel):
    enhanced_filename: str
    swatch_count: int = 0   # if > 0 skip Claude and generate even splits

@app.post("/api/detect-swatches")
def detect_swatches(payload: DetectPayload):
    enhanced_path = ENHANCED_DIR / payload.enhanced_filename
    if not enhanced_path.exists():
        raise HTTPException(404, "Enhanced image not found")

    # Manual split fallback (no API key or user override)
    if payload.swatch_count > 0 or not anthropic_key:
        count = payload.swatch_count or 4
        step  = 100 / count
        swatches = [
            {"index": i, "y_percent": round(step * i + step / 2, 1),
             "color_description": f"Color {i+1}"}
            for i in range(count)
        ]
        return JSONResponse({"swatches": swatches, "method": "manual"})

    try:
        from swatch_detector import detect_swatches as _detect
        swatches = _detect(str(enhanced_path), anthropic_key)
        if not swatches:
            raise ValueError("No swatches detected")
        return JSONResponse({"swatches": swatches, "method": "claude-vision"})
    except Exception as e:
        logger.warning(f"Swatch detection failed: {e} — using even split")
        count = 4
        step  = 100 / count
        swatches = [
            {"index": i, "y_percent": round(step * i + step / 2, 1),
             "color_description": f"Color {i+1}"}
            for i in range(count)
        ]
        return JSONResponse({"swatches": swatches, "method": "fallback",
                             "warning": str(e)})


# ── Step 3: Compose ────────────────────────────────────────────
class ComposePayload(BaseModel):
    enhanced_filename: str
    product_code: str = ""
    gsm: str = ""
    width: str = ""
    colors: list = []
    swatches: list = []
    logo_base64: str = ""

@app.post("/api/compose")
def compose_image(payload: ComposePayload):
    enhanced_path = ENHANCED_DIR / payload.enhanced_filename
    if not enhanced_path.exists():
        raise HTTPException(404, "Enhanced image not found")

    composed_name = payload.enhanced_filename.replace("_enhanced.jpg", "_composed.jpg")
    composed_path = COMPOSED_DIR / composed_name

    try:
        from composer import compose
        compose(
            enhanced_path=str(enhanced_path),
            output_path=str(composed_path),
            product_code=payload.product_code,
            gsm=payload.gsm,
            width=payload.width,
            colors=payload.colors,
            swatches=payload.swatches,
            logo_base64=payload.logo_base64,
        )
    except Exception as e:
        logger.exception("Composition failed")
        raise HTTPException(500, f"Composition failed: {e}")

    return JSONResponse({
        "composed_url":      f"/api/image/composed/{composed_name}",
        "composed_filename": composed_name,
    })


# ── Image serving ──────────────────────────────────────────────
@app.get("/api/image/raw/{filename}")
def get_raw(filename: str):
    p = UPLOADS_DIR / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p))

@app.get("/api/image/enhanced/{filename}")
def get_enhanced(filename: str):
    p = ENHANCED_DIR / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p))

@app.get("/api/image/composed/{filename}")
def get_composed(filename: str):
    p = COMPOSED_DIR / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
