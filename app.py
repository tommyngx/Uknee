import base64
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app_function import load_model, predict_to_file


REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/content/kneeSeg/RMKV_kneeSeg.pth"))
DEVICE = os.environ.get("DEVICE", "auto")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

API_DIR = Path(os.environ.get("API_DIR", "/content/API"))
INPUT_PATH = API_DIR / "input.png"
OUTPUT_PATH = API_DIR / "output.png"

app = FastAPI(title="Knee Segmentation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNTIME = None


class ImageIn(BaseModel):
    image_b64: str


@app.on_event("startup")
def startup_event():
    global RUNTIME
    API_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME = load_model(
        weight_path=MODEL_PATH,
        repo_root=REPO_ROOT,
        device=DEVICE,
        threshold=THRESHOLD,
    )


@app.get("/")
def root():
    return {"status": "alive"}


def _decode_base64_to_file(b64_str: str, out_path: Path):
    if "," in b64_str and b64_str.strip().lower().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    try:
        data = base64.b64decode(b64_str, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        file.write(data)


def _file_to_b64(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


@app.post("/predict")
def predict(payload: ImageIn):
    global RUNTIME
    if RUNTIME is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    _decode_base64_to_file(payload.image_b64, INPUT_PATH)

    try:
        predict_to_file(
            runtime=RUNTIME,
            image_input=INPUT_PATH,
            output_path=OUTPUT_PATH,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    output_b64 = _file_to_b64(OUTPUT_PATH)
    if output_b64 is None:
        raise HTTPException(status_code=500, detail="Output mask was not created.")

    return {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "mask_b64": output_b64,
    }
