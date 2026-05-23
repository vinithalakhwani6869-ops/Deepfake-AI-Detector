# Deepfake Detector

Production FastAPI + PyTorch deepfake image detection API.

## Architecture

| Package | Responsibility |
|---------|----------------|
| `api/` | FastAPI app, routers, request/response schemas |
| `core/` | Inference detector, preprocessor, postprocessor, model registry |
| `models/` | Neural network architectures |
| `data/` | Datasets, transforms (training) |
| `training/` | Training pipeline (not imported by API) |
| `evaluation/` | Offline metrics and benchmarks |
| `configs/` | YAML configuration |
| `scripts/` | CLI entry points |

## Quick start

```bash
cd deepfake-detector
pip install -r requirements.txt

# Place weights (optional — falls back to ImageNet B0)
# cp your_weights.pth checkpoints/deepfake_model.pth

uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

### Legacy entry point

The original `deepfake-api/` folder still works:

```bash
cd deepfake-api
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## API (unchanged)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health ping |
| GET | `/health` | Model + device status |
| POST | `/detect` | Upload image → `{ result, confidence, filename }` |

## Tests

```bash
cd deepfake-detector
pip install -r requirements-dev.txt
pytest tests/unit -q
pytest tests/integration -q -m "not slow"
```

## Checkpoints

See [checkpoints/README.md](checkpoints/README.md).
