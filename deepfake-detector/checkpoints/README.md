# Model checkpoints

Place trained weights here. The API loads checkpoints in this order:

1. `deepfake-detector/checkpoints/deepfake_model.pth` (preferred)
2. `deepfake-api/model/deepfake_model.pth` (legacy path)

## Filename convention

| File | Model |
|------|-------|
| `deepfake_model.pth` | EfficientNet-B0 binary classifier |
| `deepfake_model_b4.pth` | EfficientNet-B4 (when implemented) |

Supported checkpoint formats: raw state dict, `model_state_dict`, `state_dict`, Lightning `model.` prefix.
