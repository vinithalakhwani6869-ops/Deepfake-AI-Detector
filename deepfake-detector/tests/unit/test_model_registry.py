"""Unit tests for core.model_registry."""

from pathlib import Path

import pytest

from core.model_registry import (
    MODEL_BUILDERS,
    build_model,
    extract_state_dict,
    get_project_root,
    list_available,
    resolve_weights_path,
)


def test_efficientnet_b0_registered():
    assert "efficientnet_b0" in MODEL_BUILDERS
    assert "efficientnet_b0" in list_available()


def test_build_model_returns_module():
    pytest.importorskip("torch")
    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    assert model is not None


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("not_a_real_model")


def test_resolve_weights_path_default_under_checkpoints():
    path = resolve_weights_path()
    assert path.is_absolute()
    assert path.name == "deepfake_model.pth"
    assert path.parent == get_project_root() / "checkpoints"


def test_resolve_weights_path_explicit(tmp_path: Path):
    custom = tmp_path / "custom.pth"
    custom.touch()
    resolved = resolve_weights_path(custom)
    assert resolved == custom.resolve()


def test_extract_state_dict_raw():
    pytest.importorskip("torch")
    import torch

    sd = {"layer.weight": torch.zeros(1)}
    assert extract_state_dict(sd) == sd


def test_extract_state_dict_nested():
    pytest.importorskip("torch")
    import torch

    inner = {"a": torch.ones(1)}
    wrapped = {"model_state_dict": inner}
    assert extract_state_dict(wrapped) == inner
