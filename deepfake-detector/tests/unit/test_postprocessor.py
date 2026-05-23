"""Unit tests for core.postprocessor."""

import torch

from core.postprocessor import logits_to_prediction


def test_fake_wins_when_higher_logit():
    logits = torch.tensor([[0.0, 5.0]])
    label, conf = logits_to_prediction(logits)
    assert label == "Fake"
    assert conf > 50.0


def test_real_wins_when_higher_logit():
    logits = torch.tensor([[5.0, 0.0]])
    label, conf = logits_to_prediction(logits)
    assert label == "Real"
    assert conf > 50.0
