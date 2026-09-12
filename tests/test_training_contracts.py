from __future__ import annotations

import math

import torch

from clip.clip_model import CLIP
from paper_repro.train import (
    CleanCLIP,
    VAETrainingModel,
    _clip_portable_transform,
    _condition_tensors,
)
from unet.unet_conditional import ECGconditional
from vae.vae_model import loss_function


def test_vae_model_and_released_loss_contract() -> None:
    torch.manual_seed(4)
    model = VAETrainingModel().eval()
    waveform = torch.randn(2, 1024, 12)
    with torch.no_grad():
        reconstruction, mean, log_variance = model(waveform)
        losses = loss_function(reconstruction, waveform, mean, log_variance, 0.1)
    assert reconstruction.shape == waveform.shape
    assert mean.shape == (2, 4, 128)
    assert log_variance.shape == mean.shape
    assert set(losses) == {"loss", "mse", "KLD"}
    assert torch.isfinite(losses["loss"])
    assert torch.allclose(losses["loss"], losses["mse"] + 0.1 * losses["KLD"])


def test_clean_clip_keeps_released_parameter_schema_and_score() -> None:
    released = CLIP(embed_dim=64)
    clean = CleanCLIP(embed_dim=64)
    assert released.state_dict().keys() == clean.state_dict().keys()
    clean.load_state_dict(released.state_dict())
    assert clean.logit_scale.shape == (1,)
    portable = _clip_portable_transform(
        {f"clip.{key}": value for key, value in clean.state_dict().items()}
    )
    assert portable["logit_scale"].shape == released.logit_scale.shape
    released.load_state_dict(portable, strict=True)
    clean.eval()
    signal = torch.randn(2, 1024, 12)
    text = torch.randn(2, 1536)
    with torch.no_grad():
        logits_signal, logits_text = clean(signal, text)
        scores = clean(signal, text, True)
        scale = clean.logit_scale.exp()
    assert logits_signal.shape == (2, 2)
    assert logits_text.shape == (2, 2)
    assert torch.allclose(scores, logits_signal.diag() / scale, atol=1e-6, rtol=1e-5)


def test_diffusion_condition_shapes_and_paper_unet_forward() -> None:
    batch = {
        "text_embedding": torch.randn(2, 1536),
        "gender": torch.tensor([0.0, 1.0]),
        "age": torch.tensor([45.0, 70.0]),
        "heart_rate": torch.tensor([60.0, 90.0]),
    }
    text, condition = _condition_tensors(batch, torch.device("cpu"))
    assert text.shape == (2, 1, 1536)
    assert all(value.shape == (2, 1, 1) for value in condition.values())
    model = ECGconditional(1000, kernel_size=7, num_levels=7, n_channels=4)
    with torch.no_grad():
        output = model(torch.randn(2, 4, 128), torch.tensor([1, 998]), text, condition)
    assert output.shape == (2, 4, 128)
    assert torch.isfinite(output).all()


def test_released_diffusion_timestep_support_excludes_endpoints() -> None:
    torch.manual_seed(11)
    values = torch.randint(1, 999, (100_000,))
    assert int(values.min()) >= 1
    assert int(values.max()) <= 998
    assert math.isclose(float(values.float().mean()), 499.5, rel_tol=0.01)
