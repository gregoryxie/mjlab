"""Tests for the float16 MLP model."""

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.rl.half_precision import HalfPrecisionMLPModel

NUM_SAMPLES = 8192
OBS_DIM = 48
ACTION_DIM = 12

if torch.cuda.is_available():
  GPU: str | None = "cuda"
elif torch.backends.mps.is_available():
  GPU = "mps"
else:
  GPU = None
needs_gpu = pytest.mark.skipif(
  GPU is None, reason="float16 layers need a CUDA or MPS device"
)


def _model(cls: type[MLPModel], obs: TensorDict) -> MLPModel:
  return cls(
    obs,
    {"actor": ["actor"]},
    "actor",
    ACTION_DIM,
    hidden_dims=(128, 64),
    obs_normalization=True,
  )


def _models(device: str) -> tuple[MLPModel, MLPModel, TensorDict]:
  """A float32 model and a float16 one with the same weights, and a batch for them."""
  torch.manual_seed(0)
  obs = TensorDict(
    {"actor": torch.randn(NUM_SAMPLES, OBS_DIM)}, batch_size=[NUM_SAMPLES]
  ).to(device)
  reference = _model(MLPModel, obs).to(device)
  half = _model(HalfPrecisionMLPModel, obs).to(device)
  half.load_state_dict(reference.state_dict())
  return reference, half, obs


def _gradients(model: MLPModel, obs: TensorDict, target: torch.Tensor) -> torch.Tensor:
  """Weight gradients of a mean-over-samples loss, as PPO's losses are."""
  loss = ((model(obs) - target) ** 2).mean()
  gradients = torch.autograd.grad(loss, list(model.parameters()))
  return torch.cat([gradient.flatten() for gradient in gradients])


def test_same_parameters_as_mlp_model():
  """Checkpoints and exported policies are interchangeable with MLPModel's."""
  reference, half, _ = _models("cpu")
  assert list(half.state_dict()) == list(reference.state_dict())


def test_float32_on_cpu():
  reference, half, obs = _models("cpu")
  half.train()
  assert torch.equal(half(obs), reference(obs))


@needs_gpu
def test_float32_in_eval_mode():
  assert GPU is not None
  reference, half, obs = _models(GPU)
  half.eval()
  assert torch.equal(half(obs), reference(obs))


@needs_gpu
def test_training_output_close_to_float32():
  assert GPU is not None
  reference, half, obs = _models(GPU)
  half.train()
  expected, actual = reference(obs), half(obs)
  assert actual.dtype == torch.float32
  assert (actual - expected).abs().max() < 2e-3 * expected.abs().max()


@needs_gpu
def test_training_gradients_match_float32():
  """Float16 gradients follow float32's with loss scaling, and less well without it."""
  assert GPU is not None
  reference, half, obs = _models(GPU)
  assert isinstance(half, HalfPrecisionMLPModel)
  half.train()
  # Per-sample gradients as small as PPO's: residuals of 1e-2 over 8192 samples.
  noise = 1e-2 * torch.randn(NUM_SAMPLES, ACTION_DIM, device=GPU)
  target = (reference(obs) + noise).detach()
  expected = _gradients(reference, obs, target)

  def cosine(gradients: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(gradients, expected, dim=0).item()

  scaled = _gradients(half, obs, target)
  assert cosine(scaled) > 0.99
  assert 0.95 < scaled.norm() / expected.norm() < 1.05

  half.grad_scale = 1.0
  assert cosine(_gradients(half, obs, target)) < 0.97, "float16 underflow went unseen"
