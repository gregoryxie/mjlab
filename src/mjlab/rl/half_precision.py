"""An MLP actor or critic whose layers run in float16 while it trains on a GPU.

Select it in place of rsl_rl's ``MLPModel``::

  --agent.actor.class-name mjlab.rl.half_precision:HalfPrecisionMLPModel
  --agent.critic.class-name mjlab.rl.half_precision:HalfPrecisionMLPModel

The weights, the optimizer, the observation normalizer, the action distribution and
the losses stay in float32, and so do checkpoints and exported policies: the model
has ``MLPModel``'s parameters under the same names. Only the layers' arithmetic,
forward and backward, is float16, and only while the model is in training mode on a
CUDA or MPS device.

Float16 holds nothing below 6e-8 and loses precision below 6e-5. PPO's loss is a mean
over tens of thousands of samples, so the gradient of one sample's output starts near
that. The gradient entering the float16 layers is therefore multiplied by
``grad_scale``, and each weight's gradient is divided by it once it is back in
float32. That is loss scaling, done inside the model so the algorithm is unchanged.
"""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


class _ScaleGradient(torch.autograd.Function):
  """Identity in the forward pass; multiplies the gradient by a constant."""

  @staticmethod
  def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
    ctx.scale = scale
    return x.view_as(x)

  @staticmethod
  def backward(ctx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
    return grad_outputs[0] * ctx.scale, None


class HalfPrecisionMLPModel(MLPModel):
  """``MLPModel`` with its MLP in float16 during training on CUDA or MPS."""

  grad_scale: float = 1024.0
  """Factor on the gradient inside the float16 layers. A power of two, so scaling and
  unscaling are exact. Large enough to lift per-sample gradients of about 1e-5 clear
  of float16's underflow, small enough that a weight gradient of 60 does not
  overflow."""

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    # MLPModel.forward (rsl-rl-lib 5.5), with the MLP call replaced.
    if masks is not None and not self.is_recurrent:
      obs = unpad_trajectories(obs, masks)  # type: ignore[assignment]
    latent = self.get_latent(obs, masks, hidden_state)
    if self.training and latent.device.type in ("cuda", "mps"):
      mlp_output = self._half_precision_mlp(latent)
    else:
      mlp_output = self.mlp(latent)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def _half_precision_mlp(self, latent: torch.Tensor) -> torch.Tensor:
    """The MLP's output in float32, computed in float16."""
    device_type = latent.device.type
    if not torch.is_grad_enabled():
      with torch.autocast(device_type, dtype=torch.float16):
        return self.mlp(latent).float()
    scale = self.grad_scale
    # The weights' gradients leave the float16 layers scaled; this undoes it.
    weights = {
      name: _ScaleGradient.apply(weight, 1.0 / scale)
      for name, weight in self.mlp.named_parameters()
    }
    with torch.autocast(device_type, dtype=torch.float16):
      output = torch.func.functional_call(self.mlp, weights, (latent,))
    return _ScaleGradient.apply(output.float(), scale)
