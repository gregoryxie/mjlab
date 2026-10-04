"""Maps the torch device mjlab runs on to the Warp device that simulates."""

import os

import warp as wp


def sim_device(device: str) -> str:
  """Return the Warp device that simulates for a torch device string.

  CUDA devices simulate on themselves. On Apple Silicon the Metal GPU simulates for
  ``"cpu"``: its memory is unified, so torch keeps aliasing the Warp arrays as CPU
  tensors. Set ``MJLAB_SIM_DEVICE`` (e.g. ``cpu``) to override.
  """
  override = os.environ.get("MJLAB_SIM_DEVICE")
  if override:
    return override
  if device == "cpu" and getattr(wp, "is_metal_available", lambda: False)():
    return "metal:0"
  return device


def synchronize(device: wp.Device) -> None:
  """Wait for a Metal device so CPU tensors aliasing its arrays read finished results."""
  if getattr(device, "is_metal", False):
    wp.synchronize_device(device)
