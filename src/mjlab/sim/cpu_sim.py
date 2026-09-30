"""Simulating part of the batch with MuJoCo (C) on the CPU, beside MJWarp on the GPU.

The last ``CpuSimCfg.num_envs`` environments of a batch are stepped by MuJoCo's C
library across CPU threads (``mujoco.rollout``), while MJWarp steps the rest on
the GPU. Both halves read and write the same MJWarp ``Data`` arrays, which stay
the single source of truth: before each physics step the CPU rows' state,
controls and applied forces are copied into MuJoCo, and afterwards their state and
sensor readings are copied back. Everything mjlab writes through torch (resets,
pushes, actions) therefore reaches both halves, and the forward and sensing that
follow the physics substeps run on the GPU for every row.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import mujoco.rollout
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp


@dataclass
class CpuSimCfg:
  num_envs: int
  """How many environments, taken from the end of the batch, MuJoCo (C) steps on
  the CPU. The rest stay on the GPU; all of them may go to the CPU."""
  threads: int = 8
  """CPU threads stepping them. Keep below the core count, so torch, Python and the
  GPU driver keep cores of their own."""


# MuJoCo state components, in the order mj_getState lays them out.
_STATE_FIELDS = (
  (mujoco.mjtState.mjSTATE_TIME, "time"),
  (mujoco.mjtState.mjSTATE_QPOS, "qpos"),
  (mujoco.mjtState.mjSTATE_QVEL, "qvel"),
  (mujoco.mjtState.mjSTATE_ACT, "act"),
  (mujoco.mjtState.mjSTATE_HISTORY, "history"),
)
_CONTROL_FIELDS = (
  (mujoco.mjtState.mjSTATE_CTRL, "ctrl"),
  (mujoco.mjtState.mjSTATE_QFRC_APPLIED, "qfrc_applied"),
  (mujoco.mjtState.mjSTATE_XFRC_APPLIED, "xfrc_applied"),
  (mujoco.mjtState.mjSTATE_EQ_ACTIVE, "eq_active"),
  (mujoco.mjtState.mjSTATE_MOCAP_POS, "mocap_pos"),
  (mujoco.mjtState.mjSTATE_MOCAP_QUAT, "mocap_quat"),
  (mujoco.mjtState.mjSTATE_USERDATA, "userdata"),
)


def lightweight_model(spec: mujoco.MjSpec, configure) -> mujoco.MjModel:
  """Compiles ``spec`` with every visual-only mesh replaced by a tiny sphere.

  A mesh is visual-only when every geom using it collides with nothing. Such
  geoms carry no mass in the models this is for, so the physics is unchanged, while
  the model shrinks from tens of megabytes (a robot's visual meshes) to under one,
  which is what makes one model per CPU environment affordable. Geom, body and
  sensor ids are unchanged. ``configure`` applies the options the full model got.
  """
  spec = spec.copy()
  mesh_geoms = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
  colliding = {g.meshname for g in mesh_geoms if g.contype or g.conaffinity}
  visual = {g.meshname for g in mesh_geoms} - colliding
  for mesh in [m for m in spec.meshes if m.name in visual]:
    name = mesh.name
    spec.delete(mesh)  # a copied mesh keeps its loaded file data internally
    sphere = spec.add_mesh(name=name)
    sphere.make_sphere(subdivision=0)
    sphere.scale[:] = [0.01, 0.01, 0.01]
  model = spec.compile()
  configure(model)
  return model


class CpuSim:
  """Steps rows ``[start, nworld)`` of an MJWarp ``Data`` with MuJoCo (C)."""

  def __init__(
    self,
    cfg: CpuSimCfg,
    model: mujoco.MjModel,
    wp_model: mjwarp.Model,
    wp_data: mjwarp.Data,
    expanded_fields: set[str],
  ) -> None:
    self.cfg = cfg
    self.num_envs = cfg.num_envs
    self.start = wp_data.nworld - cfg.num_envs
    self._wp_model = wp_model
    self._expanded_fields = expanded_fields
    self._timestep = model.opt.timestep
    unsupported = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PLUGIN)
    if unsupported:
      raise ValueError("CPU simulation does not support plugin state")

    self.models = [copy.copy(model) for _ in range(self.num_envs)]
    self._datas = [mujoco.MjData(model) for _ in range(cfg.threads)]
    self._pool = mujoco.rollout.Rollout(nthread=cfg.threads)

    rows = slice(self.start, wp_data.nworld)
    self._state_spec, self._state_parts = self._layout(
      model, wp_data, _STATE_FIELDS, rows
    )
    self._control_spec, self._control_parts = self._layout(
      model, wp_data, _CONTROL_FIELDS, rows
    )
    n = self.num_envs
    self._state = np.zeros((n, mujoco.mj_stateSize(model, self._state_spec)))
    self._control = np.zeros((n, 1, mujoco.mj_stateSize(model, self._control_spec)))
    self._warmstart = np.zeros((n, model.nv))
    self._state_out = np.zeros((n, 1, self._state.shape[1]))
    self._sensor_out = np.zeros((n, 1, model.nsensordata))
    self._qvel = wp.to_torch(wp_data.qvel)[rows]
    self._qacc_warmstart = wp.to_torch(wp_data.qacc_warmstart)[rows]
    self._sensordata = wp.to_torch(wp_data.sensordata)[rows]
    qvel_slice = next(s for name, _, s in self._state_parts if name == "qvel")
    self._qvel_cols = qvel_slice
    self._synced: dict[str, torch.Tensor] = {}
    self._model_rows: dict[str, tuple[wp.array, torch.Tensor]] = {}

  @staticmethod
  def _layout(model, wp_data, fields, rows):
    """The state spec and (name, torch rows, column slice) of each nonempty part."""
    spec, parts, col = 0, [], 0
    for bit, name in fields:
      size = mujoco.mj_stateSize(model, bit)
      if size == 0:
        continue
      spec |= int(bit)
      tensor = wp.to_torch(getattr(wp_data, name))[rows]
      parts.append((name, tensor.reshape(tensor.shape[0], -1), slice(col, col + size)))
      col += size
    return spec, parts

  def sync_models(self) -> None:
    """Copies randomized model fields of the CPU rows into their MjModels.

    Domain randomization writes per-world model arrays through torch; only the
    environments whose values changed since the last call are updated.
    """
    for name in self._expanded_fields:
      rows = self._rows_of(name)
      if rows.shape[0] != self.num_envs:
        continue  # not per-world for these rows (a shared array)
      previous = self._synced.get(name)
      if previous is not None and previous.shape == rows.shape:
        changed = (rows != previous).reshape(self.num_envs, -1).any(dim=1).nonzero()
        env_ids = changed.flatten().tolist()
      else:
        env_ids = range(self.num_envs)
      values = rows.cpu().numpy()
      for i in env_ids:
        getattr(self.models[i], name)[...] = values[i]
      self._synced[name] = rows.clone()

  def _rows_of(self, name: str) -> torch.Tensor:
    """The CPU rows of a model array, as a torch view.

    Views are cached because creating one waits for the GPU on Metal, which would
    serialize the two halves; a new view is made only when mjlab replaces the
    array (``expand_model_fields`` does).
    """
    array = getattr(self._wp_model, name)
    cached = self._model_rows.get(name)
    if cached is None or cached[0] is not array:
      cached = (array, wp.to_torch(array)[self.start :])
      self._model_rows[name] = cached
    return cached[1]

  def step(self) -> None:
    """One physics step of every CPU row."""
    self.sync_models()
    for _, tensor, cols in self._state_parts:
      self._state[:, cols] = tensor.cpu().numpy()
    for _, tensor, cols in self._control_parts:
      self._control[:, 0, cols] = tensor.cpu().numpy()
    self._warmstart[:] = self._qacc_warmstart.cpu().numpy()

    self._pool.rollout(
      self.models,
      self._datas,
      self._state,
      self._control,
      control_spec=self._control_spec,
      nstep=1,
      initial_warmstart=self._warmstart,
      state=self._state_out,
      sensordata=self._sensor_out,
      skip_checks=True,
    )

    state = self._state_out[:, 0]
    for _, tensor, cols in self._state_parts:
      tensor.copy_(torch.from_numpy(state[:, cols]))
    # rollout returns no warm start; the step's velocity change over the timestep
    # (its acceleration, exactly so for the Euler integrator) seeds the next solve.
    qvel_before = self._state[:, self._qvel_cols]
    qvel_after = state[:, self._qvel_cols]
    self._qacc_warmstart.copy_(
      torch.from_numpy((qvel_after - qvel_before) / self._timestep)
    )
    self._sensordata.copy_(torch.from_numpy(self._sensor_out[:, 0]))

  def close(self) -> None:
    self._pool.close()
