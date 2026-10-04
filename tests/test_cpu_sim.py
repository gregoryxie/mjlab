"""Tests for stepping part of the batch with MuJoCo (C) on the CPU."""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.sim import Simulation, SimulationCfg
from mjlab.sim.cpu_sim import CpuSimCfg

_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.3">
      <freejoint/>
      <geom name="torso" type="box" size="0.1 0.1 0.05" mass="1"/>
      <site name="imu"/>
      <body name="leg" pos="0 0 -0.05">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="shin" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.2"/>
        <site name="foot" pos="0 0 -0.2" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip" joint="hip" kp="20"/>
  </actuator>
  <sensor>
    <framepos objtype="body" objname="base"/>
    <velocimeter site="imu"/>
    <touch site="foot"/>
  </sensor>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _sim(device, num_envs, num_cpu, **cfg):
  model = mujoco.MjModel.from_xml_string(_XML)
  sim_cfg = SimulationCfg(cpu_sim=CpuSimCfg(num_envs=num_cpu, threads=2), **cfg)
  return Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)


def _ctrl(step, num_envs):
  return 0.8 * np.sin(0.05 * step + np.arange(num_envs))[:, None]


def test_cpu_rows_reproduce_mujoco(device):
  """CPU rows step exactly as MuJoCo (C) does, given the batch's float32 state."""
  num_envs, num_cpu, steps = 4, 2, 100
  sim = _sim(device, num_envs, num_cpu)
  model = sim.mj_model  # with SimulationCfg's options applied, as the CPU rows use
  datas = [mujoco.MjData(model) for _ in range(num_cpu)]
  f32 = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)  # noqa: E731

  for step in range(steps):
    ctrl = _ctrl(step, num_envs)
    sim.data.ctrl[:] = torch.as_tensor(ctrl, dtype=torch.float32)
    sim.step()
    for i, d in enumerate(datas):
      row = num_envs - num_cpu + i
      qvel_before = f32(d.qvel)
      d.qpos[:], d.qvel[:], d.time = f32(d.qpos), qvel_before, f32(d.time)
      d.qacc_warmstart[:] = f32(d.qacc_warmstart)
      d.ctrl[:] = f32(ctrl[row])
      mujoco.mj_step(model, d)
      d.qacc_warmstart[:] = (d.qvel - qvel_before) / model.opt.timestep

  for i, d in enumerate(datas):
    row = num_envs - num_cpu + i
    np.testing.assert_allclose(sim.data.qpos[row].cpu().numpy(), d.qpos, atol=1e-5)
    np.testing.assert_allclose(
      sim.data.sensordata[row].cpu().numpy(), d.sensordata, atol=1e-4
    )


def test_cpu_rows_track_gpu_rows(device):
  """The same environment stepped on the GPU and on the CPU stays together."""
  sim = _sim(device, num_envs=2, num_cpu=1)
  for step in range(60):
    sim.data.ctrl[:] = torch.as_tensor(np.repeat(_ctrl(step, 1), 2, axis=0))
    sim.step()
  gpu, cpu = sim.data.qpos[0].cpu().numpy(), sim.data.qpos[1].cpu().numpy()
  np.testing.assert_allclose(cpu, gpu, atol=1e-3)
  assert sim.data.sensordata[1, -1] > 0  # the foot rests on the floor: touch reads


def test_applied_force_and_reset_reach_cpu_rows(device):
  sim = _sim(device, num_envs=2, num_cpu=1)
  base = 1  # body id of "base"
  sim.data.xfrc_applied[1, base, 2] = 100.0  # lift the CPU row's base
  for _ in range(20):
    sim.step()
  assert sim.data.qpos[1, 2] > sim.data.qpos[0, 2] + 0.05

  sim.data.xfrc_applied[:] = 0.0
  sim.reset()
  sim.step()  # the CPU row must continue from the reset state, not its old one
  np.testing.assert_allclose(
    sim.data.qpos[1].cpu().numpy(), sim.mj_model.qpos0, atol=1e-3
  )


def test_randomized_model_field_reaches_cpu_model(device):
  sim = _sim(device, num_envs=3, num_cpu=2)
  sim.expand_model_fields(("geom_friction",))
  sim.model.geom_friction[2, :, 0] = 0.123
  sim.step()
  cpu_sim = sim._cpu_sim
  assert cpu_sim is not None
  np.testing.assert_allclose(cpu_sim.models[1].geom_friction[:, 0], 0.123, rtol=1e-6)
  np.testing.assert_allclose(cpu_sim.models[0].geom_friction[:, 0], 1.0)


def test_all_envs_on_cpu(device):
  sim = _sim(device, num_envs=2, num_cpu=2)
  for _ in range(10):
    sim.step()
  sim.forward()
  assert torch.isfinite(sim.data.qpos).all()
  assert sim.data.qpos[:, 2].max() < 0.3  # fell toward the floor


def test_off_by_default(device):
  model = mujoco.MjModel.from_xml_string(_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  assert sim._cpu_sim is None and sim._step_data is sim.wp_data


@pytest.mark.parametrize("num_cpu", [-1, 3])
def test_rejects_invalid_num_envs(device, num_cpu):
  with pytest.raises(ValueError, match="cpu_sim.num_envs"):
    _sim(device, num_envs=2, num_cpu=num_cpu)
