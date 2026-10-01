"""The robust flat G1 velocity task: its randomization terms and command delay."""

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

FLAT = "Mjlab-Velocity-Flat-Unitree-G1"
ROBUST = "Mjlab-Velocity-Flat-Robust-Unitree-G1"


def test_adds_randomization_terms_before_base_com():
  flat, robust = load_env_cfg(FLAT), load_env_cfg(ROBUST)
  added = ["link_inertia", "actuator_gains", "actuator_strength"]
  assert [n for n in robust.events if n not in flat.events] == added
  # link_inertia rewrites every body's COM from its default, so base_com must follow it.
  names = list(robust.events)
  assert names.index("link_inertia") < names.index("base_com")
  assert all(robust.events[n].mode == "startup" for n in added)


def test_delays_every_actuator_and_leaves_flat_task_alone():
  flat, robust = load_env_cfg(FLAT), load_env_cfg(ROBUST)
  flat_actuators = flat.scene.entities["robot"].articulation.actuators
  robust_actuators = robust.scene.entities["robot"].articulation.actuators
  assert len(robust_actuators) == len(flat_actuators)
  assert all(a.delay_max_lag == 3 and a.delay_min_lag == 0 for a in robust_actuators)
  assert all(a.delay_max_lag == 0 for a in flat_actuators)


def test_play_config_keeps_randomization():
  play = load_env_cfg(ROBUST, play=True)
  assert "link_inertia" in play.events and "push_robot" not in play.events
