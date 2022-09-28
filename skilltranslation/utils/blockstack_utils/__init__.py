from .env_tools import (
    register_gym_env,
    get_actor_state,
    get_articulation_state,
    set_actor_state,
    set_articulation_state
)
from .sampler import UniformSampler
from .agents import BaseAgent, AgentConfig
from pathlib import Path

import numpy as np



# FLOATING_HAND_CONFIG=AgentConfig(
#     name='floating_panda_hand',
#     urdf_file="robot/fpanda/panda_hand_2.urdf",
#     fix_root_link=True,
#     magic_control=False,
#     default_init_qpos=[0, 0, 0.2, np.pi, 0, np.pi / 2, 0, 0]
# )

# FLOATING_HAND_CONFIG=AgentConfig(
#     name='floating_panda_hand',
#     urdf_file="robot/fpanda/panda_hand_2.urdf",
#     fix_root_link=True,
#     magic_control=False,
#     default_init_qpos=[0, 0, 0.2, np.pi, 0, np.pi / 2, 0.04, 0.04]
# )

FLOATING_HAND_CONFIG=AgentConfig(
    name='floating_panda_hand',
    urdf_file="robot/fpanda/panda_hand_2.urdf",
    fix_root_link=True,
    magic_control=False,
    default_init_qpos=[0, 0, 0.2, 0.04, 0.04]
)

MAGIC_HAND_CONFIG=AgentConfig(
    name='magic_panda_hand',
    urdf_file="robot/fpanda/panda_hand_2.urdf",
    fix_root_link=True,
    magic_control=True,
    default_init_qpos=[0, 0, 0.2, np.pi, 0, np.pi / 2, 0, 0]
)