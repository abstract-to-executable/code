import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import attr
import numpy as np
import sapien.core as sapien
import yaml
from gym import spaces

from skilltranslation import ASSET_DIR
from skilltranslation.utils.blockstack_utils.env_tools import get_pairwise_contact_impulse, compute_angle_between



@attr.s(auto_attribs=True, kw_only=True)
class AgentConfig:
    #agent_class: str
    name: str
    urdf_file: str
    fix_root_link: bool = True,
    fix_rotation:bool = True
    magic_control:bool = False
    default_init_qpos: List[float]




class BaseAgent:
    """Base class for agents.

    Agent is an interface of the robot (sapien.Articulation).

    Args:
        config (AgentConfig):  agent configuration.
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).

    """

    _config: AgentConfig
    _scene: sapien.Scene
    _robot: sapien.Articulation
    #_cameras: Dict[str, sapien.CameraEntity]
    _sim_time_step: float

    def __init__(self, config: AgentConfig, scene: sapien.Scene, control_freq: int):
        self._config = copy.deepcopy(config)
        self._scene = scene
        self._sim_time_step = scene.timestep
        self._control_freq = control_freq

        self._initialize_robot()
        # self._initialize_controllers()
        # self._initialize_cameras()
        self.finger1_link:sapien.Link = self._robot.get_links()[-2]
        self.finger2_link:sapien.Link = self._robot.get_links()[-1]



    def set_magic_control(self, magic_control):
        self._config.magic_control = magic_control
    def _initialize_robot(self):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = self._config.fix_root_link
        urdf_file = ASSET_DIR / self._config.urdf_file
        # print(self._config.urdf_file)

        finger_material = self._scene.create_physical_material(
            static_friction=100,
            dynamic_friction=100,
            restitution=0, 
        )
        finger_config = {
            'surface_material': finger_material,
            'patch_radius': 0.1, 
            'min_patch_radius': 0.1,
        }
        urdf_config = {'link': {}}
        urdf_config['link']['panda_leftfinger'] = finger_config.copy()
        urdf_config['link']['panda_rightfinger'] = finger_config.copy()

        self._robot:sapien.Articulation = loader.load(str(urdf_file), urdf_config)
        self._robot.set_name(self._config.name)
        for joint in self._robot.get_active_joints():
            joint.set_drive_property(stiffness=20000, damping=400)
        # for joint in self._robot.get_active_joints():
        #     if 'finger' in joint.get_name():
        #         joint.set_friction(0.1)
        #         joint.set_drive_property(stiffness=0, damping=1)
        #     else:
        #         joint.set_friction(0.1)
        #         joint.set_drive_property(stiffness=0, damping=1000)

    @property
    def action_space(self):
        # TODO: modify action space
        '''
        xyz + rpy + 1dim for ee ->
        :return:
        '''
        l = len(self._robot.get_active_joints()) - 1

        high = np.ones(l) * 0.015
        low = -copy.deepcopy(high)
        return spaces.Box(low, high, (l,))

    # @property
    # def action_range(self) -> spaces.Box:
    #     return spaces.Box(
    #         self._combined_controllers[self._control_mode].action_range[:, 0],
    #         self._combined_controllers[self._control_mode].action_range[:, 1],
    #     )

    def set_action(self, action: np.ndarray):
        assert action.shape == self.action_space.shape
        # dup the last dim to control two fingers with same action
        clipped_action = np.clip(action.copy(), self.action_space.low, self.action_space.high)
        if clipped_action[-1] < 0: # close gripper
            clipped_action[-1] = clipped_action[-1] * 0.1

        _action = np.zeros(self.action_space.shape[0] + 1)
        _action[-2:] = clipped_action[-1], clipped_action[-1]
        _action[:-2] = clipped_action[:-1]

        # adjust if use magic control
        if self._config.magic_control:
            _action[-2:] = np.zeros(2)
        active_joints = self._robot.get_active_joints()
        '''
        actions are delta qpos, and there might be a caveat that delta qpos + cur_qpos != targeted qpos
        (think of grippers' case) 
        '''
        for i in range(len(active_joints)):
            active_joints[i].set_drive_target(_action[i] + self._robot.get_qpos()[i])
            ## signal > 0.04
        # adjust if fix rotation
        # found bugs here
        # if self._config.fix_rotation:
        #     for j in range(3,6):
        #         active_joints[j].set_drive_target(self._config.default_init_qpos[j])


    def simulation_step(self):
        qf=self._robot.compute_passive_force(
            gravity=True,
            coriolis_and_centrifugal=True)
        self._robot.set_qf(qf)

    def reset(self, init_qpos=None):
        if init_qpos is None:
            init_qpos = self._config.default_init_qpos
        self._robot.set_qpos(init_qpos)
        self._robot.set_qvel(np.zeros(self._robot.dof))
        self._robot.set_qacc(np.zeros(self._robot.dof))
        self._robot.set_qf(np.zeros(self._robot.dof))

    @classmethod
    def from_config(cls, config: AgentConfig, scene: sapien.Scene, control_freq: int):
        return cls(config, scene, control_freq)


    def get_proprioception(self) -> Dict:
        state_dict=OrderedDict()
        qpos=self._robot.get_qpos()
        qvel=self._robot.get_qvel()
        state_dict['qpos']=qpos
        state_dict['qvel']=qvel
        return state_dict

    def get_state(self) -> Dict:
        """Get current state for MPC"""
        state = OrderedDict()

        # robot state
        root_link = self._robot.get_links()[0]
        state["robot_root_pose"] = root_link.get_pose()
        state["robot_root_vel"] = root_link.get_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self._robot.get_qpos()
        state["robot_qvel"] = self._robot.get_qvel()
        state["robot_qacc"] = self._robot.get_qacc()

        return state

    def set_state(self, state: Dict):
        # robot state
        self._robot.set_root_pose(state["robot_root_pose"])
        self._robot.set_root_velocity(state["robot_root_vel"])
        self._robot.set_root_angular_velocity(state["robot_root_qvel"])
        self._robot.set_qpos(state["robot_qpos"])
        self._robot.set_qvel(state["robot_qvel"])
        self._robot.set_qacc(state["robot_qacc"])

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts=self._scene.get_contacts()

        limpulse=get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse=get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection=self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection=-self.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle=compute_angle_between(ldirection, limpulse)
        rangle=compute_angle_between(rdirection, rimpulse)

        lflag=(
                np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag=(
                np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

