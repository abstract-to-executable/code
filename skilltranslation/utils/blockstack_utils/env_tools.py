import gym
import numpy as np
from gym import spaces
import sapien.core as sapien
from sapien.core import Pose
from typing import List, Tuple
def register_gym_env(name, **kwargs):
    """A decorator to register gym environments.
    Args:
        name (str): a unique id to register in gym.
    """
    if name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[name]

    def _register(cls):
        entry_point = "{}:{}".format(cls.__module__, cls.__name__)
        gym.register(name, entry_point=entry_point, **kwargs)
        return cls

    return _register


def get_actor_state(actor: sapien.Actor):
    pose = actor.get_pose()
    if actor.type == "static":
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
    else:
        vel = actor.get_velocity()  # [3]
        ang_vel = actor.get_angular_velocity()  # [3]
    return np.hstack([pose.p, pose.q, vel, ang_vel])


def set_actor_state(actor: sapien.Actor, state: np.ndarray):
    assert len(state) == 13, len(state)
    actor.set_pose(Pose(state[0:3], state[3:7]))
    if actor.type != "static":
        actor.set_velocity(state[7:10])
        actor.set_angular_velocity(state[10:13])


def get_articulation_state(articulation: sapien.Articulation):
    root_link = articulation.get_links()[0]
    pose = root_link.get_pose()
    vel = root_link.get_velocity()  # [3]
    ang_vel = root_link.get_angular_velocity()  # [3]
    qpos = articulation.get_qpos()
    qvel = articulation.get_qvel()
    return np.hstack([pose.p, pose.q, vel, ang_vel, qpos, qvel])


def set_articulation_state(articulation: sapien.Articulation, state: np.ndarray):
    articulation.set_root_pose(Pose(state[0:3], state[3:7]))
    articulation.set_root_velocity(state[7:10])
    articulation.set_root_angular_velocity(state[10:13])
    qpos, qvel = np.split(state[13:], 2)
    articulation.set_qpos(qpos)
    articulation.set_qvel(qvel)

def get_pairwise_contacts(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
) -> List[Tuple[sapien.Contact, bool]]:
    pairwise_contacts = []
    for contact in contacts:
        if contact.actor0 == actor0 and contact.actor1 == actor1:
            pairwise_contacts.append((contact, True))
        elif contact.actor0 == actor1 and contact.actor1 == actor0:
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[sapien.Contact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first actor
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse


def get_pairwise_contact_impulse(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
):
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse


def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)