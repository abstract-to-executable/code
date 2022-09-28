import sapien.core as sapien
from typing import List
import numpy as np


class SAPIENKinematicsModel:
    def __init__(self, urdf_path):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.robot = loader.load(urdf_path)
        self.scene.step()
        self.robot.set_pose(sapien.Pose())
        self.robot_model = self.robot.create_pinocchio_model()
        self.joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.link_name2id = {self.robot.get_links()[i].get_name(): i for i in range(len(self.robot.get_links()))}

        self.cached_mapping = []
        self.cached_names = ""

    def get_link_pose(self, qpos, joint_names: List[str], link_name: str):
        cat_names = "-".join(joint_names)
        if cat_names == self.cached_names:
            forward_mapping = self.cached_mapping
        else:
            print(f"Build new cached names")
            forward_mapping, _ = self.get_bidir_mapping(joint_names)
            self.cached_names = cat_names
            self.cached_mapping = forward_mapping

        inner_qpos = np.array(qpos)[forward_mapping]
        self.robot_model.compute_forward_kinematics(inner_qpos)

        link_index = self.link_name2id[link_name]
        pose = self.robot_model.get_link_pose(link_index)
        return np.concatenate([pose.p, pose.q])

    def get_bidir_mapping(self, joint_names: List[str]):
        # outter_qpos[forward_mapping] == inner_qpos
        assert len(joint_names) == len(self.joint_names)
        forward_mapping = []
        backward_mapping = []
        for joint_name in self.joint_names:
            index = joint_names.index(joint_name)
            forward_mapping.append(index)
        for joint_name in joint_names:
            index = self.joint_names.index(joint_name)
            backward_mapping.append(index)
        return forward_mapping, backward_mapping


def main():
    urdf_path = "/home/ABC/project/dexmv2/our_assets/robot/xarm6_description/xarm6_allegro_wrist_mounted_rotate.urdf"
    kinematics_model = SAPIENKinematicsModel(urdf_path)

    link_name = "palm_center"
    joint_names = kinematics_model.joint_names.copy()
    qpos = np.ones(kinematics_model.robot.dof)
    pose = kinematics_model.get_link_pose(qpos, joint_names, link_name)
    print(pose)

    print(kinematics_model.cached_mapping)
    joint_names = sorted(joint_names)
    pose = kinematics_model.get_link_pose(qpos, joint_names, link_name)
    print(pose)
    print(kinematics_model.cached_mapping)

    qpos = np.zeros(qpos.shape[0])
    pose = kinematics_model.get_link_pose(qpos, joint_names, link_name)
    print(pose)
    print(kinematics_model.cached_mapping)


if __name__ == '__main__':
    main()
