import os.path as osp

import gym
import numpy as np
from gym import spaces

from skilltranslation.envs.boxpusher.env import BoxPusherEnv
from skilltranslation.envs.boxpusher.traj_env import BoxPusherTrajectory
from skilltranslation.planner.base import HighLevelPlanner
from skilltranslation.planner.boxpusherteacher import (BoxPusherReacherPlanner,
                                                       BoxPusherTaskPlanner)
from skilltranslation.utils.animate import animate


class BoxPusherTranslationEnv(BoxPusherTrajectory):
    def __init__(
        self,
        planner: HighLevelPlanner,
        balls=1,
        trajectories=...,
        max_trajectory_length=100,
        max_ep_len_factor=2,
        data_dir=None,
        stack_size=1,
        fixed_max_ep_len=None,
        max_trajectory_skip_steps=15,
        dense_obs_only=False,
        max_stray_dist=0.3,
        exclude_target_state=False,
        max_plan_length=100,
        obs_mode="dense",
        control_type="2D-continuous",
        render_plan=False,
        save_plan_videos=False,
        **kwargs
    ):
        self.planner = planner
        self.planning_env = gym.make('BoxPusher-v0',
            balls=balls,
            # controlled_ball_radius=controlled_ball_radius,
            # target_radius=target_radius,
            # ball_radius=ball_radius,
            control_type="2D",
            magic_control=True,
            obs_mode='dict',
            reward_type='sparse',
            **kwargs,
        )
        self.plan_id = 0
        self.save_plan_videos = save_plan_videos
        self.render_plan=render_plan
        self.max_plan_length = max_plan_length
        super().__init__(
            trajectories,
            max_trajectory_length,
            max_ep_len_factor,
            data_dir,
            stack_size,
            fixed_max_ep_len,
            max_trajectory_skip_steps,
            dense_obs_only,
            max_stray_dist,
            exclude_target_state,
            obs_mode=obs_mode,
            control_type=control_type,
            **kwargs
        )
        
    def plan_trajectory(self, start_state):
        # turn on magic grip and more
        print("PLANNING TRAJECTORY")
        self.planning_env._set_state(start_state)

        obs = self.planning_env._get_obs()
        done = False,
        if self.render_plan:
            self.planning_env.render()
            viewer = self.planning_env.viewer
            # viewer.paused=True
        prev_successes = 0
        for meta in self.env.balls_meta:
            if meta["done"]:
                prev_successes += 1

        # TODO, fixed / hardcoded to observe agent and target ball only
        observations = []
        imgs = []
        # self.reward_trajectory = trajectory['observations'][:-1]
        for i in range(self.max_plan_length):
            if self.render_plan:
                self.planning_env.render()
            if self.save_plan_videos:
                img = self.planning_env.render(mode="rgb_array")
                imgs.append(img)
            a = self.planner.act(obs)
            obs, reward, done, info = self.planning_env.step(a)
            # stop planning once we finish a sub-task or finish whole task
            if info["successes"] > prev_successes: break
            if done: break
            stacked_obs = np.hstack([obs["target"], obs["agent_ball"], obs["target_ball"]])
            observations.append(stacked_obs)
        observations = np.stack(observations).copy()
        print("STEPS", len(observations))
        # self.steps = 0
        if self.save_plan_videos:
            animate(imgs, filename=f"plan_{self.plan_id}.mp4", _return=False, fps=24)
        self.plan_id += 1
        return {
            "observations": observations
        }
    def get_trajectory(self, t_idx):
        self.env.reset()
        obs = self.env._get_obs()
        obs[:2] = np.array([-0.5, 0.25])
        obs[2:4] = np.array([-0.25, -0.25])
        obs[4:] = np.array([0.25, 0.25])
        self.planning_env.reset()
        return self.plan_trajectory(obs)

if __name__ == "__main__":
    env = BoxPusherTranslationEnv(
        trajectories=[0], stack_size=20,
        planner=BoxPusherTaskPlanner(),
        max_trajectory_length=150,
        # data_dir="./data_v3/trajectories",
        obs_mode="dense",
        # dense_obs_only=True,
        fixed_env=True
    )
    env.seed(0)
    o = env.reset()
    print(o)