import pickle
import torch
import numpy as np
"""Types shared across modules."""

import enum


@enum.unique
class SequenceType(enum.Enum):
  """Sequence data types we know how to preprocess.

  If you need to preprocess additional video data, you must add it here.
  """

  FRAMES = "frames"
  FRAME_IDXS = "frame_idxs"
  VIDEO_NAME = "video_name"
  VIDEO_LEN = "video_len"

  def __str__(self):  # pylint: disable=invalid-str-returned
    return self.value
def obs_to_img(obs):
    def to_idx(a):
        return int(a[0]), int(a[1])
    
    t_xy = obs[:2]
    a_xy = obs[2:4]
    b_xy = obs[4:]
    img = np.ones((224,224, 3))
    half_size = 5
    def draw_box(a_xy, color=[0,0,0]):
        a_xy = a_xy * 224/2 + 224 / 2
        a_xy = np.round(a_xy)
        a_xy = to_idx(a_xy)
        c1=np.ones((2*half_size+1, 2*half_size+1, 3))
        c1*=0
        for i in range(3):
            c1[:,:,i]=color[i]
        img[a_xy[0]-half_size:a_xy[0]+half_size+1, a_xy[1]-half_size:a_xy[1]+half_size+1, :] = c1
    draw_box(t_xy, color=[0,0,1])
    draw_box(a_xy)
    draw_box(b_xy, color=[0,1,0])
    return img
class SkillTranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, frames_per_sequence = 10, image = False
    ):
        with open(dataset, "rb") as f:
            data = pickle.load(f)
        self.teacher_trajectories = data['teacher']
        self.traj_ids = list(self.teacher_trajectories.keys())
        self.frames_per_sequence = frames_per_sequence
        self.image = image
    def __len__(self):
        return len(self.traj_ids)
    def __getitem__(self, idx):
        traj_id = self.traj_ids[idx]

        traj = self.teacher_trajectories[traj_id]

        frames = torch.as_tensor(traj["observations"]).float()
        # sample frames
        frame_idxs = torch.randperm(len(frames))[:self.frames_per_sequence]
        frame_idxs = torch.sort(frame_idxs).values
        # _frames = frames[frame_idxs]
        image_frames = []
        sampled_frames = frames[frame_idxs]
        if self.image:
            for i in range(len(sampled_frames)):
                if self.image:
                    img = obs_to_img(sampled_frames[i]).transpose(2,0,1)
                    image_frames.append(torch.as_tensor(img).float())
            sampled_frames = torch.stack(image_frames)
        return {
            "frames": sampled_frames,
            "frame_idxs": frame_idxs,
            "video_len": torch.tensor([len(sampled_frames)]),
        }
    def collate_fn(
        self,
        batch,
    ):
        """A custom collate function for video data."""

        def _stack(key):
            return torch.stack([b[key] for b in batch])

        # Convert the keys to their string representation so that a batch can be
        # more easily indexed into without an extra import to SequenceType.
        return {
            str(SequenceType.FRAMES.value):
                _stack(SequenceType.FRAMES.value),
            str(SequenceType.FRAME_IDXS.value):
                _stack(SequenceType.FRAME_IDXS.value),
            str(SequenceType.VIDEO_LEN.value):
                _stack(SequenceType.VIDEO_LEN.value),
            # str(SequenceType.VIDEO_NAME): [
            #     b[SequenceType.VIDEO_NAME] for b in batch
            # ],
        }