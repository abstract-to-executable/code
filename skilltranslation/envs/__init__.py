import sapien
sapien_v2 = True

if not hasattr(sapien, "__version__"):
    sapien_v2 = False

import skilltranslation.envs.xmagical.traj_env
if sapien_v2:
    import skilltranslation.envs.boxpusher.env
    import skilltranslation.envs.boxpusher.traj_env
    import skilltranslation.envs.maze.traj_env
    import skilltranslation.envs.blockstacking
else:
    import skilltranslation.envs.maniskill.traj_env
