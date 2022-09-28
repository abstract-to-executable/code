from typing import List, Tuple
import numpy as np
class UniformSampler:
    """Uniform placement sampler.

    Args:
        ranges: ((low1, low2, ...), (high1, high2, ...))
        rrng (np.random.RandomState): random generator
    """

    def __init__(
        self, ranges: Tuple[List[float], List[float]], rng: np.random.RandomState
    ) -> None:
        assert len(ranges) == 2 and len(ranges[0]) == len(ranges[1])
        self._ranges = ranges
        self._rng = rng
        self._fixtures = []

    def sample(self, radius, max_trials, append=True):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            np.ndarray: a sampled position.

        """
        min_dis = 0.01 # easier to grasp
        if len(self._fixtures) == 0:
            pos = self._rng.uniform(*self._ranges)
        else:
            fixture_pos = np.array([x[0] for x in self._fixtures])
            fixture_radius = np.array([x[1] for x in self._fixtures])
            for i in range(max_trials):
                pos = self._rng.uniform(*self._ranges)
                # TODO: original code did not specify axis
                dist = np.linalg.norm(pos - fixture_pos, axis=1)
                # print(dist)
                # print(fixture_radius + radius + min_dis)
                if np.all(dist > (fixture_radius + radius + min_dis)):
                    break


        if append:
            self._fixtures.append((pos, radius))
        return pos