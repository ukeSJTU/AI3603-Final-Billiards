import random
from abc import ABC, abstractmethod

from utils import ShotAction, ShotParams


class Agent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def decision(self, *args, **kwargs) -> ShotAction:
        pass

    @staticmethod
    def _random_action() -> ShotAction:
        return ShotAction(
            V0=round(random.uniform(ShotParams.V0_MIN, ShotParams.V0_MAX), 2),
            phi=round(random.uniform(ShotParams.PHI_MIN, ShotParams.PHI_MAX), 2),
            theta=round(random.uniform(ShotParams.THETA_MIN, ShotParams.THETA_MAX), 2),
            a=round(random.uniform(ShotParams.OFFSET_MIN, ShotParams.OFFSET_MAX), 3),
            b=round(random.uniform(ShotParams.OFFSET_MIN, ShotParams.OFFSET_MAX), 3),
        )
