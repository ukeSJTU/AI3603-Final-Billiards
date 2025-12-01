import copy
import random
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pooltool as pt
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from config import BayesianOptConfig, RewardConfig, ShotParams
from logger import logger
from utils import BallState, BallTargets, ShotAction


def analyze_shot_for_reward(
    shot: pt.System, last_state: BallState, player_targets: BallTargets
) -> float:
    """
    分析击球结果并计算奖励分数

    Args:
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态
        player_targets: 当前玩家目标球ID列表

    Returns:
        奖励分数:
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    # 1. 分析进球情况
    new_pocketed = [
        bid
        for bid, ball in shot.balls.items()
        if ball.state.s == 4 and last_state[bid].state.s != 4
    ]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid
        for bid in new_pocketed
        if bid not in player_targets and bid not in ["cue", "8"]
    ]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = _get_first_contact_ball(shot)
    foul_first_hit = _check_first_hit_foul(
        first_contact_ball_id, last_state, player_targets
    )

    # 3. 分析碰库
    foul_no_rail = _check_rail_foul(shot, new_pocketed, first_contact_ball_id)

    # 4. 计算奖励分数
    return _calculate_reward(
        cue_pocketed=cue_pocketed,
        eight_pocketed=eight_pocketed,
        own_pocketed=own_pocketed,
        enemy_pocketed=enemy_pocketed,
        foul_first_hit=foul_first_hit,
        foul_no_rail=foul_no_rail,
        player_targets=player_targets,
    )


def _get_first_contact_ball(shot: pt.System) -> str | None:
    """获取白球首次接触的球ID"""
    for event in shot.events:
        event_type = str(event.event_type).lower()
        ids = list(event.ids) if hasattr(event, "ids") else []

        if "cushion" not in event_type and "pocket" not in event_type and "cue" in ids:
            other_ids = [i for i in ids if i != "cue"]
            if other_ids:
                return other_ids[0]
    return None


def _check_first_hit_foul(
    first_contact_ball_id: str | None,
    last_state: BallState,
    player_targets: BallTargets,
) -> bool:
    """检查首球碰撞犯规"""
    if first_contact_ball_id is None:
        # 只有白球和8号球时不算犯规
        return len(last_state) > 2

    remaining_own = [bid for bid in player_targets if last_state[bid].state.s != 4]

    opponent_plus_eight = [
        bid for bid in last_state.keys() if bid not in player_targets and bid != "cue"
    ]
    if "8" not in opponent_plus_eight:
        opponent_plus_eight.append("8")

    return len(remaining_own) > 0 and first_contact_ball_id in opponent_plus_eight


def _check_rail_foul(
    shot: pt.System, new_pocketed: list[str], first_contact_ball_id: str | None
) -> bool:
    """检查碰库犯规"""
    if len(new_pocketed) > 0 or first_contact_ball_id is None:
        return False

    cue_hit_cushion = False
    target_hit_cushion = False

    for event in shot.events:
        event_type = str(event.event_type).lower()
        ids = list(event.ids) if hasattr(event, "ids") else []

        if "cushion" in event_type:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id in ids:
                target_hit_cushion = True

    return not (cue_hit_cushion or target_hit_cushion)


def _calculate_reward(
    cue_pocketed: bool,
    eight_pocketed: bool,
    own_pocketed: list[str],
    enemy_pocketed: list[str],
    foul_first_hit: bool,
    foul_no_rail: bool,
    player_targets: BallTargets,
) -> float:
    """计算最终奖励分数"""
    score = 0.0

    # 白球和8号球进袋判断
    if cue_pocketed and eight_pocketed:
        score += RewardConfig.CUE_AND_EIGHT_POCKETED
    elif cue_pocketed:
        score += RewardConfig.CUE_POCKETED
    elif eight_pocketed:
        is_legal_eight = len(player_targets) == 1 and player_targets[0] == "8"
        score += (
            RewardConfig.LEGAL_EIGHT_BALL
            if is_legal_eight
            else RewardConfig.ILLEGAL_EIGHT_BALL
        )

    # 犯规扣分
    if foul_first_hit:
        score += RewardConfig.FOUL_FIRST_HIT
    if foul_no_rail:
        score += RewardConfig.FOUL_NO_RAIL

    # 进球得分
    score += len(own_pocketed) * RewardConfig.OWN_BALL_POCKETED
    score += len(enemy_pocketed) * RewardConfig.ENEMY_BALL_POCKETED

    # 合法无进球
    if score == 0 and not any(
        [cue_pocketed, eight_pocketed, foul_first_hit, foul_no_rail]
    ):
        score = RewardConfig.LEGAL_NO_POCKET

    return score


class Agent(ABC):
    """台球智能体基类"""

    def __init__(self):
        """初始化智能体"""
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


class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""

    def __init__(self, target_balls: BallTargets | None = None):
        """
        初始化 BasicAgent

        Args:
            target_balls: 保留参数，暂未使用
        """
        super().__init__()

        # 搜索空间
        self.pbounds = {
            "V0": (ShotParams.V0_MIN, ShotParams.V0_MAX),
            "phi": (ShotParams.PHI_MIN, ShotParams.PHI_MAX),
            "theta": (ShotParams.THETA_MIN, ShotParams.THETA_MAX),
            "a": (ShotParams.OFFSET_MIN, ShotParams.OFFSET_MAX),
            "b": (ShotParams.OFFSET_MIN, ShotParams.OFFSET_MAX),
        }

        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {"V0": 0.1, "phi": 0.1, "theta": 0.1, "a": 0.003, "b": 0.003}
        self.enable_noise = False

        logger.info("BasicAgent (贝叶斯优化) 已初始化")

    def _create_optimizer(
        self,
        reward_function: Callable[[float, float, float, float, float], float],
        seed: int,
    ) -> BayesianOptimization:
        """
        创建贝叶斯优化器

        Args:
            reward_function: 目标函数 (V0, phi, theta, a, b) -> score
            seed: 随机种子

        Returns:
            配置好的贝叶斯优化器
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=BayesianOptConfig.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed,
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=BayesianOptConfig.GAMMA_OSC, gamma_pan=BayesianOptConfig.GAMMA_PAN
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer,
        )
        optimizer._gp = gpr

        return optimizer

    def _apply_noise(self, params: dict[str, float]) -> dict[str, float]:
        """
        对参数添加噪声

        Args:
            params: 原始参数

        Returns:
            添加噪声后的参数
        """
        if not self.enable_noise:
            return params

        noisy_params = {}
        for key, value in params.items():
            noisy_value = value + np.random.normal(0, self.noise_std[key])

            # 限制在合法范围内
            if key == "V0":
                noisy_params[key] = np.clip(
                    noisy_value, ShotParams.V0_MIN, ShotParams.V0_MAX
                )
            elif key == "phi":
                noisy_params[key] = noisy_value % 360
            elif key == "theta":
                noisy_params[key] = np.clip(
                    noisy_value, ShotParams.THETA_MIN, ShotParams.THETA_MAX
                )
            else:  # a, b
                noisy_params[key] = np.clip(
                    noisy_value, ShotParams.OFFSET_MIN, ShotParams.OFFSET_MAX
                )

        return noisy_params

    def _create_reward_function(
        self,
        balls: BallState,
        my_targets: BallTargets,
        table: pt.objects.Table,
        last_state_snapshot: BallState,
    ) -> Callable[[float, float, float, float, float], float]:
        """
        创建奖励函数闭包

        Args:
            balls: 当前球状态
            my_targets: 目标球列表
            table: 球桌对象
            last_state_snapshot: 击球前状态快照

        Returns:
            奖励函数 (V0, phi, theta, a, b) -> score
        """

        def reward_fn(V0: float, phi: float, theta: float, a: float, b: float) -> float:
            # 创建模拟沙盒
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

            try:
                # 设置击球参数（可选噪声）
                params = self._apply_noise(
                    {"V0": V0, "phi": phi, "theta": theta, "a": a, "b": b}
                )
                # 只传入击球参数,不传入 cue_ball_id (已在创建 Cue 时指定)
                shot.cue.set_state(
                    V0=params["V0"],
                    phi=params["phi"],
                    theta=params["theta"],
                    a=params["a"],
                    b=params["b"],
                )

                # 运行物理模拟
                pt.simulate(shot, inplace=True)

            except Exception as e:
                logger.warning(f"模拟失败: {e}")
                return BayesianOptConfig.SIMULATION_FAILURE_PENALTY

            # 计算奖励
            return analyze_shot_for_reward(
                shot=shot, last_state=last_state_snapshot, player_targets=my_targets
            )

        return reward_fn

    def decision(self, *args, **kwargs) -> ShotAction:
        return self._random_action()
