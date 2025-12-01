class ShotParams:
    """击球参数的合法范围"""

    V0_MIN, V0_MAX = 0.5, 8.0
    PHI_MIN, PHI_MAX = 0.0, 360.0
    THETA_MIN, THETA_MAX = 0.0, 90.0
    OFFSET_MIN, OFFSET_MAX = -0.5, 0.5


class RewardConfig:
    """奖励分数配置"""

    OWN_BALL_POCKETED = 50
    LEGAL_EIGHT_BALL = 100
    LEGAL_NO_POCKET = 10

    CUE_POCKETED = -100
    ILLEGAL_EIGHT_BALL = -150
    CUE_AND_EIGHT_POCKETED = -150
    FOUL_FIRST_HIT = -30
    FOUL_NO_RAIL = -30
    ENEMY_BALL_POCKETED = -20


class BayesianOptConfig:
    """贝叶斯优化配置"""

    INITIAL_SEARCH = 20
    OPT_SEARCH = 10
    ALPHA = 1e-2
    GAMMA_OSC = 0.8
    GAMMA_PAN = 1.0
    MIN_ACCEPTABLE_SCORE = 10
    SIMULATION_FAILURE_PENALTY = -500
