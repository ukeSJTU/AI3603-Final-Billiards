from typing import TypedDict


class ShotAction(TypedDict):
    V0: float  # 初速度(m/s)
    phi: float  # 水平角度(度)
    theta: float  # 垂直角度(度)
    a: float  # 杆头横向偏移（单位：球半径比例）
    b: float  # 杆头纵向偏移（单位：球半径比例）
