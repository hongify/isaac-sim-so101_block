#!/usr/bin/env python3

import time
import math
import numpy as np

import rclpy
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from lerobot.teleoperators import make_teleoperator_from_config, so_leader
from lerobot.utils.robot_utils import precise_sleep


LEADER_PORT = "/dev/ttyUSB0"
LEADER_ID = "test_leader"

FPS = 30
COMMAND_TOPIC = "/joint_command"

DEBUG_PRINT = False
DEBUG_PRINT_EVERY_N = 10

# 전체 관절이면 None
# 하나만 테스트하려면 "elbow_flex" 같은 이름 사용
TEST_ONLY_JOINT = None
# TEST_ONLY_JOINT = "elbow_flex"

# LeRobot에서 degree로 읽음
USE_DEGREES = True
INPUT_UNIT = "deg"

# 시작할 때 leader 현재 자세를 center로 잡고 Isaac 0 자세 기준으로 움직일지 여부
# 팔이 시작하자마자 튀면 True로 바꿔서 테스트
AUTO_SET_LEADER_CENTER_ON_START = False

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ZMQ 서버에서 쓰던 Isaac SO101 joint limits
LOWER_LIMITS = np.array([
    -1.9198622,
    -1.7453293,
    -1.69,
    -1.6580628,
    -2.7438474,
    -0.1745330,
], dtype=np.float32)

UPPER_LIMITS = np.array([
     1.9198622,
     1.7453293,
     1.69,
     1.6580628,
     2.8412063,
     1.7453292,
], dtype=np.float32)

LOWER_LIMIT = dict(zip(JOINT_ORDER, LOWER_LIMITS.tolist()))
UPPER_LIMIT = dict(zip(JOINT_ORDER, UPPER_LIMITS.tolist()))

# Isaac이 반대로 움직이는 관절만 -1로 바꾸기
SIGN = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.0,
    "gripper": 1.0,
}

# leader 중립값. INPUT_UNIT과 같은 단위로 기록.
# 처음에는 0으로 두고, 필요하면 print(action) 보고 수정.
LEADER_CENTER = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

# Isaac 중립값. radian.
ISAAC_CENTER_RAD = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def to_radian(value):
    if INPUT_UNIT == "deg":
        return math.radians(value)
    if INPUT_UNIT == "rad":
        return float(value)
    raise ValueError(f"Unsupported INPUT_UNIT: {INPUT_UNIT}")


def normalize_action(raw_action):
    action = {}

    for k, v in raw_action.items():
        name = k.removesuffix(".pos")
        action[name] = float(v)

    return action


def convert_joint(joint_name, leader_value):
    q_leader = to_radian(leader_value)
    q_center = to_radian(LEADER_CENTER[joint_name])

    q_isaac = (
        SIGN[joint_name] * (q_leader - q_center)
        + ISAAC_CENTER_RAD[joint_name]
    )

    q_isaac = clamp(
        q_isaac,
        LOWER_LIMIT[joint_name],
        UPPER_LIMIT[joint_name],
    )

    return q_isaac


def build_joint_state_msg(node, action):
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()

    for joint_name in JOINT_ORDER:
        if TEST_ONLY_JOINT is not None and joint_name != TEST_ONLY_JOINT:
            continue

        if joint_name not in action:
            continue

        q = convert_joint(joint_name, action[joint_name])

        msg.name.append(joint_name)
        msg.position.append(q)

    return msg


def maybe_auto_set_leader_center(action):
    if not AUTO_SET_LEADER_CENTER_ON_START:
        return

    for joint_name in JOINT_ORDER:
        if joint_name in action:
            LEADER_CENTER[joint_name] = action[joint_name]

    print("AUTO leader center set:")
    for joint_name in JOINT_ORDER:
        print(f"  {joint_name}: {LEADER_CENTER[joint_name]:.4f}")


def main():
    rclpy.init()
    node = rclpy.create_node("so101_leader_to_isaac")

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    pub = node.create_publisher(JointState, COMMAND_TOPIC, qos)

    teleop_cfg = so_leader.SO101LeaderConfig(
        port=LEADER_PORT,
        id=LEADER_ID,
        use_degrees=USE_DEGREES,
    )

    teleop = make_teleoperator_from_config(teleop_cfg)
    teleop.connect()

    count = 0
    center_initialized = False

    print(f"Publishing JointState to {COMMAND_TOPIC}")
    print(f"JOINT_ORDER: {JOINT_ORDER}")
    print(f"TEST_ONLY_JOINT: {TEST_ONLY_JOINT}")
    print(f"INPUT_UNIT: {INPUT_UNIT}")
    print(f"USE_DEGREES: {USE_DEGREES}")

    try:
        while rclpy.ok():
            t0 = time.perf_counter()

            raw_action = teleop.get_action()
            action = normalize_action(raw_action)

            if not center_initialized:
                maybe_auto_set_leader_center(action)
                center_initialized = True

            msg = build_joint_state_msg(node, action)

            if len(msg.name) > 0:
                pub.publish(msg)

                if DEBUG_PRINT and count % DEBUG_PRINT_EVERY_N == 0:
                    print("subscription count:", pub.get_subscription_count())
                    print("raw action:", raw_action)
                    print("normalized action:", action)
                    print("published names:", msg.name)
                    print("published positions:", [round(x, 4) for x in msg.position])
                    print("-" * 60)
            else:
                if DEBUG_PRINT and count % DEBUG_PRINT_EVERY_N == 0:
                    print("WARNING: JointState msg is empty")
                    print("action keys:", list(action.keys()))
                    print("-" * 60)

            rclpy.spin_once(node, timeout_sec=0.0)

            count += 1

            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / FPS - dt, 0.0))

    except KeyboardInterrupt:
        pass

    finally:
        teleop.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()