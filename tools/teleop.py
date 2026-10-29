#!/usr/bin/env python3

import argparse
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np
import zmq


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

INITIAL_POSITION = {
    "shoulder_pan": -0.3,
    "shoulder_lift": -1.2,
    "elbow_flex": 0.55,
    "wrist_flex": 1.27,
    "wrist_roll": -1.59,
    "gripper": 0.13,
}

JOINT_LIMITS = {
    "shoulder_pan": (-3.14, 3.14),
    "shoulder_lift": (-1.57, 1.57),
    "elbow_flex": (-1.57, 1.57),
    "wrist_flex": (-1.57, 1.57),
    "wrist_roll": (-3.14, 3.14),
    "gripper": (0.0, 1.22),
}

STEP = {
    "shoulder_pan": 0.05,
    "shoulder_lift": 0.05,
    "elbow_flex": 0.05,
    "wrist_flex": 0.05,
    "wrist_roll": 0.05,
    "gripper": 0.005,
}

KEY_BINDINGS = {
    "a": ("shoulder_pan", +1),
    "d": ("shoulder_pan", -1),
    "w": ("shoulder_lift", +1),
    "s": ("shoulder_lift", -1),
    "r": ("elbow_flex", +1),
    "f": ("elbow_flex", -1),
    "t": ("wrist_flex", +1),
    "g": ("wrist_flex", -1),
    "y": ("wrist_roll", +1),
    "h": ("wrist_roll", -1),
    "o": ("gripper", +1),
    "p": ("gripper", -1),
}

HELP_TEXT = """
SO-101 Keyboard Teleop via ros_wrapper_zmp_server.py

Joint controls:
  a / d : shoulder_pan   + / -
  w / s : shoulder_lift  + / -
  r / f : elbow_flex     + / -
  t / g : wrist_flex     + / -
  y / h : wrist_roll     + / -
  o / p : gripper        open / close

Other:
  space : hold current position
  z     : reset to initial pose
  q     : quit
"""

ZMQ_ADDR = "tcp://127.0.0.1:5555"


def clamp(joint_name, value):
    lower, upper = JOINT_LIMITS[joint_name]
    return max(lower, min(upper, value))


def get_key(timeout=0.01):
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if readable:
        return sys.stdin.read(1)
    return None


def get_obs(socket):
    socket.send_json({"type": "get_obs"})
    obs = socket.recv_pyobj()

    if obs["status"] != "ok":
        return None, obs

    return obs, None


def send_action(socket, target):
    socket.send_json({
        "type": "send_action",
        "joint_positions": target.tolist(),
    })
    return socket.recv_json()


def apply_key(position, key):
    if key in KEY_BINDINGS:
        joint_name, direction = KEY_BINDINGS[key]
        position[joint_name] = clamp(
            joint_name,
            position[joint_name] + STEP[joint_name] * direction,
        )
        return True

    if key == "z":
        position.update(INITIAL_POSITION)
        return True

    return False


def position_to_array(position):
    return np.array([position[name] for name in JOINT_NAMES], dtype=np.float32)


def blank_panel(label, width=640, height=360):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        label,
        (24, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def show_camera_views(obs):
    images = obs.get("images")
    if not isinstance(images, dict):
        image = obs.get("image")
        images = {"rgb_1": image, "rgb_2": image, "rgb_3": None}

    for topic, title in (
        ("rgb_1", "camera 1 / rgb_1"),
        ("rgb_2", "camera 2 / rgb_2"),
        ("rgb_3", "camera 3 / rgb_3"),
    ):
        image = images.get(topic)
        if image is None:
            frame = blank_panel(f"{title} (waiting)")
        else:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, frame)

    cv2.waitKey(1)


def print_state(position):
    state = " | ".join(f"{name}: {position[name]:+.3f}" for name in JOINT_NAMES)
    print("\r" + state, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default=ZMQ_ADDR, help="ZMQ address of ros_wrapper_zmp_server.py")
    parser.add_argument("--rate", type=float, default=20.0, help="Control/display loop rate in Hz")
    parser.add_argument("--dry-run", action="store_true", help="Do not send commands to the wrapper")
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be positive")

    print(HELP_TEXT)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(args.addr)
    print("connected to ROS2 wrapper:", args.addr)
    print("mode:", "dry-run" if args.dry_run else "send")

    position = dict(INITIAL_POSITION)
    old_settings = termios.tcgetattr(sys.stdin)
    dt = 1.0 / args.rate

    try:
        tty.setcbreak(sys.stdin.fileno())
        print_state(position)

        while True:
            t0 = time.time()

            obs, err = get_obs(socket)
            if obs is None:
                print("\nobs not ready:", err)
                time.sleep(0.1)
                continue

            show_camera_views(obs)

            key = get_key(timeout=0.0)
            if key == "q":
                print("\nQuit.")
                break

            changed = False
            if key is not None:
                changed = apply_key(position, key)
                if changed:
                    print_state(position)

            target = position_to_array(position)

            if not args.dry_run:
                reply = send_action(socket, target)
                if key is not None or changed:
                    print("\nsend_action reply:", reply)
            else:
                if key is not None and changed:
                    print("\ndry-run only. Run without --dry-run to actually move the robot.")

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
