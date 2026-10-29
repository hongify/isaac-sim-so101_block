import os
import time
import zmq
import cv2
import numpy as np


SAVE_ROOT = "episodes"
EPISODE_NAME = "ep_0000"

HZ = 10
DT = 1.0 / HZ
MAX_STEPS = 300  # 30초


def make_dummy_action(q):
    action = q.copy()

    # 현재는 안전한 고정 자세 유지
    action[1] = -0.3
    action[2] = 0.5
    action[5] = 0.02

    return action


def main():
    episode_dir = os.path.join(SAVE_ROOT, EPISODE_NAME)
    image_dir = os.path.join(episode_dir, "images")

    os.makedirs(image_dir, exist_ok=True)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5555")

    joint_list = []
    action_list = []
    timestamp_list = []
    image_file_list = []

    print("Recording episode...")
    print(f"save dir: {episode_dir}")

    for step in range(MAX_STEPS):
        t0 = time.time()

        # 1. obs 받기
        socket.send_json({"type": "get_obs"})
        obs = socket.recv_pyobj()

        if obs["status"] != "ok":
            print(obs)
            time.sleep(0.1)
            continue

        q = np.array(obs["joint_positions"], dtype=np.float32)
        image_rgb = obs["image"]

        # 2. action 생성
        action = make_dummy_action(q)

        # 3. action 전송
        socket.send_json({
            "type": "send_action",
            "joint_positions": action.tolist(),
        })
        reply = socket.recv_json()

        if reply.get("status") != "ok":
            print("action error:", reply)
            continue

        # 4. image 저장
        image_name = f"{step:06d}.png"
        image_path = os.path.join(image_dir, image_name)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image_bgr)

        # 5. metadata 저장용으로 누적
        joint_list.append(q)
        action_list.append(action)
        timestamp_list.append(time.time())
        image_file_list.append(os.path.join("images", image_name))

        print(
            f"step {step:04d}",
            "joint=", np.round(q, 3),
            "action=", np.round(action, 3),
            "image=", image_rgb.shape,
        )

        # 6. rate 맞추기
        elapsed = time.time() - t0
        sleep_time = max(0.0, DT - elapsed)
        time.sleep(sleep_time)

    np.savez(
        os.path.join(episode_dir, "data.npz"),
        joint_positions=np.stack(joint_list),
        actions=np.stack(action_list),
        timestamps=np.array(timestamp_list, dtype=np.float64),
        image_files=np.array(image_file_list),
    )

    print("Saved episode.")
    print(f"episode dir: {episode_dir}")


if __name__ == "__main__":
    main()