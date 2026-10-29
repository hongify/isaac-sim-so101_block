import time
import argparse

import zmq
import numpy as np
import torch
import cv2


MODEL_ID = "lerobot/smolvla_base"
TASK = "pick up the box with the gripper"

ZMQ_ADDR = "tcp://127.0.0.1:5555"

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


def to_image_tensor(image_rgb, size=256):
    """
    image_rgb: H x W x 3, uint8, RGB
    return: 3 x H x W, float32, [0, 1]
    """
    image = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    image = image.float() / 255.0
    return image


def get_image_views(obs):
    images = obs.get("images")
    if isinstance(images, dict) and "rgb_1" in images and "rgb_2" in images:
        return images["rgb_1"], images["rgb_2"]
    image = obs["image"]
    return image, image


def make_language_inputs(processor, task, policy, device):
    text_inputs = processor.tokenizer(
        task,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )

    language_tokens = text_inputs["input_ids"].to(device)
    language_attention_mask = text_inputs["attention_mask"].to(device).bool()

    return language_tokens, language_attention_mask


def import_make_pre_post_processors():
    try:
        from lerobot.policies.factory import make_pre_post_processors
        return make_pre_post_processors
    except Exception:
        from lerobot.policies import make_pre_post_processors
        return make_pre_post_processors


def extract_action_array(action):
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.array(action, dtype=np.float32)

    # 가능한 shape 처리:
    # (6,)
    # (1, 6)
    # (T, 6)
    if action.ndim == 2:
        action = action[0]

    if action.shape != (6,):
        raise ValueError(f"Expected action shape (6,), got {action.shape}")

    return action


def raw_action_to_target(raw_action):
    """
    For lerobot/smolvla_base, postprocessed actions are continuous joint
    positions. Clip them to the calibrated SO101 range before sending.
    """
    raw_action = np.array(raw_action, dtype=np.float32)

    if raw_action.shape != (6,):
        raise ValueError(f"Expected raw_action shape (6,), got {raw_action.shape}")
    return np.clip(raw_action, LOWER_LIMITS, UPPER_LIMITS).astype(np.float32)


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
    reply = socket.recv_json()
    return reply


def print_obs_info(obs):
    q = np.array(obs["joint_positions"], dtype=np.float32)
    image = obs["image"]

    print("==== obs ====")
    print("joint_positions:", np.round(q, 4))
    print("image:", image.shape, image.dtype, image.min(), image.max())


def print_action_info(q, raw_action, target):
    print("==== action ====")
    print("q:          ", np.round(q, 4))
    print("raw_action: ", np.round(raw_action, 4))
    print("target:     ", np.round(target, 4))
    print("delta:      ", np.round(target - q, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="Actually send one direct policy action to Isaac Sim")
    parser.add_argument("--loop", action="store_true", help="Run repeated direct control loop")
    parser.add_argument("--rate", type=float, default=10.0, help="Control loop rate in Hz")
    parser.add_argument(
        "--replan-each-step",
        action="store_true",
        help="Reset SmolVLA action queue before every inference so each step replans from the latest observation",
    )
    args = parser.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("loading policy:", MODEL_ID)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy = policy.to(device)
    policy.eval()
    print("policy chunk_size:", getattr(policy.config, "chunk_size", None))
    print("policy n_action_steps:", getattr(policy.config, "n_action_steps", None))
    print("model:", MODEL_ID)
    print("action space: continuous joint positions after LeRobot postprocess")
    print("replan each step:", args.replan_each_step)

    make_pre_post_processors = import_make_pre_post_processors()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={
            "device_processor": {
                "device": str(device),
            }
        },
    )

    if hasattr(policy, "reset"):
        policy.reset()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(ZMQ_ADDR)

    print("connected to ROS2 wrapper:", ZMQ_ADDR)
    print("task:", TASK)

    dt = 1.0 / args.rate

    while True:
        t0 = time.time()

        obs, err = get_obs(socket)
        if obs is None:
            print("obs not ready:", err)
            time.sleep(0.1)
            continue

        q = np.array(obs["joint_positions"], dtype=np.float32)

        print_obs_info(obs)

        if args.replan_each_step and hasattr(policy, "reset"):
            policy.reset()

        rgb_1, rgb_2 = get_image_views(obs)
        frame = {
            "observation.state": torch.from_numpy(np.array(obs["joint_positions"], dtype=np.float32)).float(),
            "observation.images.camera1": to_image_tensor(rgb_1, size=256),
            "observation.images.camera2": to_image_tensor(rgb_2, size=256),
            "task": TASK,
            "robot_type": "so101_follower",
        }
        batch = preprocess(frame)

        with torch.inference_mode():
            action = policy.select_action(batch)

        action = postprocess(action)
        raw_action = extract_action_array(action)
        target = raw_action_to_target(raw_action)

        print_action_info(q, raw_action, target)

        if args.send:
            reply = send_action(socket, target)
            print("send_action reply:", reply)
        else:
            print("dry-run only. Use --send to actually move the robot.")

        if not args.loop:
            break

        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
