# test_smolvla.py

import time
import argparse
import traceback

import zmq
import numpy as np
import torch
import cv2
from transformers import AutoProcessor


MODEL_ID = "lerobot/smolvla_base"
DEFAULT_TASK = "pick up the red box on the white ground"
ZMQ_ADDR = "tcp://127.0.0.1:5555"

OBS_STATE = "observation.state"

# SmolVLA image input convention used here:
# camera1 = top view from Isaac Sim
# camera2 = wrist view from Isaac Sim
# camera3 = optional side/third view from Isaac Sim
OBS_IMAGE_1 = "observation.images.camera1"
OBS_IMAGE_2 = "observation.images.camera2"
OBS_IMAGE_3 = "observation.images.camera3"

OBS_LANG_TOKENS = "observation.language.tokens"
OBS_LANG_ATTENTION_MASK = "observation.language.attention_mask"

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Exact all-zero pose. Use only when explicitly requested with --home-pose zero.
ALL_ZERO_POSE = np.array([
    0.0,  # shoulder_pan
    0.0,  # shoulder_lift
    0.0,  # elbow_flex
    0.0,  # wrist_flex
    0.0,  # wrist_roll
    0.0,  # gripper
], dtype=np.float32)

# Safer manipulation start pose that you previously verified in Isaac Sim.
# This is the default because all-radian-zero can be a bad physical pose for the imported URDF.
NEUTRAL_POSE = np.array([
   -0.3,   # shoulder_pan
   -1.2,   # shoulder_lift
    0.55,  # elbow_flex
    1.27,  # wrist_flex
   -1.59,  # wrist_roll
    0.13,  # gripper
], dtype=np.float32)

# Hard guard matched to so101_new_calib.{urdf,xml}.
# The previous guard was tighter than the actual robot limits and could stop on valid poses.
HARD_LOWER = np.array([
    -1.9198622,   # shoulder_pan
    -1.7453293,   # shoulder_lift
    -1.69,        # elbow_flex
    -1.6580628,   # wrist_flex
    -2.7438474,   # wrist_roll
    -0.1745330,   # gripper
], dtype=np.float32)
HARD_UPPER = np.array([
     1.9198622,   # shoulder_pan
     1.7453293,   # shoulder_lift
     1.69,        # elbow_flex
     1.6580628,   # wrist_flex
     2.8412063,   # wrist_roll
     1.7453292,   # gripper
], dtype=np.float32)

IMAGE_SIZE = 256


def import_make_pre_post_processors():
    try:
        from lerobot.policies.factory import make_pre_post_processors
        return make_pre_post_processors
    except Exception:
        from lerobot.policies import make_pre_post_processors
        return make_pre_post_processors


def to_image_tensor(image_rgb, size=IMAGE_SIZE):
    """
    image_rgb: H x W x 3, uint8, RGB
    return: 3 x H x W, float32, [0, 1]
    """
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"Expected image shape HxWx3, got {image_rgb.shape}")

    image = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    image = image.float() / 255.0
    return image


def get_image_views(obs):
    """
    Prefer the multi-image schema from ros_wrapper_zmp_server.py.
    Fallback to the legacy single-image field if needed.
    """
    images = obs.get("images")
    if isinstance(images, dict) and "rgb_1" in images and "rgb_2" in images:
        rgb_1 = images["rgb_1"]
        rgb_2 = images["rgb_2"]
        rgb_3 = images.get("rgb_3", None)
        return rgb_1, rgb_2, rgb_3, list(images.keys())

    image = obs["image"]
    return image, image, None, ["image"]


def make_placeholder_image_like(wrist_tensor, mode):
    if mode == "black":
        return torch.zeros_like(wrist_tensor)
    if mode == "repeat":
        return wrist_tensor.clone()
    raise ValueError(f"Unknown placeholder mode: {mode}")


def make_policy_frame(obs, task, placeholder_mode, wrist_only=True):
    """
    Frame for LeRobot's official preprocessors.
    rgb_1 -> camera1, rgb_2 -> camera2, optional rgb_3 -> camera3.
    """
    q = np.array(obs["joint_positions"], dtype=np.float32)
    rgb_1, rgb_2, rgb_3, image_keys = get_image_views(obs)

    state = torch.from_numpy(q).float()
    img_1 = to_image_tensor(rgb_1)
    img_2 = to_image_tensor(rgb_2)
    img_3 = to_image_tensor(rgb_3) if rgb_3 is not None else None

    frame = {
        OBS_STATE: state,
        OBS_IMAGE_1: img_1,
        OBS_IMAGE_2: img_2,
        "task": task,
        "robot_type": "so101_follower",
    }

    if not wrist_only:
        if img_3 is None:
            img_3 = make_placeholder_image_like(img_2, placeholder_mode)
        frame[OBS_IMAGE_3] = img_3

    return frame


def make_manual_batch(policy, processor, obs, task, device, placeholder_mode, wrist_only=True):
    """
    Fallback path if official preprocess/postprocess fails on the local LeRobot version.
    """
    q = np.array(obs["joint_positions"], dtype=np.float32)
    rgb_1, rgb_2, rgb_3, image_keys = get_image_views(obs)

    state_tensor = torch.from_numpy(q).float().to(device)
    img_1 = to_image_tensor(rgb_1).to(device)
    img_2 = to_image_tensor(rgb_2).to(device)
    img_3 = to_image_tensor(rgb_3).to(device) if rgb_3 is not None else None

    text_inputs = processor.tokenizer(
        task,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )

    language_tokens = text_inputs["input_ids"].to(device)
    language_attention_mask = text_inputs["attention_mask"].to(device).bool()

    batch = {
        OBS_STATE: state_tensor.unsqueeze(0),
        OBS_IMAGE_1: img_1.unsqueeze(0),
        OBS_IMAGE_2: img_2.unsqueeze(0),
        OBS_LANG_TOKENS: language_tokens,
        OBS_LANG_ATTENTION_MASK: language_attention_mask,
    }

    if not wrist_only:
        if img_3 is None:
            img_3 = make_placeholder_image_like(img_2, placeholder_mode)
        batch[OBS_IMAGE_3] = img_3.unsqueeze(0)

    return batch


def extract_action_array(action):
    """
    Convert policy output or postprocessed action to numpy shape (6,).
    """
    if isinstance(action, dict):
        if "action" in action:
            action = action["action"]
        else:
            raise ValueError(f"Action dict has no 'action' key: {action.keys()}")

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.array(action, dtype=np.float32)

    if action.ndim == 3:
        # Possible: [B, T, 6]
        action = action[0, 0]
    elif action.ndim == 2:
        # Possible: [B, 6] or [T, 6]
        action = action[0]

    if action.shape != (6,):
        raise ValueError(f"Expected action shape (6,), got {action.shape}")

    return action.astype(np.float32)


def policy_action_to_target(policy_action):
    """
    Convert the policy output to a clipped Isaac Sim joint target.

    For SmolVLA pick-and-place checkpoints, the official LeRobot postprocess returns
    continuous joint positions in robot units. We therefore clip only to the
    calibrated SO101 hard limits here.
    """
    policy_action = np.array(policy_action, dtype=np.float32)

    if policy_action.shape != (6,):
        raise ValueError(f"Expected policy_action shape (6,), got {policy_action.shape}")

    if not np.all(np.isfinite(policy_action)):
        raise ValueError(f"Non-finite policy_action: {policy_action}")

    return np.clip(policy_action, HARD_LOWER, HARD_UPPER).astype(np.float32)


def is_hard_limit_violation(q, tolerance=0.0):
    q = np.array(q, dtype=np.float32)
    return bool(np.any(q < HARD_LOWER - tolerance) or np.any(q > HARD_UPPER + tolerance))


def print_hard_limit_violations(q):
    q = np.array(q, dtype=np.float32)
    print("q:", np.round(q, 6))
    print("HARD_LOWER:", np.round(HARD_LOWER, 6))
    print("HARD_UPPER:", np.round(HARD_UPPER, 6))
    print("q-lower:", np.round(q - HARD_LOWER, 6))
    print("upper-q:", np.round(HARD_UPPER - q, 6))

    for name, value, lower, upper in zip(JOINT_ORDER, q, HARD_LOWER, HARD_UPPER):
        if value < lower:
            print(f"violates lower: {name}: q={value:.6f}, lower={lower:.6f}, diff={value - lower:.6f}")
        elif value > upper:
            print(f"violates upper: {name}: q={value:.6f}, upper={upper:.6f}, diff={value - upper:.6f}")


def print_joint_table(q):
    q = np.array(q, dtype=np.float32)
    print("==== joint limit table ====")
    for name, value, lower, upper in zip(JOINT_ORDER, q, HARD_LOWER, HARD_UPPER):
        status = "ok"
        if value < lower:
            status = "below"
        elif value > upper:
            status = "above"
        print(
            f"{name:14s} q={value: .6f} lower={lower: .6f} upper={upper: .6f} "
            f"margin_low={value - lower: .6f} margin_high={upper - value: .6f} {status}"
        )


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


def get_home_pose(name):
    if name == "neutral":
        return NEUTRAL_POSE.copy()
    if name == "zero":
        return ALL_ZERO_POSE.copy()
    raise ValueError(f"Unknown home pose: {name}")


def hold_home_pose(socket, pose, seconds=3.0, rate=20.0):
    """
    Send a fixed start pose before running the policy.
    """
    print("moving to start pose:", pose)

    n = int(seconds * rate)
    dt = 1.0 / rate

    for _ in range(n):
        reply = send_action(socket, pose)
        if reply.get("status") != "ok":
            print("start pose send error:", reply)
        time.sleep(dt)

    obs, err = get_obs(socket)
    if obs is None:
        print("obs after start pose not ready:", err)
        return

    q = np.array(obs["joint_positions"], dtype=np.float32)
    image = obs.get("image", None)
    print("q after start pose hold:", np.round(q, 4))
    if image is not None:
        print("image after start pose:", image.shape, image.dtype, image.min(), image.max())


def print_batch_keys(batch):
    print("==== batch keys ====")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape, v.dtype, v.device)
        else:
            print(k, type(v))


def print_step_info(step, q, policy_action, target, image, task):
    print(f"\n==== step {step} ====")
    print("task:", task)
    print("image:", image.shape, image.dtype, image.min(), image.max())
    print("q:             ", np.round(q, 4))
    print("policy_action:  ", np.round(policy_action, 4))
    print("target:        ", np.round(target, 4))
    print("target-q:      ", np.round(target - q, 4))
    print("policy|maxabs:  ", np.round(np.max(np.abs(policy_action)), 4))


def print_obs_schema(obs):
    print("==== obs schema ====")
    print("status:", obs.get("status"))
    print("joint_names:", obs.get("joint_names"))
    if "images" in obs and isinstance(obs["images"], dict):
        print("image keys:", list(obs["images"].keys()))
        for key, img in obs["images"].items():
            print(f"{key}: {img.shape} {img.dtype} {img.min()} {img.max()}")
    else:
        image = obs.get("image")
        if image is not None:
            print("image:", image.shape, image.dtype, image.min(), image.max())


def load_policy_and_processors(device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print("loading policy:", MODEL_ID)
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy = policy.to(device)
    policy.eval()

    print("policy input_features:", policy.config.input_features)
    print("policy output_features:", policy.config.output_features)
    print("policy chunk_size:", getattr(policy.config, "chunk_size", None))
    print("policy n_action_steps:", getattr(policy.config, "n_action_steps", None))

    make_pre_post_processors = import_make_pre_post_processors()

    print("loading official pre/post processors")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={
            "device_processor": {
                "device": str(device),
            }
        },
    )

    return policy, preprocess, postprocess


def infer_action(
    policy,
    preprocess,
    postprocess,
    processor,
    obs,
    task,
    device,
    use_official_processors,
    wrist_only,
):
    if use_official_processors:
        frame = make_policy_frame(
            obs=obs,
            task=task,
            placeholder_mode="black",
            wrist_only=wrist_only,
        )
        batch = preprocess(frame)
    else:
        batch = make_manual_batch(
            policy=policy,
            processor=processor,
            obs=obs,
            task=task,
            device=device,
            placeholder_mode="black",
            wrist_only=wrist_only,
        )

    with torch.inference_mode():
        action = policy.select_action(batch)

    action = postprocess(action)
    policy_action = extract_action_array(action)
    mode = "official_prepost" if use_official_processors else "manual_batch_plus_postprocess"
    return policy_action, batch, mode


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Language instruction",
    )
    parser.add_argument("--send", action="store_true", help="Actually send action to Isaac Sim")
    parser.add_argument("--loop", action="store_true", help="Run closed-loop")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=0, help="0 means infinite")
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument(
        "--camera-mode",
        type=str,
        default="multi-camera",
        choices=["wrist-only", "multi-camera"],
        help="Use only the wrist camera, or include placeholder side camera inputs",
    )

    parser.add_argument(
        "--home-pose",
        type=str,
        default="neutral",
        choices=["neutral", "zero"],
        help="Start pose before policy. neutral is safer; zero sends all joint positions as 0.",
    )

    parser.add_argument(
        "--zero-seconds",
        type=float,
        default=3.0,
        help="Seconds to hold the selected start pose before model start",
    )

    parser.add_argument(
        "--manual-fallback",
        action="store_true",
        help="Use manual batch instead of official preprocess/postprocess",
    )

    parser.add_argument(
        "--allow-black-image",
        action="store_true",
        help="Do not skip frames when camera image is all black",
    )

    parser.add_argument(
        "--no-hard-stop",
        action="store_true",
        help="Do not stop on hard joint limit violation",
    )
    parser.add_argument(
        "--hard-limit-tolerance",
        type=float,
        default=0.0,
        help="Tolerance added to hard joint guards before triggering hard stop",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Print current observation/joint/camera schema and exit before loading the policy",
    )

    parser.add_argument(
        "--replan-each-step",
        action="store_true",
        help="Reset SmolVLA action queue before every inference so each step replans from the latest observation",
    )

    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be positive")

    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")

    if args.hard_limit_tolerance < 0:
        raise ValueError("--hard-limit-tolerance must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(ZMQ_ADDR)
    print("connected to ROS2 wrapper:", ZMQ_ADDR)

    start_pose = get_home_pose(args.home_pose)

    if args.send:
        hold_home_pose(socket, pose=start_pose, seconds=args.zero_seconds, rate=20.0)
    else:
        print("dry-run mode: start pose is not sent. Use --send to move robot.")
        print("selected start pose:", args.home_pose, start_pose)

    if args.diagnose_only:
        obs, err = get_obs(socket)
        if obs is None:
            print("obs not ready:", err)
            return
        q = np.array(obs["joint_positions"], dtype=np.float32)
        print_obs_schema(obs)
        print_joint_table(q)
        if is_hard_limit_violation(q, tolerance=args.hard_limit_tolerance):
            print("diagnosis: current q violates hard guard before policy inference.")
            print_hard_limit_violations(q)
        else:
            print("diagnosis: current q is inside hard guard.")
        return

    policy, preprocess, postprocess = load_policy_and_processors(device)

    if hasattr(policy, "reset"):
        policy.reset()

    processor = None
    if args.manual_fallback:
        print("loading AutoProcessor for manual fallback")
        processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)

    use_official_processors = not args.manual_fallback

    print("task:", args.task)
    print("camera mode:", args.camera_mode)
    print("home pose:", args.home_pose)
    print("send:", args.send)
    print("rate:", args.rate)
    print("steps:", args.steps)
    print("hard limit tolerance:", args.hard_limit_tolerance)
    print("model:", MODEL_ID)
    print("action space: continuous joint positions after LeRobot postprocess")
    print("expected image features:", OBS_IMAGE_1, OBS_IMAGE_2, OBS_IMAGE_3)
    print("expected image tensor shape:", (3, IMAGE_SIZE, IMAGE_SIZE))
    print("replan each step:", args.replan_each_step)

    wrist_only = args.camera_mode == "wrist-only"

    dt = 1.0 / args.rate
    step = 0

    while True:
        if args.steps > 0 and step >= args.steps:
            print("finished max steps:", args.steps)
            break

        t0 = time.time()

        obs, err = get_obs(socket)
        if obs is None:
            print("obs not ready:", err)
            time.sleep(0.1)
            continue

        q = np.array(obs["joint_positions"], dtype=np.float32)
        rgb_1, rgb_2, rgb_3, image_keys = get_image_views(obs)
        image = rgb_1

        if args.replan_each_step and hasattr(policy, "reset"):
            policy.reset()

        if (not args.no_hard_stop) and is_hard_limit_violation(q, tolerance=args.hard_limit_tolerance):
            print("HARD STOP: current q is outside hard guard before policy action is applied.")
            print_hard_limit_violations(q)
            print("Reset manually or run with --home-pose neutral --send without --loop.")
            break

        required_images = {"rgb_1": rgb_1, "rgb_2": rgb_2}
        if wrist_only:
            required_images = {"rgb_2": rgb_2}

        black_images = [name for name, img in required_images.items() if img.max() == 0]
        if black_images and not args.allow_black_image:
            print("WARNING: camera image is black. Skip this step.")
            print("black image views:", black_images)
            print("q:", np.round(q, 4))
            print("Reset to neutral pose or check camera graph/path/lighting.")
            time.sleep(0.1)
            continue

        if step == 0:
            print_obs_schema(obs)
            print("selected image views:", image_keys)

        try:
            policy_action, batch, mode = infer_action(
                policy=policy,
                preprocess=preprocess,
                postprocess=postprocess,
                processor=processor,
                obs=obs,
                task=args.task,
                device=device,
                use_official_processors=use_official_processors,
                wrist_only=wrist_only,
            )
        except Exception:
            print("official preprocess/postprocess failed.")
            traceback.print_exc()

            if processor is None:
                processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)

            print("retrying with manual fallback batch")
            policy_action, batch, mode = infer_action(
                policy=policy,
                preprocess=preprocess,
                postprocess=postprocess,
                processor=processor,
                obs=obs,
                task=args.task,
                device=device,
                use_official_processors=False,
                wrist_only=wrist_only,
            )

        if step == 0:
            print("inference mode:", mode)
            print_batch_keys(batch)

        target = policy_action_to_target(policy_action)

        if step % args.print_every == 0:
            print_step_info(
                step=step,
                q=q,
                policy_action=policy_action,
                target=target,
                image=image,
                task=args.task,
            )

        if args.send:
            reply = send_action(socket, target)
            if step % args.print_every == 0:
                print("send_action reply:", reply)
        else:
            if step % args.print_every == 0:
                print("dry-run only. Use --send to move robot.")

        if not args.loop:
            break

        step += 1

        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
