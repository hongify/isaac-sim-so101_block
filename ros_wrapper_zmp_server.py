#!/usr/bin/env python3

import threading
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState, Image, CameraInfo


JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

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

REQUIRED_IMAGE_TOPICS = ["/rgb_1", "/rgb_2"]
OPTIONAL_IMAGE_TOPICS = ["/rgb_3"]
ALL_IMAGE_TOPICS = REQUIRED_IMAGE_TOPICS + OPTIONAL_IMAGE_TOPICS
CAMERA_INFO_TOPIC = "/camera_info"


class SO101Wrapper(Node):
    def __init__(self):
        super().__init__("so101_zmq_wrapper")

        self.lock = threading.Lock()

        self.latest_joint_positions = np.zeros(6, dtype=np.float32)
        self.latest_images = {topic: None for topic in ALL_IMAGE_TOPICS}
        self.latest_camera_info = None
        self._subscriptions = []

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._subscriptions.append(self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        ))

        for topic in ALL_IMAGE_TOPICS:
            self._subscriptions.append(self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic: self.image_callback(topic, msg),
                sensor_qos,
            ))

        self._subscriptions.append(self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            sensor_qos,
        ))

        self.joint_pub = self.create_publisher(
            JointState,
            "/joint_command",
            10,
        )

    def joint_state_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))

        try:
            joint_positions = np.array(
                [name_to_pos[name] for name in JOINT_ORDER],
                dtype=np.float32,
            )
        except KeyError:
            return

        with self.lock:
            self.latest_joint_positions = joint_positions

    def image_callback(self, topic, msg):
        try:
            image = self.image_msg_to_numpy(msg)
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed for {topic}: {e}")
            return

        with self.lock:
            self.latest_images[topic] = image

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def image_msg_to_numpy(self, msg):
        h = msg.height
        w = msg.width
        enc = msg.encoding.lower()

        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == "rgb8":
            image = data.reshape((h, msg.step))[:, :w * 3]
            image = image.reshape((h, w, 3))
            return image.copy()

        if enc == "bgr8":
            image = data.reshape((h, msg.step))[:, :w * 3]
            image = image.reshape((h, w, 3))
            image = image[:, :, ::-1]
            return image.copy()

        if enc == "rgba8":
            image = data.reshape((h, msg.step))[:, :w * 4]
            image = image.reshape((h, w, 4))
            image = image[:, :, :3]
            return image.copy()

        if enc == "bgra8":
            image = data.reshape((h, msg.step))[:, :w * 4]
            image = image.reshape((h, w, 4))
            image = image[:, :, :3]
            image = image[:, :, ::-1]
            return image.copy()

        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    def get_obs(self):
        with self.lock:
            missing = [topic for topic in REQUIRED_IMAGE_TOPICS if self.latest_images[topic] is None]
            if missing:
                return {
                    "status": "not_ready",
                    "reason": f"image not received yet: {missing}",
                }

            images = {
                topic.lstrip("/"): self.latest_images[topic].copy()
                for topic in REQUIRED_IMAGE_TOPICS
            }

            for topic in OPTIONAL_IMAGE_TOPICS:
                if self.latest_images[topic] is not None:
                    images[topic.lstrip("/")] = self.latest_images[topic].copy()

            obs = {
                "status": "ok",
                "joint_positions": self.latest_joint_positions.copy(),
                "joint_names": list(JOINT_ORDER),
                "image": images["rgb_1"],
                "images": images,
            }

            if self.latest_camera_info is not None:
                obs["camera_info"] = {
                    "width": self.latest_camera_info.width,
                    "height": self.latest_camera_info.height,
                    "k": list(self.latest_camera_info.k),
                    "p": list(self.latest_camera_info.p),
                    "d": list(self.latest_camera_info.d),
                    "distortion_model": self.latest_camera_info.distortion_model,
                    "frame_id": self.latest_camera_info.header.frame_id,
                }

            return obs

    def publish_action(self, action):
        action = np.array(action, dtype=np.float32)

        if action.shape != (6,):
            self.get_logger().warn(f"Invalid action shape: {action.shape}")
            return

        clipped = np.clip(action, LOWER_LIMITS, UPPER_LIMITS)
        if not np.allclose(clipped, action):
            self.get_logger().warn(
                f"Clipped out-of-range action to joint limits: {np.round(action, 4)} -> {np.round(clipped, 4)}"
            )
        action = clipped

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_ORDER)
        msg.position = action.tolist()

        self.joint_pub.publish(msg)


def ros_spin_thread(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = SO101Wrapper()

    thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    thread.start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

    print("ZMQ ROS wrapper running on tcp://127.0.0.1:5555")
    print(f"Required image topics: {', '.join(REQUIRED_IMAGE_TOPICS)}")
    print(f"Optional image topics: {', '.join(OPTIONAL_IMAGE_TOPICS)}")
    print(f"Camera info topic: {CAMERA_INFO_TOPIC}")

    try:
        while True:
            request = socket.recv_json()

            if request["type"] == "get_obs":
                obs = node.get_obs()
                socket.send_pyobj(obs)

            elif request["type"] == "send_action":
                action = request["joint_positions"]
                node.publish_action(action)
                socket.send_json({"status": "ok"})

            else:
                socket.send_json({
                    "status": "error",
                    "message": "unknown request",
                })

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
