import time
import zmq
import numpy as np
import cv2


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5555")

    print("Connected to ROS2 wrapper")
    print("Press q on image window to quit")

    while True:
        # 1. observation 요청
        socket.send_json({"type": "get_obs"})
        obs = socket.recv_pyobj()

        if obs["status"] != "ok":
            print(obs)
            time.sleep(0.1)
            continue

        q = np.array(obs["joint_positions"], dtype=np.float32)
        image_rgb = obs["image"]  # H x W x 3, RGB, uint8

        print(
            "joint:", q,
            "image:", image_rgb.shape,
            image_rgb.dtype,
            image_rgb.min(),
            image_rgb.max(),
        )

        # 2. OpenCV는 BGR 기준이라 RGB -> BGR 변환
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 필요하면 화면 크기 줄이기
        preview = cv2.resize(image_bgr, None, fx=0.75, fy=0.75)

        cv2.imshow("SO-101 Wrist Camera from ZMQ", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # 3. dummy action 전송
        action = q.copy()
        action[1] = -0.3
        action[2] = 0.5
        action[5] = 0.02

        socket.send_json({
            "type": "send_action",
            "joint_positions": action.tolist(),
        })
        reply = socket.recv_json()

        if reply.get("status") != "ok":
            print("action error:", reply)

        time.sleep(0.03)  # 약 30Hz 이하

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()