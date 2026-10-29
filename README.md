# SO101 Block Picking

Block-picking experiments with the SO101 robot in Isaac Sim.
Control the robot with a leader arm or run inference with SmolVLA.

- `so101_testScene.usd`: test scene
- `so101_robot.usd`: robot model
- `so101_teleop_ros2.py`: sends leader arm input as ROS 2 joint commands
- `ros_wrapper_zmp_server.py`: connects ROS 2 and ZMQ
- `test_smolvla.py`: SmolVLA inference test
- `tools/`: teleoperation, recording, and debugging scripts

Update serial ports and model paths for your setup before running.

2026.06.22
