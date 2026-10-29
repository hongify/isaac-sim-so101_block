import sys
import torch
import lerobot

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))

print("lerobot import ok")