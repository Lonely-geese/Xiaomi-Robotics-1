"""USB inference client with the exact training camera arrangement."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['XR1_VLM_PATH'] = str(ROOT / 'assets/Qwen3-VL-4B-Instruct')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from mibot.server.runtime.client import Client

TASK = 'Pick up the USB cable and insert it into the USB port.'

def prepare_views(head_rgb, left_rgb):
    head_rgb, left_rgb = head_rgb.convert('RGB'), left_rgb.convert('RGB')
    if head_rgb.size != (848, 480) or left_rgb.size != (848, 480):
        raise ValueError('Expected original head/left RGB images at 848x480; do not pre-resize them.')
    return head_rgb, left_rgb, left_rgb.crop((296, 112, 552, 368))

class USBClient(Client):
    def infer(self, robot_state, head_rgb, left_rgb, action_prefix=None):
        head, left, right = prepare_views(head_rgb, left_rgb)
        return self(robot_state, head, left, right, TASK, action_prefix=action_prefix)
