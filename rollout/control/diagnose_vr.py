"""Read-only Quest event diagnostics; never connects to robot HAL."""
import json, logging, sys, time
from pathlib import Path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/vr'))
import deeptouch_vr_controls as vr
vr._ensure_teleop_on_path(vr._find_tuozhuai())
from deeptouch_teleop.devices.vr.quest.device import VRDevice
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
device=VRDevice(mode='local')
mapper=vr.VrManualActionMapper()
last_ref=last_keys=None
last_report=0
try:
 device.start()
 deadline=time.monotonic()+180
 while time.monotonic()<deadline:
  snap=device.raw_snapshot()
  actions,last_ref=vr.process_vr_snapshot(snap,mapper=mapper,last_ref=last_ref)
  frame=snap.frame
  keys=None if frame is None else (frame.left_keys.one,frame.left_keys.two,round(frame.left_keys.index_trigger,2),round(frame.left_keys.thumb_x,2),round(frame.left_keys.thumb_y,2),frame.right_keys.one,frame.right_keys.two)
  now=time.monotonic()
  if keys!=last_keys or actions or now-last_report>5:
   print(json.dumps({'state':str(snap.connection_state),'epoch':snap.epoch,'sequence':snap.sequence,'timestamp':None if frame is None else frame.timestamp,'age_s':None if snap.received_at is None else round((time.monotonic_ns()-snap.received_at)/1e9,3),'flags':None if frame is None else frame.flags,'keys_left_X_Y_trigger_stick_right_A_B':keys,'mapped_actions':actions,'stats':vars(device.stats)}),flush=True)
   last_report=now;last_keys=keys
  time.sleep(.01)
finally:
 device.stop()
