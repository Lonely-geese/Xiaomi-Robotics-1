import sys,time,json
from pathlib import Path
from common import ROOT, verify_dependencies
p=verify_dependencies();s=p.load('feedback_probe',p.STABLE/'run_robot.py')
c=s.RobotHalClient(host='127.0.0.1',port=7860,transport='tcp');c.connect()
n=bad=0;start=time.monotonic();next_report=15
try:
 while time.monotonic()-start<float(sys.argv[1]):
  d=c.get(s.FEEDBACK_KEYS);n+=1
  try:s.decode_feedback(d)
  except Exception as e:
   bad+=1
   if bad<4: print({'error':str(e),'keys':list(d)},flush=True)
  elapsed=time.monotonic()-start
  if elapsed>=next_report:
   print({'elapsed_s':round(elapsed,1),'samples':n,'bad':bad},flush=True);next_report+=15
  time.sleep(.016)
finally:c.close()
report={'samples':n,'bad':bad,'duration_s':round(time.monotonic()-start,2),'robot_commands_sent':0}
(ROOT/'runs').mkdir(exist_ok=True)
(ROOT/'runs/hal_restart_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
