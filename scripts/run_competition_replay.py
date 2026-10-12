"""Software-only competition replay; real FPGA/PCIe is deliberately out of scope."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2
from competition_ai import CompetitionAdas,CompetitionDetector,SceneInput,TrafficLightState,classify_traffic_lights

def main():
 p=argparse.ArgumentParser(); p.add_argument('--video',required=True); p.add_argument('--model',required=True); p.add_argument('--input-size',type=int,default=320); p.add_argument('--output-dir',required=True); p.add_argument('--scenario',default='unknown'); p.add_argument('--max-frames',type=int); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 cap=cv2.VideoCapture(a.video); fps=cap.get(cv2.CAP_PROP_FPS) or 25; writer=None; detections=[]; logs=[]; adas=CompetitionAdas(); fid=0
 with CompetitionDetector(a.model,a.input_size) as det:
  while cap.isOpened() and (a.max_frames is None or fid<a.max_frames):
   ok,frame=cap.read()
   if not ok: break
   if writer is None: writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
   result=det.infer(frame,stream_id=0,frame_id=fid,timestamp_ns=int(fid*1e9/fps)); lights=[d for d in result.detections if d.class_name=='traffic light']; state=TrafficLightState.UNKNOWN
   if lights: state=classify_traffic_lights(frame,lights)
   scene=SceneInput(traffic_light_state=state.value,lane_type='unknown',navigation_command='none',vehicle_command='none',pedestrian_hazard=any(d.class_name=='person' and d.y2>frame.shape[0]*.8 for d in result.detections)); command=adas.decide(scene)
   for d in result.detections: detections.append({'frame_id':fid,'timestamp':fid/fps,'class':d.class_name,'confidence':d.confidence,'x1':d.x1,'y1':d.y1,'x2':d.x2,'y2':d.y2,'inference_ms':result.inference_ms}); cv2.rectangle(frame,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(0,255,0),2)
   logs.append({'frame_id':fid,'timestamp':fid/fps,'detections':len(result.detections),'traffic_light_state':state.value,'lane_state':'UNKNOWN','adas_action':command.action,'warning_flags':list(command.warning_flags),'inference_ms':result.inference_ms,'scenario':a.scenario}); cv2.putText(frame,f'{state.value} {command.action}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,255) if command.action=='STOP' else (0,255,0),2); writer.write(frame); fid+=1
 cap.release(); writer and writer.release()
 for name,rows,fields in [('detections.csv',detections,['frame_id','timestamp','class','confidence','x1','y1','x2','y2','inference_ms']),('adas_log.csv',logs,['frame_id','timestamp','detections','traffic_light_state','lane_state','adas_action','warning_flags','inference_ms','scenario'])]:
  with (out/name).open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 summary={'status':'SOFTWARE PIPELINE TEST','video':a.video,'frames':fid,'detections':len(detections),'mean_inference_ms':sum(x['inference_ms'] for x in logs)/len(logs) if logs else 0,'empty_frame_ratio':sum(x['detections']==0 for x in logs)/len(logs) if logs else 0,'scenario':a.scenario}; (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); (out/'summary.md').write_text('# Competition software replay\n\n'+json.dumps(summary,indent=2)+'\n',encoding='utf8')
if __name__=='__main__': main()
