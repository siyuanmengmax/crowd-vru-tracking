"""Self-training baseline, cloud inference of the trained model (out/ft_ckpt_voxelnext/ep8.pth) on the evaluation clips,
exactly as lumpi_det.py with DET=st_voxelnext (config point-cloud range, boxes >= 0.1, late fusion).
Writes st_eval_out/<measurement>__<clip>/det_st_voxelnext.csv."""
import glob, os, sys
import numpy as np, pandas as pd
P = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, P)
from detector import PretrainedDetector
from fusion import late_fuse
from st_label import inputs, CFG
if __name__ == "__main__":
    det = PretrainedDetector(CFG, f"{P}/out/ft_ckpt_voxelnext/ep8.pth", openpcdet_root=f"{P}/OpenPCDet", score_thresh=0.1)
    for d in sorted(glob.glob(f"{P}/st_eval_in/*")):
        rows = []
        for f in sorted(glob.glob(f"{d}/*.npz")):
            i = int(os.path.basename(f)[:4]); clouds, offs = inputs(det, np.load(f)); mats = []
            for _, off in offs: T = np.eye(4); T[:3, 3] = off; mats.append(T)
            fused = late_fuse(det.infer(clouds), mats)
            for b, sc, l, n in zip(fused["boxes"], fused["scores"], fused["labels"], fused["num_sensors"]):
                rows.append(dict(frame_id=i, x=b[0], y=b[1], z=b[2], l=b[3], w=b[4], h=b[5], yaw=b[6], score=sc, cls=l, num_sensors=n))
        o = f"{P}/st_eval_out/{os.path.basename(d)}"; os.makedirs(o, exist_ok=True); pd.DataFrame(rows).to_csv(f"{o}/det_st_voxelnext.csv", index=False)
        print(os.path.basename(d), len(rows), flush=True)
    print("ST_INFER_DONE", flush=True)
