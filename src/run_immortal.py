"""ImmortalTracker (Wang et al. 2021, official code third_party/ImmortalTracker, commit 11f8855) on an existing input tag.
Published nuScenes 'immortal' configuration (configs/nu_configs/immortal.yaml: tracks never die, GIoU association with
threshold 1.5, bipartite matching, NMS 0.1), except min_hits_to_birth = MINHITS (default 1 = published), which is the only
lifecycle parameter left and is swept on the V2X-Real training splits like AB3DMOT's track management.
Input: VRU boxes (label 0) of <src>/detection_bbox_results.csv; ego identity (static sensor); 10 Hz frame index as time.
Output: <WORK>/<seq>/<src>_im<MINHITS>/Immortal_matched.csv (frame_id, track_id, label, confidence, x, y, z, w, l, h, yaw),
only boxes the tracker marks valid in that frame (alive and associated), as in the official output.
usage: WORK=... MINHITS=1 python run_immortal.py <src_tag> <seq> [<seq> ...]"""
import os, sys, yaml, numpy as np, pandas as pd
from multiprocessing import Pool
IT = os.environ.get("IMMORTAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "ImmortalTracker"))
sys.path.insert(0, IT)
if not hasattr(np, "int"): np.int = int   # the official code uses the alias removed in NumPy 1.24; same behavior
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from mot_3d.data_protos import BBox, Validity
from mot_3d.preprocessing import nms
import mot_3d.association as A
from mot_3d import utils as U

def gated_iou_distance(dets, tracks, asso="iou"):
    """Same as A.compute_iou_distance, but pairs whose ground-plane centers are farther apart than
    3 x (longest side of det + longest side of track) are not computed and get distance 2.0. With the 1.5 GIoU-distance
    threshold, two disjoint boxes can only match when their center distance is at most about 3x the longest side, so this
    gate never excludes a pair that could be accepted; it only avoids the quadratic cost of never-dying tracks."""
    D = np.full((len(dets), len(tracks)), 2.0)
    if not len(dets) or not len(tracks): return D
    dc = np.array([[b.x, b.y, max(b.l, b.w)] for b in dets]); tc = np.array([[b.x, b.y, max(b.l, b.w)] for b in tracks])
    dist = np.hypot(dc[:, None, 0] - tc[None, :, 0], dc[:, None, 1] - tc[None, :, 1]); gate = 3 * (dc[:, None, 2] + tc[None, :, 2])
    for d, t in zip(*np.nonzero(dist <= gate)):
        D[d, t] = 1 - (U.iou3d(dets[d], tracks[t])[1] if asso == "iou" else U.giou3d(dets[d], tracks[t]))
    return D
A.compute_iou_distance = gated_iou_distance

WORK = os.environ.get("WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_vru"))
H = int(os.environ.get("MINHITS", 1))
CFG = yaml.safe_load(open(f"{IT}/configs/nu_configs/immortal.yaml"))
CFG["running"]["min_hits_to_birth"]["immortal"] = H

def run(args):
    src, seq = args
    tag = f"{src}_im{H}"; d = f"{WORK}/{seq}/{tag}"; os.makedirs(d, exist_ok=True)
    for f in ("detection_bbox_results.csv", "members.pkl", "detection"):
        s = f"{WORK}/{seq}/{src}/{f}"
        if os.path.exists(s) and not os.path.lexists(f"{d}/{f}"): os.symlink(s, f"{d}/{f}")
    det = pd.read_csv(f"{WORK}/{seq}/{src}/detection_bbox_results.csv"); det = det[det.label == 0]
    # Fix 2026-11-13: the official nms() drops boxes with a non-positive side (weird_bbox) but loops forever on a box whose
    # self-IoU is <= its threshold (a 0.7 mm x 6 mm KDE part in LUMPI c0300 frame 13). Same rule, extended to sides < 1 cm.
    det = det[det[["w", "l", "h"]].min(axis=1) >= 0.01]
    G = dict(tuple(det.groupby("frame_id")))
    trk = MOTModel(CFG); rows = []
    for f in range(int(det.frame_id.max()) + 1 if len(det) else 0):
        g = G.get(f)
        boxes, types = [], []
        if g is not None:
            for r in g.itertuples():
                b = BBox(x=r.x, y=r.y, z=r.z, h=r.h, w=r.w, l=r.l, o=r.yaw); b.s = float(r.confidence)
                boxes.append(b); types.append(1)
        if CFG["data_loader"]["nms"] and boxes:
            keep, types = nms(boxes, types, CFG["data_loader"]["nms_thres"]); boxes = [boxes[i] for i in keep]
        fd = FrameData(dets=[BBox.bbox2array(b) for b in boxes], ego=np.eye(4), time_stamp=f / 10.0, pc=None,
                       det_types=types, aux_info={"is_key_frame": True, "velos": None})
        for bb, tid, st, _ in trk.frame_mot(fd):
            if Validity.valid(st):
                rows.append((f, tid, 0, bb.s if bb.s is not None else 1.0, bb.x, bb.y, bb.z, bb.w, bb.l, bb.h, bb.o))
    pd.DataFrame(rows, columns=["frame_id", "track_id", "label", "confidence", "x", "y", "z", "w", "l", "h", "yaw"]).to_csv(
        f"{d}/Immortal_matched.csv", index=False)
    return seq

if __name__ == "__main__":
    src, seqs = sys.argv[1], sys.argv[2:]
    with Pool(int(os.environ.get("NPROC", 4))) as p:
        for s in p.imap_unordered(run, [(src, q) for q in seqs]): pass
    print("done", src, H, len(seqs))
