"""Fine-tune nuScenes VoxelNeXt on the V2X-Real infrastructure train split (ft_prep.py) - the site-specific,
label-supervised upper bound. Same network and class heads as the zero-shot detector; augmentation = flip, rotation,
scaling, translation (no nuScenes gt_sampling database). AdamW, cosine LR, EPOCHS epochs; checkpoint per epoch."""
import os, pickle, sys, time, math
from pathlib import Path
import numpy as np, torch
if not hasattr(np, "int"): np.int = int   # old OpenPCDet code (base_bev_backbone) uses the removed alias
OPC = Path(os.environ.get("OPENPCDET", str(Path(__file__).resolve().parents[1] / "third_party" / "OpenPCDet"))); sys.path.insert(0, str(OPC))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
HERE = Path(__file__).resolve().parent; DATA = Path(os.environ.get("FT_DATA", HERE / "ft_data"))   # cloud runs set FT_DATA / CKPT_DIR / OUT_DIR
ARCH = os.environ.get("ARCH", "voxelnext")   # voxelnext | centerpoint (dynamic pillars, dense 2D backbone)
CFGS = {"voxelnext": ("cbgs_voxel0075_voxelnext", "cbgs_voxel0075_voxelnext_nds6665.pth"), "centerpoint": ("cbgs_dyn_pp_centerpoint", "cbgs_dyn_pp_centerpoint_nds6070.pth")}
CKPT = f"{os.environ.get('CKPT_DIR', str(Path(__file__).resolve().parents[1] / 'checkpoints'))}/{CFGS[ARCH][1]}"
OUT = Path(os.environ.get("OUT_DIR", HERE)) / f"ft_ckpt_{ARCH}"; OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.environ.get("EPOCHS", 8)); BS = int(os.environ.get("BS", 4)); LR = float(os.environ.get("LR", 1e-4)); MAXIT = int(os.environ.get("MAXIT", 0))

class V2X(DatasetTemplate):
    def __init__(self, dcfg, names, logger):
        super().__init__(dataset_cfg=dcfg, class_names=names, training=True, root_path=HERE, logger=logger)
        self.infos = pickle.load(open(DATA / "infos_train.pkl", "rb"))
    def __len__(self): return len(self.infos)
    def __getitem__(self, i):
        f = self.infos[i]
        p = np.load(DATA / f["lidar_path"].split("/ft_data/", 1)[1]); r = np.array(self.point_cloud_range); e = 0.25   # strictly inside the grid (max edge is inclusive)
        p = p[np.all((p[:, :3] > r[:3] + e) & (p[:, :3] < r[3:] - e), 1)]
        b = f["gt_boxes"]; m = np.all((b[:, :3] > r[:3] + 1.0) & (b[:, :3] < r[3:] - 1.0), 1)   # box centres off the head's map edge
        return self.prepare_data(dict(points=p, frame_id=f["frame_id"], gt_boxes=b[m].copy(), gt_names=f["gt_names"][m].copy()))

def config():
    cwd = os.getcwd(); os.chdir(OPC / "tools")
    try: cfg_from_yaml_file(f"cfgs/nuscenes_models/{CFGS[ARCH][0]}.yaml", cfg)
    finally: os.chdir(cwd)
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST = [a for a in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST if a.NAME != "gt_sampling" and not os.environ.get("NOAUG")]
    return cfg

if __name__ == "__main__":
    cfg = config(); logger = common_utils.create_logger(str(OUT / "train.log"))
    ds = V2X(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger)
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=int(os.environ.get("NW", 8)), collate_fn=ds.collate_batch, drop_last=True, pin_memory=True)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    if not os.environ.get("SCRATCH"): model.load_params_from_file(filename=CKPT, logger=logger, to_cpu=True)
    nc = [n for n, q in model.named_parameters() if not q.is_contiguous()]
    for q in model.parameters(): q.data = q.data.contiguous()   # the loader permutes old spconv weights into non-contiguous views
    logger.info(f"made {len(nc)} non-contiguous parameters contiguous"); model.cuda().train()
    import spconv.pytorch.ops as sops   # spconv 2.3.6 backward kernels assume a contiguous output gradient
    if os.environ.get("NO_DIRECT_TABLE"):
        fn = sops.get_indice_pairs_implicit_gemm; names = fn.__code__.co_varnames[:fn.__code__.co_argcount]; d = list(fn.__defaults__)
        d[names[len(names) - len(d):].index("direct_table")] = False; fn.__defaults__ = tuple(d)
    for fn in ("indice_conv_backward", "implicit_gemm_backward"):
        def wrap(orig):
            def f(features, filters, grad, *a, **k): return orig(features, filters, grad.contiguous(), *a, **k)
            return f
        setattr(sops, fn, wrap(getattr(sops, fn)))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); total = EPOCHS * len(dl); it = 0
    for ep in range(EPOCHS):
        t0 = time.time(); run = []
        for batch in dl:
            for g in opt.param_groups: g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * it / total))
            load_data_to_gpu(batch); ret, tb, _ = model(batch); loss = ret["loss"].mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step()
            run.append(loss.item()); it += 1
            if os.environ.get("VERBOSE"): print(it, f"{loss.item():.3f}", batch["points"].shape[0], flush=True)
            if it % 100 == 0: logger.info(f"ep {ep} it {it}/{total} loss {np.mean(run[-100:]):.3f} {(time.time() - t0) / len(run):.2f}s/it")
            if MAXIT and it >= MAXIT: sys.exit(0)
        torch.save(dict(model_state=model.state_dict(), epoch=ep + 1), OUT / f"ep{ep + 1}.pth")
        logger.info(f"epoch {ep + 1} done, loss {np.mean(run):.3f}, {time.time() - t0:.0f}s")
