"""Train independent EA2 supervision from frozen collected maps."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from ..config import SLConfig,SL_DATA_DIR,run_dir
from ..dataset import load_map,SLDataset
from ..train import train,save_checkpoint
from ..provenance import training_provenance


def main():
    ap=argparse.ArgumentParser(description=__doc__);defaults=SLConfig()
    ap.add_argument("--data-dir",type=Path,default=Path(SL_DATA_DIR))
    ap.add_argument("--seeds",default=None)
    ap.add_argument("--run-name",default="baseline")
    ap.add_argument("--out-dir",type=Path,default=None)
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--seed",type=int,default=1)
    for flag,default,typ in [('epochs',defaults.train.epochs,int),('batch-size',defaults.train.batch_size,int),
                             ('seq-len',defaults.train.seq_len,int),('window-stride',defaults.train.window_stride,int),
                             ('lr',defaults.train.learning_rate,float),('aux-beta',defaults.train.aux_beta,float),
                             ('safe-lambda',defaults.train.safe_lambda,float)]:
        ap.add_argument('--'+flag,type=typ,default=default)
    ap.add_argument("--aux-ks",default="")
    ap.add_argument("--aux-mode",choices=['recall','forward'],default='recall')
    args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(1)
    cfg=SLConfig();cfg.train.epochs=args.epochs;cfg.train.batch_size=args.batch_size
    cfg.train.seq_len=args.seq_len;cfg.train.window_stride=args.window_stride
    cfg.train.learning_rate=args.lr;cfg.train.aux_beta=args.aux_beta;cfg.train.safe_lambda=args.safe_lambda
    cfg.model.aux_ks=[int(k) for k in args.aux_ks.split(',') if k];cfg.model.aux_mode=args.aux_mode
    seeds=[int(s) for s in args.seeds.split(',')] if args.seeds else sorted(int(p.stem.removeprefix('map_seed')) for p in args.data_dir.glob('map_seed*.pt'))
    files=[args.data_dir/f'map_seed{s}.pt' for s in seeds]
    maps=[load_map(str(p)) for p in files];dataset=SLDataset(cfg,maps)
    out=args.out_dir or Path(run_dir(args.run_name));out.mkdir(parents=True,exist_ok=True)
    if (out/"model.pt").exists():
        raise FileExistsError(f"Preserving existing checkpoint: {out}; choose a new run")
    provenance=training_provenance(cfg,maps,files,dataset,args.seed)
    (out/'sl_config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'TRAIN windows={len(dataset)} train={len(dataset.train_idx)} val={len(dataset.val_idx)}',flush=True)
    result=train(cfg,dataset,device=args.device,log_every=1)
    save_checkpoint(result,cfg,str(out/'model.pt'),provenance=provenance)
    with torch.no_grad():
        probe=dataset.batch(dataset.val_idx[:2]).transpose(0,1).to(args.device)
        torch.save({'observation':probe.cpu(),'prediction':result['net'](probe).cpu()},out/'reload_probe.pt')
    print(f'TRAIN SAVED {out / "model.pt"}',flush=True)


if __name__=='__main__':main()
