#!/usr/bin/env python3
"""Pose-dependent collision sweeps over one angular period, independent of URDF stops."""
from pathlib import Path
import argparse,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter,JOINT_NAMES,DEFAULT_XML
from morphology.provenance import identity,write_new


def sweep(meter,j,step):
    out={};q=np.zeros(18)
    for sign in (-1,1):
        angle=0.;last=0.;hit=None;n=0
        for angle in np.r_[np.arange(step,np.pi,step),np.pi]:
            angle=min(float(angle),np.pi)*sign;q[j]=angle;meter.set_state((0,0,.8),(1,0,0,0),q);pairs=meter.self_collision();n+=1
            if pairs:hit={'angle':angle,'pairs':pairs};break
            last=angle
        out[str(sign)]={'last_collision_free_angle':last,'first_collision':hit,'samples':n,'search_boundary_reached':hit is None}
    return {'q0_local_free_interval':[out['-1']['last_collision_free_angle'],out['1']['last_collision_free_angle']],
            'directions':out,'validity':'only other joints fixed at zero; not a global mechanical interval'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--xml',type=Path,default=DEFAULT_XML);ap.add_argument('--step',type=float,default=.005);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if not 0<a.step<.1:raise ValueError('positive fine scan step required')
    m=MorphologyMeter(a.xml,floor=False);joints={n:sweep(m,i,a.step) for i,n in enumerate(JOINT_NAMES)}
    for n,r in joints.items():print(n,r['q0_local_free_interval'])
    write_new(a.out,{'provenance':identity(a.xml,{'step':a.step,'joint_domain':[-np.pi,np.pi],'URDF_stops_disabled':True,'collision_scope':'configured nonadjacent bodies; intended attachment exclusions retained; physical adjacent-joint stops unknown'}),'joints':joints,'declared_URDF_model_ranges':m.declared_joint_ranges.tolist()})
    return 0

if __name__=='__main__':sys.exit(main())
