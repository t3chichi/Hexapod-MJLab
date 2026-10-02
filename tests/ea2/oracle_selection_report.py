"""Summarise frozen evidence without selecting frames or rerunning a solver."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('logs/ea2/oracle_refinement/20261003')
TAGS={1:'axial_reserve015_v18',7:'frozen_axial015_v19',89:'heldout_axial015_v19',101:'heldout_axial015_v19'}


def main():
    audit=ROOT/'derived_audit_v19_complete.json'
    records=json.loads(audit.read_text())['records']
    selected=[];sensitivity=[]
    for seed,tag in TAGS.items():
        name=f'lookahead_w4_seed{seed}_m0.10_stride1_{tag}'
        record=next(r for r in records if r['run']==name)
        for suffix,key in [('.json','json_sha256'),('.npz','npz_sha256')]:
            assert hashlib.sha256((ROOT/(name+suffix)).read_bytes()).hexdigest()==record[key]
        selected.append(record)
        tag2='frozen_axial015_v19' if seed==1 else tag
        sensitivity.append(next(r for r in records if r['run']==f'lookahead_w4_seed{seed}_m0.20_stride1_{tag2}'))
    summary={'protocol':'All current-geometrically-feasible adjacent pairs, including emergencies and unsafe legacy predictions; shape on same legacy-safe frames. No emergency masking in benefit comparisons.',
             'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
             'primary_margin_m':.1,'selected':selected,'sensitivity_margin_m':.2,'sensitivity':sensitivity,
             'limitations':['local shape optimum only','current full polygon certificate, no swept volume certificate',
                            'three explicit emergencies, masked for supervision only',
                            'heldout seed101 worst jump is worse than legacy',
                            '.20 sensitivity is not uniformly better than .10 production baseline',
                            'teacher geometry does not certify student predictions or physical robot feasibility']}
    path=ROOT/'selection_report_v1.json'
    if path.exists():
        assert json.loads(path.read_text()) == summary, "Preserving differing earlier summary"
    else:
        path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    labels=[f'seed {r["seed"]}' for r in selected];x=np.arange(4)
    old=[100*r['legacy_safe_all_frames']/r['frames'] for r in selected]
    new=[100*r['candidate_valid']/r['frames'] for r in selected]
    axes[0,0].bar(x-.17,old,.34,label='production legacy',color='#999999')
    axes[0,0].bar(x+.17,new,.34,label='selected teacher',color='#0072B2')
    axes[0,0].set(ylabel='All-frame fraction (%)',title='Whole-polygon clearance >= 0.10 m',ylim=(0,105))
    axes[0,0].legend()
    for ax,values,title in [
        (axes[0,1],[r['relative_to_production_legacy_percent']['tv_percent'] for r in selected],'Total variation change'),
        (axes[1,0],[r['relative_to_production_legacy_percent']['acceleration']['mean'] for r in selected],'Mean acceleration change'),
        (axes[1,1],[r['relative_to_production_legacy_percent']['jump']['max'] for r in selected],'Worst jump change (tradeoff retained)'),
    ]:
        ax.bar(x,values,color=['#0072B2' if v<=0 else '#D55E00' for v in values])
        ax.axhline(0,color='black',lw=.8);ax.set(title=title,ylabel='Change from legacy (%)')
        for i,v in enumerate(values):ax.text(i,v,f'{v:+.1f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
    for ax in axes.flat:ax.set(xticks=x,xticklabels=labels);ax.grid(axis='y',alpha=.2)
    fig.suptitle('EA2 selected teacher: frozen two-map + independent-map audit')
    fig.supxlabel('Negative change is better. MIN-infeasible frames remain in safety denominators.\nEmergency frames remain in trajectory comparisons.', fontsize=9)
    out=Path('docs/figures');out.mkdir(exist_ok=True)
    fig.savefig(out/'ea2_oracle_selection_20261003.png',dpi=160);plt.close(fig)
    print('SELECTION REPORT PASS',path)
    for r in selected:print(r['seed'],r['candidate_valid'],r['legacy_safe_shape'],r['relative_to_production_legacy_percent'])


if __name__=='__main__':main()
