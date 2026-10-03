"""Immutable experiment directories and exact input/source provenance."""
from pathlib import Path
import hashlib,json,sys
import mujoco,numpy,scipy

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(xml,config,inputs=()):
    xml=Path(xml).resolve();assets=xml.parent.parent/'meshes'
    return {'schema_version':2,'argv':sys.argv,'config':config,
            'sources_sha256':{p.name:sha(p) for p in sorted(TOOLS.glob('*.py'))},
            'model':{'xml':str(xml),'xml_sha256':sha(xml),'meshes_sha256':{p.name:sha(p) for p in sorted(assets.glob('*.STL'))}},
            'inputs_sha256':{str(Path(p).resolve()):sha(p) for p in inputs},
            'dependencies':{'python':sys.version,'mujoco':mujoco.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__}}


def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def model_content_matches(first,second):
    # Paths document original locations; byte identity determines replay compatibility.
    return first['xml_sha256']==second['xml_sha256'] and first['meshes_sha256']==second['meshes_sha256']
