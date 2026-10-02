"""Export the inference GRU/MLP with explicit hidden state and reset mask."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from ..evaluate import load_checkpoint, checkpoint_teacher


class StatefulInference(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.gru = net.gru
        self.mlp = net.mlp

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, reset: torch.Tensor):
        # Reset BEFORE consuming the first observation of the next episode.
        hidden = torch.where(reset[None, :, None], torch.zeros_like(hidden), hidden)
        output, new_hidden = self.gru(obs.unsqueeze(0), hidden)
        return self.mlp(output[0]).clamp(0., 1.), new_hidden


def export(path, output):
    net, meta = load_checkpoint(str(path))
    module = StatefulInference(net).eval()
    scripted = torch.jit.script(module)
    hidden = net.init_hidden(3)
    torch.manual_seed(13)
    for i in range(15):
        obs = torch.rand(3, net.cfg.obs_dim)
        reset = torch.tensor([i == 4, i == 8, False])
        a, ha = module(obs, hidden, reset)
        b, hb = scripted(obs, hidden, reset)
        assert torch.allclose(a, b, atol=1e-6, rtol=0)
        assert torch.allclose(ha, hb, atol=1e-6, rtol=0)
        reference, _ = net.step(obs[None], torch.where(reset[None,:,None], 0., hidden))
        assert torch.allclose(a, reference.clamp(0,1), atol=1e-6, rtol=0)
        hidden = ha
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    restored = torch.jit.load(str(output))
    a, ha = module(obs, hidden, reset); b, hb = restored(obs, hidden, reset)
    assert torch.equal(a,b) and torch.equal(ha,hb)
    metadata = {"teacher": checkpoint_teacher(meta), "obs_dim": net.cfg.obs_dim,
                "shape_dim": net.cfg.action_dim, "hidden_shape": [net.cfg.rnn_num_layers, "batch", net.cfg.rnn_hidden_dim],
                "interface": "obs(B,190), hidden(layers,B,187), reset(B) -> s(B,5), hidden",
                "max_parity_tolerance": 1e-6, "auxiliary_heads_exported": False,
                "prediction_safety_certified": False}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'EXPORT PASS {output}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export(args.checkpoint,args.out or args.checkpoint.parent/'inference.pt')


if __name__=='__main__':main()
