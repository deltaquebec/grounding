"""gridworld_audit.audit with numpy stand-in for torch agent"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
import gridworld_audit as g
from audit_common import RunContext

class StubSystem:
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        self.emb_dim, self.hidden_dim = 32, 64
        self.E = r.standard_normal((len(g.VOCAB), self.emb_dim)) * 0.5
        self.Wih = r.standard_normal((self.hidden_dim, self.emb_dim)) * 0.2
        self.Whh = r.standard_normal((self.hidden_dim, self.hidden_dim)) * 0.1
        self.W = r.standard_normal((2, self.hidden_dim)) * 0.3
        self.b = np.array([5.0, 5.0])
    def output(self, cmd, ablate=None, hidden_noise=None, embed_noise=None):
        embeds = np.stack([self.E[g.INDEX[w]] for w in cmd])
        if embed_noise is not None: embeds = embeds + embed_noise
        if ablate == "modifier-step": embeds = embeds[:1]
        elif ablate == "direction-embedding": embeds = embeds.copy(); embeds[1:] = 0
        h = np.zeros(self.hidden_dim)
        for x in embeds: h = np.tanh(self.Wih @ x + self.Whh @ h)
        if hidden_noise is not None: h = h + hidden_noise
        return self.W @ h + self.b
    def decoder_weight(self): return self.W

parser = g.build_parser()
argv = ["--noise-samples", "64", "--lipschitz", "1.0", "--eps-pres", "1.5", "--eta", "0.10"]
args = parser.parse_args(argv)
ctx = RunContext(Path("gridworld_audit.py"), parser, args, argv)
report = g.audit(ctx, args, StubSystem(), {"first_batch_loss": 1.0, "last_batch_loss": 0.5})
report["positions"] = {" ".join(c): [*map(float, g.intended(c)), *map(float, StubSystem().output(list(c)))] for c in g.COMMANDS}
ctx.write_json("report", report, prefix="data")
g.plot_modulus(ctx, report); g.plot_positions(ctx, report)
ctx.finish("completed")
