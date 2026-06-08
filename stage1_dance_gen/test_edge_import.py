"""Quick test: can we import and load the EDGE model?"""
import sys, os, types
EDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "external", "EDGE")
sys.path.insert(0, EDGE_DIR)
os.chdir(EDGE_DIR)

# stub wandb and p_tqdm
import importlib.machinery
wandb_stub = types.ModuleType("wandb")
wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
wandb_stub.init = lambda **kw: None
wandb_stub.log = lambda *a, **kw: None
class _FR:
    def finish(self): pass
wandb_stub.run = _FR()
sys.modules["wandb"] = wandb_stub

def _p_map(fn, it, **kw): return [fn(x) for x in it]
pt = types.ModuleType("p_tqdm"); pt.p_map = _p_map
sys.modules["p_tqdm"] = pt

# patch torch.load to accept old-style checkpoints
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

print("Importing EDGE...")
from EDGE import EDGE
print("  EDGE module imported OK")

ckpt = os.path.join(EDGE_DIR, "checkpoint.pt")
print(f"Checkpoint size: {os.path.getsize(ckpt)/1e6:.0f} MB")

print("Loading model (this takes ~30s)...")
model = EDGE("jukebox", ckpt)
model.eval()
print("  Model loaded and set to eval mode!")

print(f"  Device: {model.accelerator.device}")
print(f"  repr_dim: {model.repr_dim}")
print(f"  horizon: {model.horizon}")
print("EDGE import test PASSED!")
