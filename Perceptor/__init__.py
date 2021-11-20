import torch
from CLIP import clip
from pytti import DEVICE, vram_usage_mode

CLIP_PERCEPTORS = None

@vram_usage_mode('CLIP')
def init_clip(clip_models):
  global CLIP_PERCEPTORS
  if CLIP_PERCEPTORS is None:
    CLIP_PERCEPTORS = [clip.load(model, jit=False)[0].eval().requires_grad_(False).to(DEVICE, memory_format=torch.channels_last) for model in clip_models]

def free_clip():
  global CLIP_PERCEPTORS
  CLIP_PERCEPTORS = None
