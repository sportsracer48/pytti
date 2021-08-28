import torch
from CLIP import clip

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLIP_PERCEPTORS = None

def init_clip(clip_models):
  global CLIP_PERCEPTORS
  if CLIP_PERCEPTORS is None:
    CLIP_PERCEPTORS = [clip.load(model, jit=False)[0].eval().requires_grad_(False).to(DEVICE) for model in clip_models]

def free_clip():
  global CLIP_PERCEPTORS
  CLIP_PERCEPTORS = None
