from pytti import *
from torch import nn
from torch.nn import functional as F
import re
from torch.nn import functional as F
from CLIP import clip
from pytti.Perceptor import CLIP_PERCEPTORS

def spherical_dist_loss(x, y):
  x = F.normalize(x, dim=-1)
  y = F.normalize(y, dim=-1)
  return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

class MultiClipPrompt(nn.Module):
  """
  Compares CLIP embeddings (text or image) to saved embeddings for multiple CLIP models simultaneously
  based on VQGAN+CLIP system by Katherine Crowson (https://github.com/crowsonkb)
  embed:  (Tensor) CLIP embeddings
  text:   (string) text representation
  weight: (nonzero float) overall strength of prompt. Negative weights negate the prompt
  stop:   (float in [-1,1], sign(stop) == sign(weight)) minimum comparison distance in CLIP embedding space
          regardless of sign, lesser stop values make the optimizer greedier and greater stop values make the optimizer lazier
          sign must match weight, so stop is in [-1,0) if weight < 0, or stop is in [0,1) if weight > 0
  """
  def __init__(self, prompt_string, perceptors=CLIP_PERCEPTORS, device=DEVICE):
    super().__init__()
    tokens = re.split(':', prompt_string, 2)
    tokens = tokens + ['', '1', '-inf'][len(tokens):]
    text, weight, stop = tokens
    text   = text.strip()
    weight = float(weight.strip())
    stop   = float(stop.strip())
    embeds = cat_with_pad([p.encode_text(clip.tokenize(text).to(device)).float() for p in perceptors])
    self.register_buffer('embeds',  embeds)
    self.register_buffer('weight', torch.as_tensor(weight))
    self.register_buffer('stop',   torch.as_tensor(stop))
    self.input_axes = ('n', 'c', 'i')
    self.txt = prompt_string

  def __repr__(self):
    return self.txt

  def __str__(self):
    return self.txt

  def forward(self, input):
    """
    input: (Tensor) input CLIP embedding
    returns the input's loss compared to the saved embedding
    """
    dists = spherical_dist_loss(input, self.embeds)
    dists = dists * self.weight.sign()
    dists = self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop))
    return dists.mean()
