from pytti import *
from pytti.Notebook import Rotoscoper
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import re, math
from CLIP import clip
import pytti
from PIL import Image
from pytti.Image import RGBImage
from collections import defaultdict
import numpy as np

def spherical_dist_loss(x, y):
  x = F.normalize(x, dim=-1)
  y = F.normalize(y, dim=-1)
  return x.sub(y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def make_mask(mask, thresh):
  if mask[0] == '[' and mask[-1] == ']':
    if mask[-5:-1] == '.mp4':
      #oh yeah this is tacked on
      return Rotoscoper(mask[1:-1])
    mask_fun = mask_image(mask[1:-1])
  else:
    mask_fun = MASK_DICT.get(mask,mask_semantic(mask))
  return lambda pos, size, emb: mask_fun(size,pos,emb,parametric_eval(thresh))

@torch.no_grad()
def mask_right(pos, size, emb, thresh = 0.5):
  cent = pos[...,0]+size[...,0]/2
  return cent.lt(thresh).float(), 1

@torch.no_grad()
def mask_left(pos, size, emb, thresh = 0.5):
  cent = pos[...,0]+size[...,0]/2
  return cent.gt(thresh).float(), 1

@torch.no_grad()
def mask_down(pos, size, emb, thresh = 0.5):
  cent = pos[...,1]+size[...,1]/2
  return cent.lt(thresh).float(), 1

@torch.no_grad()
def mask_near(pos, size, emb, thresh = 0.5):
  return size.min(dim = -1)[0].lt(thresh).float(), 1

@torch.no_grad()
def mask_far(pos, size, emb, thresh = 0.5):
  return size.min(dim = -1)[0].gt(thresh).float(), 1

@torch.no_grad()
def mask_up(pos, size, emb, thresh = 0.5):
  cent = pos[...,1]+size[...,1]/2
  return cent.gt(thresh).float(), 1

@torch.no_grad()
def mask_all(pos, size, emb, thresh = 0.5):
  return torch.zeros_like(size[...,0]).fill_(float('-inf')), 1

@torch.no_grad()
def mask_image(path, inverted = False, device = DEVICE):
  if isinstance(path, Image.Image):
    mask_pil = path
  else:
    if path[0] == '-':
      path = path[1:]
      inverted = True
    else:
      inverted = False
    mask_pil = Image.open(fetch(path)).convert('L')
  mask_tensor = TF.to_tensor(mask_pil).squeeze().to(device)
  mask_tensor = 1 - mask_tensor if inverted else mask_tensor
  mu = mask_tensor.mean()
  err_tensor = torch.zeros_like(mask_tensor)
  height, width = mask_tensor.shape[-2:]
  size_tensor = torch.as_tensor([width, height], device = device).view(1,1,-1)

  @torch.no_grad()
  def mask(pos, size, emb, thresh = 0.5):
    if mu < 0.001:
      #no use trying on a tiny mask I think
      return torch.zeros_like(size[...,0]).fill_(float('-inf')), 0 
    low, high = pos, pos + size
    low_px, high_px = (low * size_tensor).floor().long(), (high * size_tensor).floor().long()
    low_xs, low_ys = low_px[...,0].contiguous(), low_px[...,1].contiguous()
    high_xs, high_ys = high_px[...,0].contiguous(), high_px[...,1].contiguous()

    low_xs.clamp_(0,width-1)
    low_ys.clamp_(0,height-1)
    high_xs.clamp_(0,width-1)
    high_ys.clamp_(0,height-1)
    As = (high_xs - low_xs) * (high_ys - low_ys)
    ind = torch.argsort(As.view(-1)).flip([0])
    
    out = torch.zeros(*pos.shape[:-1], device = device)
    N = out.numel()
    for i in ind:
      low_x = low_xs.view(-1)[i]
      low_y = low_ys.view(-1)[i]
      high_x = high_xs.view(-1)[i]
      high_y = high_ys.view(-1)[i]
      A = (high_x - low_x) * (high_y - low_y)
      if A > 0:
        weight_sample, err_sample = mask_tensor[low_y:high_y,low_x:high_x], err_tensor[low_y:high_y,low_x:high_x]
        err_sample = err_sample.sign()*err_sample.abs().sqrt()
        weight = (weight_sample + err_sample).mean()
        err = mask_tensor[low_y:high_y,low_x:high_x] - weight
        err_tensor[low_y:high_y,low_x:high_x] += err.square().div(N).mul(err.sign())
      else:
        weight = torch.as_tensor(0,device = device)
      out.view(-1)[i] = weight
    #err_tensor.mul_(0.99)
    return torch.zeros_like(out).fill_(-math.inf), out/mu.sqrt()
  return mask

@torch.no_grad()
def mask_semantic(text, device = DEVICE):
  perceptors = pytti.Perceptor.CLIP_PERCEPTORS
  embeds = cat_with_pad([p.encode_text(clip.tokenize(text).to(device)).float() for p in perceptors])
  @torch.no_grad()
  def mask(pos, size, emb, thresh = 0.5):
    #that's right, it's ridiculous garbage!
    if thresh == 0.5000873264:
      return spherical_dist_loss(emb, embeds), 1
    else:
      thresh = thresh*0.3 + 0.7
      return spherical_dist_loss(emb, embeds).gt(thresh), 1
  return mask

MASK_DICT = {'a':mask_all, 'r':mask_right, 'l':mask_left, 'd':mask_down, 'u':mask_up, 'n':mask_near, 'f':mask_far}

@torch.no_grad()
def parse_prompt(embedder, prompt_string = "", pil_image=None, device = DEVICE):
  text, weight, stop = parse(prompt_string,r":(?![^\[]*\])",['', '1', '-inf'])
  weight, mask, cutoff = parse(weight,r"_(?![^\[]*\])",['1','a','0.5000873264']) #can you guess what this does?
  text = text.strip()
  mask = make_mask(mask.strip(), cutoff)
  if isinstance(mask, Rotoscoper):
    roto = mask
    mask = mask_all
  else:
    roto = None
  if text[0] == '[' and text[-1] == ']':
    pil_image = Image.open(fetch(text[1:-1].strip())).convert("RGB")
  if pil_image is not None:
    dummy = RGBImage(*pil_image.size)
    dummy.encode_image(pil_image)
    out = LocationAwareMCIP(*embedder(dummy), embedder, weight, stop, text, prompt_string, mask = mask)
  else:
    perceptors = pytti.Perceptor.CLIP_PERCEPTORS
    embeds = cat_with_pad([p.encode_text(clip.tokenize(text).to(device)).float() for p in perceptors])
    out = Prompt(embeds, weight, stop, text, prompt_string, mask = mask)
  if roto is not None:
    roto.target = out
    roto.update(0)
  return out

class Prompt(nn.Module):
  @torch.no_grad()
  def __init__(self, embeds, weight, stop, text, prompt_string, mask = mask_all):
    super().__init__()
    if embeds is not None:
      self.register_buffer('embeds',  embeds)
    self.weight = weight
    self.stop = stop
    self.input_axes = ('n', 'c', 'i')
    self.prompt_string = prompt_string
    self.text = text.encode('ascii','ignore').decode('ascii')
    self.text = (self.text[:20] + '..'+self.text[-5:]) if len(self.text) > 27 else self.text
    self.mask = mask
    self.enabled = True

  def __repr__(self):
    return self.prompt_string

  def __str__(self):
    return self.text

  def set_mask(self, pil_image, inverted = False):
    self.mask = mask_image(pil_image, inverted = inverted)

  def set_enabled(self, enabled):
    self.enabled = enabled

  def forward(self, embed, position, size, offset = 0.0, device = DEVICE):
    """
    input: (Tensor) input CLIP embedding
    returns the input's loss compared to the saved embedding
    """
    if not self.enabled or self.weight in ['0',0]:
      return torch.as_tensor(offset, device=device), offset
    dists_raw = spherical_dist_loss(embed, self.embeds) + offset
    weight = torch.as_tensor(parametric_eval(self.weight), device=device)
    stop   = torch.as_tensor(parametric_eval(self.stop),   device=device)

    mask_stops, mask_weights = self.mask(position, size, embed.detach())
    weight = torch.as_tensor(mask_weights, device = device) * weight
    sign_offset = weight.sign().clamp(max=0)

    dists = dists_raw*weight.sign()
    stops = torch.maximum(mask_stops+sign_offset, stop)
    dists = weight.abs() * replace_grad(dists, torch.maximum(dists, stops))
    return dists.mean(), dists_raw.mean()

class MultiClipImagePrompt(Prompt):
  @torch.no_grad()
  @vram_usage_mode('Image Prompts')
  def __init__(self, embeds, positions, sizes, embedder, weight, stop, text, prompt_string, mask = mask_all):
    self.input_axes = ('c', 'n', 'i')
    super().__init__(format_input(embeds,embedder,self), weight, stop, text+" (semantic)", prompt_string, mask = mask)
    self.input_axes = ('c', 'n', 'i')
    self.register_buffer('positions',format_input(positions,embedder,self))
    self.register_buffer('sizes'    ,format_input(sizes,embedder,self))
  
  @torch.no_grad()
  @vram_usage_mode('Image Prompts')
  def set_image(self, embedder, pil_image):
    width, height = pil_image.size
    img = RGBImage(width, height)
    img.encode_image(pil_image)
    embeds, positions, sizes = embedder(img)
    embeds = embeds.clone()
    self.positions.set_(format_input(positions,embedder,self))
    self.sizes.set_(format_input(sizes,embedder,self))
    self.embeds.set_(format_input(embeds,embedder,self))

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def minimize_average_distance(tensor_a, tensor_b, device=DEVICE):
  """
  tensor_a: pytorch tensor
  tensor_b: pytorch tensor
  returns: tensor of indicies in tensor_a which will minimize the euclidian distance between the elments of the two tensors
  """
  tensor_a = tensor_a.detach().cpu().numpy()
  tensor_b = tensor_b.detach().cpu().numpy()
  out = []
  for c in range(tensor_a.shape[0]):
    a = tensor_a[c,:,:]
    b = tensor_b[c,:,:]
    distances = cdist(a, b)
    row_ind, col_ind = linear_sum_assignment(distances)
    col_ind = torch.as_tensor(col_ind)
    out.append(col_ind)
  return out

class LocationAwareMCIP(MultiClipImagePrompt):
  def forward(self, embed, position, size):
    """
    input: (Tensor) input CLIP embedding
    returns the input's loss compared to the saved embedding
    """
    cent_a = self.positions + self.sizes/2
    cent_b = position + size/2
    indices = minimize_average_distance(cent_a, cent_b)
    embed = torch.stack([a[i] for a,i in zip(embed, indices)])
    position = torch.stack([a[i] for a,i in zip(position, indices)])
    size = torch.stack([a[i] for a,i in zip(size, indices)])
    return super().forward(embed, position, size, offset = 0.7)

