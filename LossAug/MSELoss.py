import math, re
from PIL import Image
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from pytti.LossAug import Loss
from pytti.Notebook import Rotoscoper
from pytti import *
import torch

class MSELoss(Loss):
  @torch.no_grad()
  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss", image_shape = None, device = DEVICE):
    super().__init__(weight, stop, name)
    self.register_buffer('comp', comp)
    if image_shape is None:
      height, width = comp.shape[-2:]
      image_shape = (width, height)
    self.image_shape = image_shape
    self.register_buffer('mask', torch.ones(1,1,1,1, device = device))
    self.use_mask = False

  @classmethod
  @vram_usage_mode('Loss Augs')
  @torch.no_grad()
  def TargetImage(cls, prompt_string, image_shape, pil_image = None, is_path = False, device = DEVICE):
    text, weight, stop = parse(prompt_string, r"(?<!^http)(?<!s):|:(?!/)" ,['', '1', '-inf'])
    weight, mask = parse(weight,r"_",['1','']) 
    text = text.strip()
    mask = mask.strip()
    if pil_image is None and text != '' and is_path:
      pil_image = Image.open(fetch(text)).convert("RGB")
      im = pil_image.resize(image_shape, Image.LANCZOS)
      comp = cls.make_comp(im)
    elif pil_image is None:
      comp = torch.zeros(1,1,1,1, device = device)
    else:
      im = pil_image.resize(image_shape, Image.LANCZOS)
      comp = cls.make_comp(im)
    if image_shape is None:
      image_shape = pil_image.size
    out = cls(comp, weight, stop, text+" (direct)", image_shape)
    out.set_mask(mask)
    return out

  @torch.no_grad()
  def set_mask(self, mask, inverted = False, device = DEVICE):
    if isinstance(mask, str) and mask != '':
      if mask[0] == '-':
        mask = mask[1:]
        inverted = True
      if mask.strip()[-4:] == '.mp4':
        r = Rotoscoper(mask,self)
        r.update(0)
        return
      mask = Image.open(fetch(mask)).convert('L')
    if isinstance(mask, Image.Image):
      with vram_usage_mode('Masks'):
        mask = TF.to_tensor(mask).unsqueeze(0).to(device, memory_format = torch.channels_last)
    if mask not in ['',None]:
      self.mask.set_(mask if not inverted else (1-mask))
    self.use_mask = mask not in ['',None]

  @classmethod
  def convert_input(cls, input, img):
    return input

  @classmethod
  def make_comp(cls, pil_image, device=DEVICE):
    out = TF.to_tensor(pil_image).unsqueeze(0).to(device, memory_format = torch.channels_last)
    return cls.convert_input(out, None)

  def set_comp(self, pil_image, device=DEVICE):
    self.comp.set_(type(self).make_comp(pil_image))
  def get_loss(self, input, img):
    input = type(self).convert_input(input, img)
    if self.use_mask:
      if self.mask.shape[-2:] != input.shape[-2:]:
        with torch.no_grad():
          mask = TF.resize(self.mask, input.shape[-2:])
          self.set_mask(mask)
      return F.mse_loss(input*self.mask, self.comp*self.mask)
    else:
      return F.mse_loss(input, self.comp)



      