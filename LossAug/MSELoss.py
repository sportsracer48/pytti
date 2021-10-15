import math, re
from PIL import Image
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from pytti.LossAug import Loss
from pytti import *
import torch

class MSELoss(Loss):
  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss", image_shape = None):
    super().__init__(weight, stop, name)
    self.register_buffer('comp', comp)
    if image_shape is None:
      height, width = comp.shape[-2:]
      image_shape = (width, height)
    self.image_shape = image_shape
    self.register_buffer('mask', torch.ones_like(comp))
    self.use_mask = False

  @classmethod
  def TargetImage(cls, prompt_string, pil_image = None, image_shape = None, device = DEVICE):
    tokens = re.split('(?<!^http)(?<!s):|:(?!//)', prompt_string, 2)
    tokens = tokens + ['', '1', '-inf'][len(tokens):]
    text, weight, stop = tokens
    text = text.strip()
    if pil_image is None and text != '':
      pil_image = Image.open(fetch(text)).convert("RGB")
      im = pil_image.resize(image_shape, Image.LANCZOS)
      comp = TF.to_tensor(im).unsqueeze(0).to(device)
    elif pil_image is None:
      comp = torch.zeros(3,image_shape[1],image_shape[0]).unsqueeze(0).to(device)
    else:
      im = pil_image.resize(image_shape, Image.LANCZOS)
      comp = cls.make_comp(im)
    if image_shape is None:
      image_shape = pil_image.size
    return cls(comp, weight, stop, text+" (direct)", image_shape)

  @torch.no_grad()
  def set_mask(self, mask, inverted = False):
    if mask is not None:
      self.mask.set_(mask if not inverted else (1-mask))
    self.use_mask = mask is not None

  @classmethod
  def make_comp(cls, pil_image, device=DEVICE):
    return TF.to_tensor(pil_image).to(device)

  def set_comp(self, pil_image, device=DEVICE):
    self.comp.set_(self.make_comp(pil_image))
  def get_loss(self, input, img):
    if self.use_mask:
      if self.mask.shape[-2:] != input.shape[-2:]:
        mask = TF.resize(self.mask, input.shape[-2:])
        self.set_mask(mask)
      return F.mse_loss(input*self.mask, self.comp*self.mask)
    else:
      return F.mse_loss(input, self.comp)



      