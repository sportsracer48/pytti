from pytti.LossAug import MSELoss
import gc, torch, os, math
from pytti import DEVICE
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image
import copy, re
from pytti import *

class LatentLoss(MSELoss):
  @torch.no_grad()
  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss", image_shape = None):
    super().__init__(comp, weight, stop, name, image_shape)
    self.pil_image = None
    self.has_latent = False
    w, h = image_shape
    self.direct_loss = MSELoss(TF.resize(comp.clone(), (h,w)), weight, stop, name, image_shape)

  @torch.no_grad()
  def set_comp(self, pil_image, device = DEVICE):
    self.pil_image = pil_image
    self.has_latent = False
    self.direct_loss.set_comp(pil_image.resize(self.image_shape, Image.LANCZOS))

  @classmethod
  @vram_usage_mode('Latent Image Loss')
  @torch.no_grad()
  def TargetImage(cls, prompt_string, image_shape, pil_image = None, is_path = False, device = DEVICE):
    text, weight, stop = parse(prompt_string, r"(?<!^http)(?<!s):|:(?!/)" ,['', '1', '-inf'])
    weight, mask = parse(weight,r"_",['1','']) 
    text = text.strip()
    mask = mask.strip()
    if pil_image is None and text != '' and is_path:
      pil_image = Image.open(fetch(text)).convert("RGB")
    comp = MSELoss.make_comp(pil_image) if pil_image is not None else torch.zeros(1,1,1,1, device = device)
    out = cls(comp, weight, stop, text+" (latent)", image_shape)
    if pil_image is not None:
      out.set_comp(pil_image)
    out.set_mask(mask)
    return out
  
  def set_mask(self, mask):
    self.direct_loss.set_mask(mask)
    super().set_mask(mask)

  def get_loss(self, input, img):
    if not self.has_latent:
      latent = img.make_latent(self.pil_image)
      with torch.no_grad():
        self.comp.set_(latent.clone())
      self.has_latent = True
    l1 = super().get_loss(img.get_latent_tensor() ,img)/2
    l2 = self.direct_loss.get_loss(input, img)/10
    return l1 + l2
  
