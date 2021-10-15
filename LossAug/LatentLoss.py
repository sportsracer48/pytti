from pytti.LossAug import MSELoss
import gc, torch, os, math
from pytti import DEVICE
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image
import copy, re
from pytti import *

class LatentLoss(MSELoss):
  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss", image_shape = None):
    super().__init__(comp, weight, stop, name, image_shape)
    self.pil_image = None
    self.has_latent = False

  @torch.no_grad()
  def set_target_image(self, pil_image, device = DEVICE):
    self.pil_image = pil_image
    self.has_latent = False
  
  @torch.no_grad()
  def set_latent(self, tensor):
    self.pil_image = None
    self.has_latent = True
    self.comp.set_(tensor)

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
    out = cls(comp, weight, stop, text+" (latent)", image_shape)
    out.set_target_image(pil_image)
    return out
    
  def get_loss(self, input, img):
    if not self.has_latent:
      dummy = copy.deepcopy(img)
      dummy.encode_image(self.pil_image)
      with torch.no_grad():
        self.comp.set_(dummy.get_latent_tensor(detach = True))
      self.has_latent = True
    return super().get_loss(img.get_latent_tensor() ,img)

  