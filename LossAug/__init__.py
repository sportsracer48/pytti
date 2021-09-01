from torch.nn import functional as F
from torch import nn
import torch, math, re
from pytti import *
from PIL import Image
from torchvision.transforms import functional as TF

def tv_loss(input):
  """L2 total variation loss, as in Mahendran et al."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
  y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
  return (x_diff**2 + y_diff**2).mean([1, 2, 3])

class Loss(nn.Module):
  def __init__(self, weight, stop, name):
    super().__init__()
    self.register_buffer('weight', torch.as_tensor(weight))
    self.register_buffer('stop', torch.as_tensor(stop))
    self.input_axes = ('n','s','y','x')
    self.name = name

  def get_loss(self, input):
    raise NotImplementedError

  def __str__(self):
    return self.name

  def forward(self, input):
    loss  = self.get_loss(input) * self.weight.sign()
    return self.weight.abs() * replace_grad(loss, torch.maximum(loss, self.stop))


class TV_Loss(Loss):
  def __init__(self, weight = 0.15, stop = -math.inf, name = "smoothing loss (TV)"):
    super().__init__(weight, stop, name)
  def get_loss(self, input):
    return tv_loss(input)
  
class MSE_Loss(Loss):
  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss"):
    super().__init__(weight, stop, name)
    self.register_buffer('comp', comp)
  @classmethod
  def TargetImage(cls, prompt_string, pil_image = None, image_shape = None, device = DEVICE):
    tokens = re.split('(?<!^http)(?<!s):|:(?!//)', prompt_string, 2)
    tokens = tokens + ['', '1', '-inf'][len(tokens):]
    text, weight, stop = tokens
    text = text.strip()
    if pil_image is None:
      pil_image = Image.open(fetch(text)).convert("RGB")
    weight = float(weight.strip())
    stop   = float(stop.strip())
    if image_shape is None:
      image_shape = pil_image.size
    im = pil_image.resize(image_shape, Image.LANCZOS)
    return cls(TF.to_tensor(im).unsqueeze(0).to(device), weight, stop, text+" (direct)")
  def get_loss(self, input):
    return F.mse_loss(input, self.comp)