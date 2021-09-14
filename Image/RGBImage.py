from pytti import *
import torch
from torch import nn
from torchvision.transforms import functional as TF
from pytti.Image import DifferentiableImage
from PIL import Image
from torch.nn import functional as F 

class RGBImage(DifferentiableImage):
  """
  Naive RGB image representation
  """
  def __init__(self, width, height, scale=1, device=DEVICE):
    super().__init__(width*scale, height*scale)
    self.tensor = nn.Parameter(torch.zeros(3, height, width).to(device))
    self.output_axes = ('n', 's', 'y', 'x')
    self.scale = scale
  def decode_tensor(self):
    width, height = self.image_shape
    out = F.interpolate(self.tensor.unsqueeze(0), (height, width) , mode='nearest')
    return clamp_with_grad(out,0,1)
  
  def get_image_tensor(self):
    return self.tensor

  @torch.no_grad()
  def set_image_tensor(self, tensor):
    self.tensor.set_(tensor)

  @torch.no_grad()
  def encode_image(self, pil_image, device=DEVICE, **kwargs):
    width, height = self.image_shape
    scale = self.scale
    pil_image = pil_image.resize((width//scale, height//scale), Image.LANCZOS)
    self.tensor.set_(TF.to_tensor(pil_image).to(device))

  @torch.no_grad()
  def encode_random(self):
    self.tensor.uniform_()
