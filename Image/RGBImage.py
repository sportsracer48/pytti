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
    self.tensor = nn.Parameter(torch.zeros(1, 3, height, width).to(device = device, memory_format = torch.channels_last))
    self.output_axes = ('n', 's', 'y', 'x')
    self.scale = scale
  def decode_tensor(self):
    width, height = self.image_shape
    out = F.interpolate(self.tensor, (height, width) , mode='nearest')
    return clamp_with_grad(out,0,1)

  def clone(self):
    width, height = self.image_shape
    dummy = RGBImage(width//self.scale, height//self.scale, self.scale)
    with torch.no_grad():
      dummy.tensor.set_(self.tensor.clone())
    return dummy
  
  def get_image_tensor(self):
    return self.tensor.squeeze(0)

  @torch.no_grad()
  def set_image_tensor(self, tensor):
    self.tensor.set_(tensor.unsqueeze(0))

  @torch.no_grad()
  def encode_image(self, pil_image, device=DEVICE, **kwargs):
    width, height = self.image_shape
    scale = self.scale
    pil_image = pil_image.resize((width//scale, height//scale), Image.LANCZOS)
    self.tensor.set_(TF.to_tensor(pil_image).unsqueeze(0).to(device, memory_format = torch.channels_last))

  @torch.no_grad()
  def encode_random(self):
    self.tensor.uniform_()
