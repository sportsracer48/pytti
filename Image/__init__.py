import torch
from torch import nn
import numpy as np
from PIL import Image
from pytti import *

SUPPORTED_MODES = ['L','RGB','I','F']
FORMAT_SAMPLES  = {'L':1, 'RGB':3, 'I':1, 'F':1}

class DifferentiableImage(nn.Module):
  """
  Base class for defining differentiable images
  width:        (positive integer) image width in pixels
  height:       (positive integer) image height in pixels
  pixel_format: (string) PIL image mode. Either 'L','RGB','I', or 'F'
  """
  def __init__(self, width, height, pixel_format = 'RGB'):
    super().__init__()
    if pixel_format not in SUPPORTED_MODES:
      raise ValueError(f"Pixel format {pixel_format} is not supported.")
    self.image_shape   = (width, height)
    self.pixel_format = format
    self.output_axes  = ('x', 'y', 's')
    self.lr = 0.02

  def decode_training_tensor(self):
    """
    returns a decoded tensor of this image for training
    """
    return self.decode_tensor()

  def decode_tensor(self):
    """
    returns a decoded tensor of this image
    """
    raise NotImplementedError

  def encode_image(self, pil_image):
    """
    overwrites this image with the input image
    pil_image: (Image) input image
    """
    raise NotImplementedError

  def encode_random(self):
    """
    overwrites this image with random noise
    """
    raise NotImplementedError

  def update(self):
    """
    callback hook called once per training step by the optimizer
    """
    pass

  def decode_image(self):
    """
    render a PIL Image version of this image
    """
    tensor = self.decode_tensor()
    tensor = named_rearrange(tensor, self.output_axes, ('y', 'x', 's'))
    array = np.array(tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:,:,:]
    return Image.fromarray(array)

  def forward(self):
    """
    returns a decoded tensor of this image
    """
    if self.training:
      return self.decode_training_tensor()
    else:
      return self.decode_tensor()

class EMAImage(DifferentiableImage):
  """
  Base class for differentiable images with Exponential Moving Average filtering
  Based on code by Katherine Crowson
  """
  def __init__(self, width, height, tensor ,decay):
    super().__init__(width, height)
    self.tensor = nn.Parameter(tensor)
    self.register_buffer('biased', torch.zeros_like(tensor))
    self.register_buffer('average', torch.zeros_like(tensor))
    self.decay  = decay
    self.register_buffer('accum', torch.tensor(1.))
    self.update()

  @torch.no_grad()
  def update(self):
    if not self.training:
      raise RuntimeError('update() should only be called during training')
    self.accum *= self.decay
    self.biased.mul_(self.decay)
    self.biased.add_((1 - self.decay) * self.tensor)
    self.average.copy_(self.biased)
    self.average.div_(1 - self.accum)

  @torch.no_grad()
  def reset(self):
    if not self.training:
      raise RuntimeError('reset() should only be called during training')
    self.biased.set_(torch.zeros_like(self.biased))
    self.average.set_(torch.zeros_like(self.average))
    self.accum.set_(torch.ones_like(self.accum))
    self.update()

  def decode_training_tensor(self):
    return self.decode(self.tensor)

  def decode_tensor(self):
    return self.decode(self.average)

  def decode(self, tensor):
    raise NotImplementedError

from pytti.Image.PixelImage import PixelImage
from pytti.Image.RGBImage import RGBImage
from pytti.Image.VQGANImage import VQGANImage

