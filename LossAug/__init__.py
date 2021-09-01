from torch.nn import functional as F
from torch import nn
import torch

def tv_loss(input):
  """L2 total variation loss, as in Mahendran et al."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
  y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
  return (x_diff**2 + y_diff**2).mean([1, 2, 3])

class TV_Loss(nn.Module):
  def __init__(self, weight = 0.15):
    super().__init__()
    self.register_buffer('weight', torch.as_tensor(weight))
    self.input_axes = ('n','s','y','x')
  def forward(self, input):
    return self.weight*tv_loss(input)
  def __str__(self):
    return "TV LOSS"

class MSE_Loss(nn.Module):
  def __init__(self, comp, weight= 0.5):
    super().__init__()
    self.register_buffer('comp', comp)
    self.register_buffer('weight', torch.as_tensor(weight))
    self.input_axes = ('n','s','y','x')
  def forward(self, input):
    return F.mse_loss(input, self.comp)*self.weight
  def __str__(self):
    return "MSE LOSS"