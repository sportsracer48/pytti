from torch import nn
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from pytti import *

class Loss(nn.Module):
  def __init__(self, weight, stop, name):
    super().__init__()
    #self.register_buffer('weight', torch.as_tensor(weight))
    #self.register_buffer('stop', torch.as_tensor(stop))
    self.weight = weight
    self.stop = stop
    self.input_axes = ('n','s','y','x')
    self.name = name
    self.enabled = True

  def get_loss(self, input, img):
    raise NotImplementedError
  
  def set_enabled(self, enabled):
    self.enabled = enabled

  def set_weight(weight):
    self.weight = weight
  def set_stop(stop):
    self.stop = stop

  def __str__(self):
    return self.name

  def forward(self, input, img, device = DEVICE):
    if not self.enabled or self.weight in [0,'0']:
      return 0, 0

    weight = torch.as_tensor(parametric_eval(self.weight), device=device)
    stop   = torch.as_tensor(parametric_eval(self.stop),   device=device)
    loss_raw = self.get_loss(input, img)
    loss  =  loss_raw * weight.sign()
    return weight.abs() * replace_grad(loss, torch.maximum(loss, stop)), loss_raw

from pytti.LossAug.TVLoss import TVLoss
from pytti.LossAug.MSELoss import MSELoss
from pytti.LossAug.OpticalFlowLoss import OpticalFlowLoss, TargetFlowLoss
from pytti.LossAug.DepthLoss import DepthLoss
from pytti.LossAug.EdgeLoss import EdgeLoss
from pytti.LossAug.LatentLoss import LatentLoss
from pytti.LossAug.HSVLoss import HSVLoss
LOSS_DICT = {'edge':EdgeLoss, 'depth':DepthLoss}
