from pytti.LossAug import MSELoss
import gc, torch, os, math
from pytti import DEVICE, vram_usage_mode
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image

class EdgeLoss(MSELoss):

  @classmethod
  def convert_input(cls, input, img):
    return EdgeLoss.get_edges(input)

  @staticmethod
  def get_edges(tensor, device = DEVICE):
    tensor = TF.rgb_to_grayscale(tensor)
    dx_ker = torch.tensor([[[[1,0,-1],[2,0,-2],[ 1, 0,-1]]]]).to(device = device, memory_format = torch.channels_last).float().div(8)
    dy_ker = torch.tensor([[[[1,2, 1],[0,0, 0],[-1,-2,-1]]]]).to(device = device, memory_format = torch.channels_last).float().div(8)
    f_x = F.conv2d(tensor, dx_ker,  padding='same')
    f_y = F.conv2d(tensor, dy_ker,  padding='same')
    return torch.cat([f_x,f_y])

