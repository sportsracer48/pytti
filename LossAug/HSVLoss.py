from pytti.LossAug import MSELoss
import torch
from kornia.color import rgb_to_hsv

class HSVLoss(MSELoss):
  @classmethod
  def convert_input(cls, input, img):
    out = rgb_to_hsv(input)
    out = torch.cat([input,out[:,1:,...]],dim = 1)
    return out