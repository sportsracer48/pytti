from torch.nn import functional as F
import math
from pytti.LossAug import Loss

def tv_loss(input):
  """L2 total variation loss, as in Mahendran et al."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
  y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
  return (x_diff**2 + y_diff**2).mean([1, 2, 3])

class TVLoss(Loss):
  def __init__(self, weight = 0.15, stop = -math.inf, name = "smoothing loss (TV)"):
    super().__init__(weight, stop, name)
  def get_loss(self, input, img):
    return tv_loss(input)