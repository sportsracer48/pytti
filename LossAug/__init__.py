from torch.nn import functional as F

def tv_loss(input):
  """L2 total variation loss, as in Mahendran et al."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
  y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
  return (x_diff**2 + y_diff**2).mean([1, 2, 3])

class TV_Loss(nn.Module):
  def __init__(self, strength):
    self.register_buffer('strength', strength)
  def forward(self, input):
    return self.strength*tv_loss(input)

class MSE_Loss(nn.Module):
  def __init__(self, comp, weight= 0.5):
    self.register_buffer['comp', comp]
    self.register_buffer['weight', height]
  def forward(self, input):
    pass