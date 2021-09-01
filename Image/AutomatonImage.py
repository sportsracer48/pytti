from pytti import *
import torch
from torch import nn
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from pytti.Image import DifferentiableImage



class AbstractCAModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_input(self, state_grid):
        self.state_grid = state_grid

    def forward(self):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

def stochastic_update_mask(ds_grid, prob=0.5):
    # Generate mask for zero out a random fraction of the updates.
    bern = torch.distributions.Bernoulli(prob)
    rand_mask = bern.sample((ds_grid.shape[2] * ds_grid.shape[3],))
    rand_mask = rand_mask.view(ds_grid.shape[2:]).float()
    return rand_mask.to(ds_grid.device)[None, None]


def alive_mask(state_grid, thr=0.1):
    # Take the alpha channel as the measure of “life”.
    alpha = state_grid[:, [3], :, :].clamp(0, 1)
    alive = (nn.MaxPool2d(3, stride=1, padding=1)(alpha) > thr).float()#.unsqueeze(1)
    return alive

class Perception(nn.Module):
    def __init__(self, channels=16, norm_kernel=False):
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]) / 8
        sobel_y = torch.tensor([[1.0, 2.0, 1.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, -2.0, -1.0]]) / 8
        identity = torch.tensor([[0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0]])

        self.kernel =  torch.stack((identity, sobel_x, sobel_y)).repeat(channels, 1, 1).unsqueeze(1)
        if norm_kernel:
            self.kernel /= channels

    def forward(self, state_grid):
        return F.conv2d(state_grid, 
                        self.kernel.to(state_grid.device), 
                        groups=self.channels,
                        padding=1)  # thanks https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch?files=1 for the group parameter

class Policy(nn.Module):
    def __init__(self, state_dim=16, interm_dim=128,
                 use_embedding=True, kernel=1, padding=0,
                 bias=False):
        super().__init__()
        dim = state_dim * 3
        if use_embedding:
            dim += 1
        self.conv1 = nn.Conv2d(dim, interm_dim, kernel, padding=padding)
        self.conv2 = nn.Conv2d(interm_dim, state_dim, kernel, padding=padding,
                               bias=bias)
        nn.init.constant_(self.conv2.weight, 0.)
        if bias:
            nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, state):
        interm = self.conv1(state)
        interm = torch.relu(interm)
        return self.conv2(interm)


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Sequential(*[nn.Conv2d(4, 16, 3),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 16, 3),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 1, 3)])

    def forward(self, img):
        return nn.AdaptiveAvgPool2d((1, 1))(self.embedder(img))

class SimpleCA(AbstractCAModel):
    def __init__(self, perception, policy):
        super().__init__()
        self.perception = perception
        self.policy = policy
        self.stochastic_prob = 0.5

    def forward(self):
        alive_pre = alive_mask(self.state_grid, thr=0.1)
        perception_grid = self.perception(self.state_grid)
        ds_grid = self.policy(perception_grid)
        mask = stochastic_update_mask(ds_grid,
                                            prob=self.stochastic_prob)
        self.state_grid = self.state_grid + ds_grid * mask
        alive_post = alive_mask(self.state_grid, thr=0.1)
        final_mask = (alive_post.bool() & alive_pre.bool()).float()
        self.state_grid = self.state_grid * final_mask

        return final_mask



class AutomatonImage(DifferentiableImage):
  """
  Naive RGB image representation
  """
  def __init__(self, width, height, scale=1, device=DEVICE):
    super().__init__(width*scale, height*scale)
    perception = Perception(channels=16,norm_kernel=False).to(device)
    self.policy = Policy(use_embedding=False, kernel=1, padding=0, interm_dim=128, bias=True).to(device)
    self.ca  = SimpleCA(perception, self.policy)
    self.ca.get_input(torch.rand(1,16,height,width).to(device))
    

    self.output_axes = ('n', 's', 'y', 'x')
    self.lr = 0.002

  def decode_tensor(self):
    for k in range(2):
      final_mask = self.ca.forward()
      
    width, height = self.image_shape
    out = self.ca.state_grid[0, :3, ...].unsqueeze(0)
    out = F.interpolate(out, (height, width) , mode='nearest')
    return clamp_with_grad(out,0,1)
  
  def parameters(self):
    return self.policy.parameters()

  @torch.no_grad()
  def update(self):
    self.ca.get_input(self.ca.state_grid.detach())

  @torch.no_grad()
  def encode_image(self, pil_image, device=DEVICE):
    pass

  @torch.no_grad()
  def encode_random(self):
    self.ca.state_grid.uniform_()
