from pytti.LossAug import MSELoss
import gc, torch, os, math
from pytti import DEVICE
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image

class EdgeLoss(MSELoss):
  @torch.no_grad()
  def set_target_image(self, pil_image, device = DEVICE):
    pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
    self.comp.set_(EdgeLoss.get_edges(TF.to_tensor(pil_image).unsqueeze(0).to(device = device, memory_format = torch.channels_last)))
    
  def get_loss(self, input, img):
    edge_map = EdgeLoss.get_edges(input)
    return super().get_loss(edge_map, img)
  
  @staticmethod
  def get_edges(tensor, device = DEVICE):
    tensor = TF.rgb_to_grayscale(tensor)
    dx_ker = torch.tensor([[[[1,0,-1],[2,0,-2],[ 1, 0,-1]]]]).to(device = device, memory_format = torch.channels_last).float().div(8)
    dy_ker = torch.tensor([[[[1,2, 1],[0,0, 0],[-1,-2,-1]]]]).to(device = device, memory_format = torch.channels_last).float().div(8)
    f_x = F.conv2d(tensor, dx_ker,  padding='same')
    f_y = F.conv2d(tensor, dy_ker,  padding='same')
    return torch.cat([f_x,f_y])

