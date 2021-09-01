from pytti import *
from pytti.Image import DifferentiableImage
from pytti.LossAug import MSE_Loss
from pytti.ImageGuide import DirectImageGuide
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

def break_tensor(tensor):
  floors = tensor.floor().long()
  ceils  = tensor.ceil().long()
  rounds = tensor.round().long()
  fracs  = tensor - floors
  return floors, ceils, rounds, fracs

class PixelImage(DifferentiableImage):
  """
  differentiable image format for pixel art images
  """
  def __init__(self, width, height, scale, pallet_size, n_pallets = 2, device=DEVICE):
    super().__init__(width*scale, height*scale)
    self.pallet_inertia = 1#math.pow(width*height*(n_pallets+1),1/3)/math.sqrt(n_pallets*pallet_size*3) 
    pallet = torch.linspace(0,self.pallet_inertia,pallet_size).view(pallet_size,1,1).repeat(1,n_pallets,3)
    #pallet.set_(torch.rand_like(pallet)*self.pallet_inertia)
    self.pallet = nn.Parameter(pallet.to(device))

    self.pallet_size = pallet_size
    self.n_pallets = n_pallets
    self.value  = nn.Parameter(torch.zeros(height,width).to(device))
    self.tensor = nn.Parameter(torch.zeros(n_pallets, height, width).to(device))
    self.output_axes = ('n', 's', 'y', 'x')
    self.scale = scale

  def sort_pallet(self):
    pallet = (self.pallet/self.pallet_inertia).clamp(0,1)
    #color values from https://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
    magic_color = pallet.new_tensor([[[0.241,0.691,0.068]]])
    color_norms = (pallet.square()*magic_color).sum(dim = -1)
    pallet_indices = color_norms.argsort(dim = 0).T
    pallet = torch.stack([pallet[i][:,j] for j,i in enumerate(pallet_indices)],dim=1)
    return pallet
  
  def decode_tensor(self):
    width, height = self.image_shape
    pallet = self.sort_pallet()

    #brightnes values of pixels
    values = self.value.clamp(0,1)*(self.pallet_size-1)
    value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
    value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

    pallet_weights = self.tensor.movedim(0,2)
    pallets = F.one_hot(pallet_weights.argmax(dim = 2), num_classes=self.n_pallets)

    pallet_weights = pallet_weights.softmax(dim = 2).unsqueeze(-1)
    pallets = pallets.unsqueeze(-1)

    colors_disc = pallet[value_rounds]
    colors_disc = (colors_disc * pallets).sum(dim = 2)
    colors_disc = F.interpolate(colors_disc.movedim(2,0).unsqueeze(0), (height, width) , mode='nearest')

    colors_cont = pallet[value_floors]*(1-value_fracs) + pallet[value_ceils]*value_fracs
    colors_cont = (colors_cont * pallet_weights).sum(dim = 2)
    colors_cont = F.interpolate(colors_cont.movedim(2,0).unsqueeze(0), (height, width) , mode='nearest')
    return replace_grad(colors_disc, colors_cont*0.5+colors_disc*0.5)

  @torch.no_grad()
  def render_value_image(self):
    width, height = self.image_shape
    values = self.value.clamp(0,1).unsqueeze(-1).repeat(1,1,3)
    array = np.array(values.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:,:,:]
    return Image.fromarray(array).resize((width,height), Image.NEAREST)

  @torch.no_grad()
  def render_pallet(self):
    pallet = self.sort_pallet()
    width, height = self.n_pallets*16, self.pallet_size*32
    array = np.array(pallet.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:,:,:]
    return Image.fromarray(array).resize((width,height), Image.NEAREST)

  @torch.no_grad()
  def render_channel(self, pallet_i):
    width, height = self.image_shape
    pallet = self.sort_pallet()
    pallet[:,:pallet_i   ,:] = 0.5
    pallet[:, pallet_i+1:,:] = 0.5

    values = self.value.clamp(0,1)*(self.pallet_size-1)
    value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
    value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

    pallet_weights = self.tensor.movedim(0,2)
    pallets = F.one_hot(pallet_weights.argmax(dim = 2), num_classes=self.n_pallets)
    pallet_weights = pallet_weights.softmax(dim = 2).unsqueeze(-1)

    colors_cont = pallet[value_floors]*(1-value_fracs) + pallet[value_ceils]*value_fracs
    colors_cont = (colors_cont * pallet_weights).sum(dim = 2)
    colors_cont = F.interpolate(colors_cont.movedim(2,0).unsqueeze(0), (height, width) , mode='nearest')

    tensor = named_rearrange(colors_cont, self.output_axes, ('y', 'x', 's'))
    array = np.array(tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:,:,:]
    return Image.fromarray(array)

  @torch.no_grad()
  def update(self):
    self.pallet.clamp_(0,self.pallet_inertia)
    self.value.clamp_(0,1)
    self.tensor.clamp_(0,float('inf'))
    #self.tensor.set_(self.tensor.softmax(dim = 0))

  def encode_image(self, pil_image, device=DEVICE, rescale = True):
    width, height = self.image_shape
    pil_image = pil_image.resize((width,height), Image.LANCZOS)
    target = TF.to_tensor(pil_image).to(device)
    mse = MSE_Loss(target)
    #no embedder needed without any prompts
    guide = DirectImageGuide(self, None, optimizer = optim.Adam([self.pallet, self.tensor], lr = .1))
    with torch.no_grad():
      scale = self.scale
      im = ImageOps.grayscale(pil_image.resize((width//scale, height//scale)))
      im = TF.to_tensor(im)
      self.value.set_(im[0].to(DEVICE))
    guide.run_steps(201,[],[mse])

  @torch.no_grad()
  def encode_random(self, random_pallet = False):
    self.value.uniform_()
    self.tensor.uniform_()
    if random_pallet:
      self.pallet.uniform_(to=self.pallet_inertia)

