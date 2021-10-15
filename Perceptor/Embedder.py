from pytti import *
import pytti

import torch
from torch import nn
from torch.nn import functional as F

import kornia.augmentation as K

PADDING_MODES = {'mirror':'reflect','smear':'replicate','wrap':'circular','black':'constant'}

class HDMultiClipEmbedder(nn.Module):
  """
  Multi-CLIP embedder that uses cutouts to view images larger than 224x224.
  with code by Katherine Crowson (https://github.com/crowsonkb)
  and jbusted (https://twitter.com/jbusted1)
  and dribnet (https://github.com/dribnet)
  """
  def __init__(self, perceptors=None, cutn = 40, cut_pow = 1.5, padding = 0.25, border_mode = 'clamp', noise_fac = 0.1):
    super().__init__()
    if perceptors is None:
      perceptors = pytti.Perceptor.CLIP_PERCEPTORS
    self.cut_sizes = [p.visual.input_resolution for p in perceptors]
    self.cutn = cutn
    self.noise_fac = noise_fac
    self.augs = nn.Sequential(K.RandomHorizontalFlip(p=0.5),
                              K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
                              K.RandomPerspective(0.2, p=0.4,),
                              K.ColorJitter(hue=0.01, saturation=0.01,  p=0.7),
                              K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7),)
    self.input_axes  = ('n', 's', 'y', 'x')
    self.output_axes = ('c', 'n', 'i')
    self.perceptors = perceptors
    self.padding = padding
    self.cut_pow = cut_pow
    self.border_mode = border_mode

  def forward(self, diff_image, input = None, device = DEVICE):
    """
    diff_image: (DifferentiableImage) input image
    returns images embeds
    """
    perceptors=self.perceptors
    side_x, side_y = diff_image.image_shape
    if input is None:
      input = format_module(diff_image, self).to(device = device, memory_format = torch.channels_last)
    else:
      input = format_input(input, diff_image, self).to(device = device, memory_format = torch.channels_last)
    max_size = min(side_x, side_y)
    image_embeds = []
    all_offsets = []
    all_sizes = []

    paddingx = min(round(side_x * self.padding), side_x)
    paddingy = min(round(side_y * self.padding), side_y)
    if self.border_mode != 'clamp':
      input = F.pad(input,(paddingx, paddingx, paddingy, paddingy), mode = PADDING_MODES[self.border_mode])
    for cut_size, perceptor in zip(self.cut_sizes, perceptors):
      min_size = min(side_x, side_y, cut_size)

      cutouts = []
      offsets = []
      sizes   = []
      for _ in range(self.cutn):
        #mean is 0.8
        #varience is 0.3
        size = int(max_size *
              (torch.zeros(1,).normal_(mean=.8, std=.3)
              .clip(cut_size/max_size, 1.) ** self.cut_pow))
        offsetx_max = side_x - size + 1
        offsety_max = side_y - size + 1
        if self.border_mode == 'clamp':
          offsetx = torch.clamp((torch.rand([])*(offsetx_max+2*paddingx) - paddingx).floor().int(), 0, offsetx_max)
          offsety = torch.clamp((torch.rand([])*(offsety_max+2*paddingy) - paddingy).floor().int(), 0, offsety_max)
          cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        else:
          px = min(size, paddingx)
          py = min(size, paddingy)
          offsetx = (torch.rand([])*(offsetx_max+2*px) - px).floor().int()
          offsety = (torch.rand([])*(offsety_max+2*py) - py).floor().int()
          cutout = input[:, :, paddingy + offsety:paddingy + offsety + size, paddingx + offsetx:paddingx + offsetx + size]
        cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
        offsets.append(torch.as_tensor([[offsetx/side_x, offsety/side_y]]).to(device))
        sizes.append(torch.as_tensor([[size/side_x, size/side_y]]).to(device))
      cutouts = self.augs(torch.cat(cutouts))
      offsets = torch.cat(offsets)
      sizes   = torch.cat(sizes)
      if self.noise_fac:
        facs    = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
        cutouts = cutouts + facs * torch.randn_like(cutouts)
      clip_in = normalize(cutouts)
      image_embeds.append(perceptor.encode_image(clip_in).float().unsqueeze(0))
      all_offsets.append(offsets)
      all_sizes.append(sizes)
    return cat_with_pad(image_embeds), torch.stack(all_offsets), torch.stack(all_sizes)
