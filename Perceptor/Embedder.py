from pytti import *
from pytti.Perceptor import CLIP_PERCEPTORS

import torch
from torch import nn

import kornia.augmentation as K

class HDMultiClipEmbedder(nn.Module):
  """
  Multi-CLIP embedder that uses cutouts to view images larger than 224x224.
  with code by Katherine Crowson (https://github.com/crowsonkb)
  and jbusted (https://twitter.com/jbusted1)
  and dribnet (https://github.com/dribnet)
  """
  def __init__(self, perceptors=CLIP_PERCEPTORS, cutn = 40, noise_fac = 0.1):
    super().__init__()
    self.cut_sizes = [p.visual.input_resolution for p in perceptors]
    self.cutn = cutn
    self.noise_fac = noise_fac
    self.augs = nn.Sequential(K.RandomHorizontalFlip(p=0.5),
                              K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
                              K.RandomPerspective(0.2, p=0.4,),
                              K.ColorJitter(hue=0.01, saturation=0.01,  p=0.7),)
    self.input_axes  = ('n', 's', 'y', 'x')
    self.output_axes = ('c', 'n', 'i')
    self.perceptors = perceptors

  def forward(self, diff_image, input = None):
    """
    diff_image: (DifferentiableImage) input image
    returns images embeds
    """
    perceptors=self.perceptors
    sideX, sideY = diff_image.image_shape
    if input is None:
      input = format_module(diff_image, self)
    else:
      input = format_input(input, diff_image, self)
    max_size = min(sideX, sideY)
    image_embeds = []
    for cut_size, perceptor in zip(self.cut_sizes, perceptors):
      min_size = min(sideX, sideY, cut_size)

      cutouts = []
      for _ in range(self.cutn):
        size = int(max_size *
              (torch.zeros(1,).normal_(mean=.8, std=.3)
              .clip(cut_size/max_size, 1.) ** 1.5))
        offsetXMax = sideX - size + 1
        paddingX = math.round(sideX * 0.25)
        offsetYMax = sideY - size + 1
        paddingY = math.round(sideY * 0.25)
        offsetx = torch.clamp(torch.randint(0 - paddingX, offsetXMax + paddingX, ()), 0, offsetXMax)
        offsety = torch.clamp(torch.randint(0 - paddingY, offsetYMax + paddingY, ()), 0, offsetYMax)
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
      cutouts = self.augs(torch.cat(cutouts))
      if self.noise_fac:
        facs    = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
        cutouts = cutouts + facs * torch.randn_like(cutouts)
      clip_in = normalize(cutouts)
      image_embeds.append(perceptor.encode_image(clip_in).float().unsqueeze(0))

    return cat_with_pad(image_embeds)
