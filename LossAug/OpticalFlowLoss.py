from pytti.LossAug import MSELoss
import sys, os, gc
import argparse
import os
import cv2
import glob
import math, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

os.chdir('GMA')
try:
  sys.path.append('core')
  from network import RAFTGMA
  from utils import flow_viz
  from utils.utils import InputPadder
finally:
  os.chdir('..')

from pytti.Transforms import apply_flow
from pytti import fetch
from pytti.Image.RGBImage import RGBImage
from pytti import DEVICE

GMA = None
def init_GMA(checkpoint_path):
  global GMA
  if GMA is None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default=checkpoint_path)
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args([])
    GMA = torch.nn.DataParallel(RAFTGMA(args))
    GMA.load_state_dict(torch.load(checkpoint_path))
    GMA.to(DEVICE)
    GMA.eval()

def sample(tensor, uv, device=DEVICE):
  height, width = tensor.shape[-2:]
  max_pos = torch.tensor([width-1,height-1], device=device).view(2,1,1)
  grid = uv.div(max_pos/2).sub(1).movedim(0,-1).unsqueeze(0)
  return F.grid_sample(tensor.unsqueeze(0), grid, align_corners = True).squeeze(0)

class TargetFlowLoss(MSELoss):

  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss", image_shape = None):
    super().__init__(comp, weight, stop, name, image_shape)
    self.register_buffer('last_step', comp.new_tensor(comp))

  @torch.no_grad()
  def set_target_flow(self, flow, device=DEVICE):
    self.comp.set_(flow.movedim(-1,1).to(device, memory_format = torch.channels_last))
  
  @torch.no_grad()
  def set_last_step(self, last_step_pil, device = DEVICE):
    last_step = TF.to_tensor(last_step_pil).unsqueeze(0).to(device, memory_format = torch.channels_last)
    self.last_step.set_(last_step)

  def get_loss(self, input, img, device=DEVICE):
    os.chdir('GMA')
    try:
      init_GMA('checkpoints/gma-sintel.pth')
      image1 = self.last_step
      image2 = input
      padder = InputPadder(image1.shape)
      image1, image2 = padder.pad(image1, image2)
      _, flow = GMA(image1, image2, iters=12, test_mode=True)
      flow = flow.to(device, memory_format = torch.channels_last)
    finally:
      os.chdir('..')
    return super().get_loss(F.adaptive_avg_pool2d(flow, self.comp.shape[-2:]), img)/50


class OpticalFlowLoss(MSELoss):
  @staticmethod
  @torch.no_grad()
  def motion_edge_map(image1, image2, img, border_mode = 'smear', sampling_mode = 'bilinear',device=DEVICE):
    # algorithm based on https://github.com/manuelruder/artistic-videos/blob/master/consistencyChecker/consistencyChecker.cpp
    # reimplemented in pytorch by Henry Rachootin
    # // consistencyChecker
    # // Check consistency of forward flow via backward flow.
    # //
    # // (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016
    gc.collect()
    torch.cuda.empty_cache()
    flow_forward = OpticalFlowLoss.get_flow(image1, image2)
    flow_backward = OpticalFlowLoss.get_flow(image2, image1)
    flow_target_direct = apply_flow(img, -flow_backward, border_mode = border_mode, sampling_mode = sampling_mode)
    flow_target_latent = img.get_latent_tensor(detach = True)

    dx_ker = torch.tensor([[[[0,0,0],[1,0,-1],[0, 0,0]]]], device = device).float().div(2).repeat(2,2,1,1)
    dy_ker = torch.tensor([[[[0,1,0],[0,0, 0],[0,-1,0]]]], device = device).float().div(2).repeat(2,2,1,1)
    f_x = nn.functional.conv2d(flow_backward, dx_ker,  padding='same')
    f_y = nn.functional.conv2d(flow_backward, dy_ker,  padding='same')
    motionedge = torch.cat([f_x,f_y]).square().sum(dim=(0,1))

    height, width = flow_forward.shape[-2:]
    y,x = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    x = x.to(device)
    y = y.to(device)

    p1 = torch.stack([x,y])
    v1 = flow_forward.squeeze(0)
    p0 = p1 + flow_backward.squeeze()
    v0 = sample(v1, p0)
    p1_back = p0 + v0
    v1_back = flow_backward.squeeze(0)
  
    r1 = torch.floor(p0)
    r2 = r1 + 1
    max_pos = torch.tensor([width-1,height-1], device=device).view(2,1,1)
    min_pos = torch.tensor([0, 0], device=device).view(2,1,1)
    overshoot = torch.logical_or(r1.lt(min_pos),r2.gt(max_pos))
    overshoot = torch.logical_or(overshoot[0],overshoot[1])

    missed = (p1_back - p1).square().sum(dim=0).ge(torch.stack([v1_back,v0]).square().sum(dim=(0,1)).mul(0.01).add(0.5))
    motion_boundary = motionedge.ge(v1_back.square().sum(dim=0).mul(0.01).add(0.002))

    reliable = torch.ones((height, width), device=device)
    reliable[motion_boundary] = 0
    reliable[missed] = -1
    reliable[overshoot] = 0
    mask = TF.gaussian_blur(reliable.unsqueeze(0), 3).clip(0,1)

    return mask, flow_target_direct, flow_target_latent

  @staticmethod
  @torch.no_grad()
  def get_flow(image1, image2, device=DEVICE):
    os.chdir('GMA')
    try:
      init_GMA('checkpoints/gma-sintel.pth')
      image1 = TF.to_tensor(image1).unsqueeze(0).to(device)
      image2 = TF.to_tensor(image2).unsqueeze(0).to(device)
      padder = InputPadder(image1.shape)
      image1, image2 = padder.pad(image1, image2)
      flow_low, flow_up = GMA(image1, image2, iters=12, test_mode=True)
    finally:
      os.chdir('..')
    return flow_up

  def __init__(self, comp, weight = 0.5, stop = -math.inf, name = "direct target loss", image_shape = None):
    super().__init__(comp, weight, stop, name, image_shape)
    self.latent_loss = MSELoss(comp.new_tensor(comp), weight, stop, name, image_shape)
    self.direct_strength = 1
    self.latent_strength = 1

  def set_latent_strength(self, strength):
    self.latent_strength = strength
  def set_direct_strength(self, strength):
    self.direct_strength = strength

  @torch.no_grad()
  def set_flow(self, frame_prev, frame_next, img, path, border_mode = 'smear', sampling_mode = 'bilinear'):
    if path is not None:
      img = copy.deepcopy(img)
      state_dict = torch.load(path)
      img.load_state_dict(state_dict)

    mask_tensor, flow_target, flow_target_latent = OpticalFlowLoss.motion_edge_map(frame_prev, frame_next, img, border_mode, sampling_mode)
    self.comp.set_(flow_target)
    self.latent_loss.comp.set_(flow_target_latent)
    self.set_mask(mask_tensor.unsqueeze(0))
    
    array = flow_target.squeeze(0).movedim(0,-1).mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)[:,:,:]
    return Image.fromarray(array), mask_tensor

  @torch.no_grad()
  def set_mask(self,mask):
    super().set_mask(TF.resize(mask, self.comp.shape[-2:]))
    if mask is not None:
      self.latent_loss.set_mask(TF.resize(mask, self.latent_loss.comp.shape[-2:]))
    else:
      self.latent_loss.set_mask(None)

  def get_loss(self, input, img):
    l1 = super().get_loss(input, img)
    l2 = self.latent_loss.get_loss(img.get_latent_tensor(), img)
    #print(float(l1),float(l2))
    return l1*self.direct_strength+l2*self.latent_strength


    