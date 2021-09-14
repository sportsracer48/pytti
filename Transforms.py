import glm, gc, torch, math, os
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
from pytti import *
from infer import InferenceHelper

PADDING_MODES = {'mirror':'reflection','smear':'border','black':'zeros','wrap':'zeros'}

def apply_grid(tensor, grid, border_mode):
  height, width = tensor.shape[-2:]
  if border_mode == 'wrap':
    max_offset = torch.max(grid.clamp(min= 1))
    min_offset = torch.min(grid.clamp(max=-1))
    max_coord = max(max_offset, abs(min_offset))
    if max_coord > 1:
      mod_offset = int(math.ceil(max_coord))
      #make it odd for sure
      mod_offset += 1-(mod_offset % 2)
      grid.add_(mod_offset)
      grid.remainder_(2.0001)
      grid.sub_(1)
  return F.grid_sample(tensor,grid,align_corners=True,padding_mode=PADDING_MODES[border_mode])

def zoom_2d(img, translate = (0,0), zoom = (0,0), rotate = 0, border_mode = 'mirror', device=DEVICE):
  try:
    tensor = img.get_image_tensor().unsqueeze(0)
    fallback = False
  except NotImplementedError:
    tensor = TF.to_tensor(img.decode_image()).unsqueeze(0).to(DEVICE)
    fallback = True
  height, width = tensor.shape[-2:]
  zy,zx = ((height-zoom[1])/height, (width-zoom[0])/width)
  ty,tx = (translate[1]*2/height, -translate[0]*2/width)
  theta = math.radians(rotate)
  affine = torch.tensor([[zx*math.cos(theta),-zy*math.sin(theta),tx],
                          [zx*math.sin(theta), zy*math.cos(theta),ty]]).unsqueeze(0).to(device)
  grid   = F.affine_grid(affine, tensor.shape, align_corners=True)
  tensor = apply_grid(tensor, grid, border_mode)
  if not fallback:
    img.set_image_tensor(tensor.squeeze(0))
  else:
    array = tensor.squeeze().movedim(0,-1).mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)[:,:,:]
    img.encode_image(Image.fromarray(array))

def render_image_3d(image, depth, P, T, border_mode, device=DEVICE):
  """
  image: n x h x w pytorch Tensor: the image tensor
  depth: h x w pytorch Tensor: the depth tensor
  P: 4 x 4 pytorch Tensor: the perspective matrix
  T: 4 x 4 pytorch Tensor: the camera move matrix
  """
  #create grid of points matching image 
  h,w = image.shape[-2:]
  f   = w/h
  image = image.unsqueeze(0)

  y,x = torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-f,f,w))
  x = x.unsqueeze(0).unsqueeze(0)
  y = y.unsqueeze(0).unsqueeze(0)
  xy = torch.cat([x,y],dim=1).to(device)

  #v,u = torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w))
  #u = u.unsqueeze(0).unsqueeze(0)
  #v = v.unsqueeze(0).unsqueeze(0)
  #uv = torch.cat([u,v],dim=1).to(device)
  identity = torch.eye(3).to(device)
  identity = identity[0:2,:].unsqueeze(0) #for batch
  uv = F.affine_grid(identity, image.shape)
  #get the depth at each point
  depth = depth.unsqueeze(0).unsqueeze(0)
  #depth = depth.to(device)

  view_pos = torch.cat([xy,-depth,torch.ones_like(depth)],dim=1)
  #view_pos = view_pos.to(device)
  #apply the camera move matrix
  next_view_pos = torch.tensordot(T.float(), view_pos.float(), ([0],[1])).movedim(0,1)
  
  #apply the perspective matrix
  clip_pos = torch.tensordot(P.float(), view_pos.float(), ([0],[1])).movedim(0,1)
  clip_pos = clip_pos/(clip_pos[:,3,...].unsqueeze(1))

  next_clip_pos = torch.tensordot(P.float(), next_view_pos.float(), ([0],[1])).movedim(0,1)
  next_clip_pos = next_clip_pos/(next_clip_pos[:,3,...].unsqueeze(1))
  
  #get the offset
  offset = (next_clip_pos - clip_pos)[:,0:2,...]
  #offset[:,1,...] *= -1
  #offset = offset.to(device)
  #render the image
  offset = offset.permute(0,2,3,1)
  grid = uv - offset
  #grid = grid.permute(0,2,3,1)
  #grid = grid.to(device)

  return apply_grid(image,grid,border_mode).squeeze(0)

infer_helper = None  
def init_AdaBins():
  global infer_helper
  if infer_helper is None:
    os.chdir('AdaBins')
    try:
      infer_helper = InferenceHelper(dataset='nyu')
    finally:
      os.chdir('..')

def zoom_3d(img, translate = (0,0,0), rotate=0, fov = 45, near=180, far=15000, border_mode='mirror'):
  print('moving:',translate)
  width, height = img.image_shape
  px = 2/height
  alpha = math.radians(fov)
  depth = 1/(math.tan(alpha/2))
  #these make sure not to re-init
  #init_opengl(width, height, depth, alpha)
  init_AdaBins()
  
  pil_image = img.decode_image()
  
  #pil_image = pil_image.filter(ImageFilter.GaussianBlur(img.scale))
  f = width/height
  
  #if the area of an image is above this, the depth model fails
  max_depth_area = 500000
  image_area     = width*height
  if image_area > max_depth_area:
    depth_scale_factor = math.sqrt(max_depth_area/image_area)
    depth_input = pil_image.resize((int(width*depth_scale_factor), int(height*depth_scale_factor)), Image.LANCZOS)
    depth_resized = True
  else:
    depth_input = pil_image
    depth_resized = False
  #run the depth model (whatever that means)
  gc.collect()
  torch.cuda.empty_cache()
  os.chdir('AdaBins')
  try:
    _, depth_map = infer_helper.predict_pil(depth_input)
  finally:
    os.chdir('..')
  gc.collect()
  torch.cuda.empty_cache()

  #convert depth map
  depth_map = depth_map
  depth_min = np.min(depth_map)
  depth_max = np.max(depth_map)
  depth_image = Image.fromarray(np.array(np.interp(depth_map.squeeze(), (depth_min, depth_max), (0,255)), dtype=np.uint8))
  depth_map = np.interp(depth_map, (1e-3, 10), (near*px,far*px))
  depth_min = np.min(depth_map)
  depth_max = np.max(depth_map)
  print("depth range:",depth_min/px,"to",depth_max/px)
  
  try:
    image_tensor = img.get_image_tensor().to(DEVICE)
    depth_tensor = TF.resize(torch.from_numpy(depth_map), image_tensor.shape[-2:], interpolation = TF.InterpolationMode.BICUBIC).squeeze().to(DEVICE)
    fallback = False
  except NotImplementedError:
    #fallback path
    image_tensor = TF.to_tensor(pil_image).to(DEVICE)
    if depth_resized:
      depth_tensor = TF.resize(torch.from_numpy(depth_map), image_tensor.shape[-2:], interpolation = TF.InterpolationMode.BICUBIC).squeeze().to(DEVICE)
    else:
      depth_tensor = torch.from_numpy(depth_map).squeeze().to(DEVICE)
    fallback = True
  p_matrix = torch.as_tensor(glm.perspective(alpha, f, 0.1, 4).to_list()).to(DEVICE)
  tx,ty,tz = translate
  if rotate != 0:
    r_matrix = glm.rotate(glm.mat4(1), math.radians(rotate), glm.vec3(0,0,-1))
  else:
    r_matrix = glm.mat4(1)
  t_matrix = torch.as_tensor(glm.translate(r_matrix, glm.vec3(tx*px,-ty*px,tz*px)).to_list()).to(DEVICE)
  new_image = render_image_3d(image_tensor, depth_tensor, p_matrix, t_matrix, border_mode = border_mode)
  if not fallback:
    img.set_image_tensor(new_image)
  else:
    #fallback path
    array = new_image.movedim(0,-1).mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)[:,:,:]
    img.encode_image(Image.fromarray(array))
  return depth_image, img.decode_image()
  

  
  



