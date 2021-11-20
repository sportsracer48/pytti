from infer import InferenceHelper
from pytti.LossAug import MSELoss
import gc, torch, os, math
from pytti import DEVICE, vram_usage_mode
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image, ImageOps

infer_helper = None
def init_AdaBins():
  global infer_helper
  if infer_helper is None:
    with vram_usage_mode('AdaBins'):
      print('Loading AdaBins...')
      os.chdir('AdaBins')
      try:
        infer_helper = InferenceHelper(dataset='nyu')
      finally:
        os.chdir('..')
      print('AdaBins loaded.')

class DepthLoss(MSELoss):
  @torch.no_grad()
  def set_comp(self, pil_image):
    #pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
    self.comp.set_(DepthLoss.make_comp(pil_image))
    if self.use_mask and self.mask.shape[-2:] != self.comp.shape[-2:]:
      self.mask.set_(TF.resize(self.mask, self.comp.shape[-2:]))
  
  def get_loss(self, input, img):
    height, width = input.shape[-2:]
    max_depth_area = 500000
    image_area     = width*height
    if image_area > max_depth_area:
      depth_scale_factor = math.sqrt(max_depth_area/image_area)
      height, width = int(height*depth_scale_factor), int(width*depth_scale_factor)
      depth_input = TF.resize(input, (height, width), interpolation = TF.InterpolationMode.BILINEAR)
      depth_resized = True
    else:
      depth_input = input
      depth_resized = False

    _, depth_map  = infer_helper.model(depth_input)
    depth_map = F.interpolate(depth_map, self.comp.shape[-2:],mode='bilinear', align_corners=True)
    #depth_map = F.interpolate(depth_map, (height, width), mode='bilinear', align_corners=True)
    return super().get_loss(depth_map, img)
  
  @classmethod
  @vram_usage_mode("Depth Loss")
  def make_comp(cls, pil_image, device = DEVICE):
    depth, _ = DepthLoss.get_depth(pil_image)
    return torch.from_numpy(depth).to(device)

  @staticmethod
  def get_depth(pil_image):
    init_AdaBins()
    width, height = pil_image.size
    
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

    return depth_map, depth_resized


