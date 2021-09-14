from os.path import exists as path_exists
import sys, subprocess
if not path_exists('./taming-transformers'):
  raise FileNotFoundError("ERROR: taming-transformers is missing!")
if './taming-transformers' not in sys.path:
  sys.path.append('./taming-transformers')
else:
  print("DEBUG: sys.path already contains ./taming transformers")
from taming.models import cond_transformer, vqgan

from pytti import *
import torch
from torch.nn import functional as F
from pytti.Image import EMAImage
from pytti import DEVICE
from torchvision.transforms import functional as TF
from PIL import Image
from omegaconf import OmegaConf

VQGAN_MODEL = None
VQGAN_NAME  = None
VQGAN_IS_GUMBEL = None

VQGAN_MODEL_NAMES = ["imagenet", "coco", "wikiart", "sflicker", "openimages"]
VQGAN_CONFIG_URLS = {
  "imagenet":   ["curl -L -o imagenet.yaml -C - https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1"],
  "coco":       ["curl -L -o coco.yaml -C - https://dl.nmkd.de/ai/clip/coco/coco.yaml"],
  "wikiart":    ["curl -L -o wikiart.yaml -C - http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml"],
  "sflicker":   ["curl -L -o sflckr.yaml -C - https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1"],
  "faceshq":    ["curl -L -o faceshq.yaml -C - https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT"],
  "openimages": ["curl -L -o openimages.yaml -C - https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"]
}
VQGAN_CHECKPOINT_URLS = {
  "imagenet":   ["curl -L -o imagenet.ckpt -C - https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1"],
  "coco":       ["curl -L -o coco.ckpt -C - https://dl.nmkd.de/ai/clip/coco/coco.ckpt"],
  "wikiart":    ["curl -L -o wikiart.ckpt -C - http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt"],
  "sflicker":   ["curl -L -o sflckr.ckpt -C - https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1"],
  "faceshq":    ["curl -L -o faceshq.ckpt -C - https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt"],
  "openimages": ["curl -L -o openimages.ckpt -C - https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1"]
}

def load_vqgan_model(config_path, checkpoint_path):
  config = OmegaConf.load(config_path)
  if config.model.target == 'taming.models.vqgan.VQModel':
    model = vqgan.VQModel(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    gumbel = False
  elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
    parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
    parent_model.eval().requires_grad_(False)
    parent_model.init_from_ckpt(checkpoint_path)
    model = parent_model.first_stage_model
    gumbel = False
  elif config.model.target == 'taming.models.vqgan.GumbelVQ':
    model = vqgan.GumbelVQ(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    gumbel = True
  else:
    raise ValueError(f'unknown model type: {config.model.target}')
  del model.loss
  return model, gumbel

def vector_quantize(x, codebook):
  d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
  indices = d.argmin(-1)
  x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
  return replace_grad(x_q, x)

class VQGANImage(EMAImage):
  """
  VQGAN latent image representation
  width:  (positive integer) approximate image width in pixels  (will be rounded down to nearest multiple of 16)
  height: (positive integer) approximate image height in pixels (will be rounded down to nearest multiple of 16)
  model:  (VQGAN) vqgan model
  """
  def __init__(self, width, height, model=VQGAN_MODEL, ema_val=0.99):
    if model is None:
      model = VQGAN_MODEL
      if model is None:
        raise RuntimeError("ERROR: model is None and VQGAN is not initialized loaded")

    if VQGAN_IS_GUMBEL:
      e_dim = 256
      n_toks = model.quantize.n_embed
      vqgan_quantize_embedding = model.quantize.embed.weight
    else:
      e_dim = model.quantize.e_dim
      n_toks = model.quantize.n_e
      vqgan_quantize_embedding = model.quantize.embedding.weight

    f = 2**(model.decoder.num_resolutions - 1)
    self.e_dim = e_dim
    self.n_toks = n_toks

    #set up parameter dimensions
    toksX, toksY = width // f, height // f
    sideX, sideY = toksX * f, toksY * f
    self.toksX, self.toksY = toksX, toksY

    #we can't use our own vqgan_quantize_embedding yet because the buffer isn't
    #registered, and we can't register the buffer without the value of z
    
    z = self.rand_latent(vqgan_quantize_embedding = vqgan_quantize_embedding)
    super().__init__(sideX, sideY, z, ema_val)
    self.output_axes = ('n', 's', 'y', 'x')
    self.lr = 0.1

    #extract the parts of VQGAN we need
    self.register_buffer('vqgan_quantize_embedding', vqgan_quantize_embedding, persistent = False) 
    self.vqgan_decode = model.decode
    self.vqgan_encode = model.encode

  def decode(self, z):
    z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1)
    out = self.vqgan_decode(z_q).add(1).div(2)
    return clamp_with_grad(out, 0, 1)

  @torch.no_grad()
  def encode_image(self, pil_image, device=DEVICE, **kwargs):
    pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
    pil_image = TF.to_tensor(pil_image)
    z, *_ = self.vqgan_encode(pil_image.to(device).unsqueeze(0) * 2 - 1)
    self.tensor.set_(z.movedim(1,3))
    self.reset()

  @torch.no_grad()
  def encode_random(self):
    self.tensor.set_(self.rand_latent())
    self.reset()

  def rand_latent(self, device=DEVICE, vqgan_quantize_embedding=None):
    if vqgan_quantize_embedding is None:
      vqgan_quantize_embedding = self.vqgan_quantize_embedding
    n_toks = self.n_toks
    toksX, toksY = self.toksX, self.toksY
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    z = one_hot @ vqgan_quantize_embedding
    z = z.view([-1, toksY, toksX, self.e_dim])
    return z

  @staticmethod
  def init_vqgan(model_name, device = DEVICE):
    global VQGAN_MODEL, VQGAN_NAME, VQGAN_IS_GUMBEL
    if VQGAN_NAME == model_name:
      return
    if model_name not in VQGAN_MODEL_NAMES:
      raise ValueError(f"VQGAN model {model_name} is not supported. Supported models are {VQGAN_MODEL_NAMES}")
    vqgan_config     = f'{model_name}.yaml'
    vqgan_checkpoint = f'{model_name}.ckpt'
    if not path_exists(vqgan_config):
      print(f"WARNING: VQGAN config file {vqgan_config} not found. Initializing download.")
      command = VQGAN_CONFIG_URLS[model_name][0].split(' ', 6)
      subprocess.run(command)
      if not path_exists(vqgan_config):
        print(f"ERROR: VQGAN model {model_name} config failed to download! Please contact model host or find a new one.")
        raise FileNotFoundError(f"VQGAN {model_name} config not found")
    if not path_exists(vqgan_checkpoint):
      print(f"WARNING: VQGAN checkpoint file {vqgan_checkpoint} not found. Initializing download.")
      command = VQGAN_CHECKPOINT_URLS[model_name][0].split(' ', 6)
      subprocess.run(command)
      if not path_exists(vqgan_checkpoint):
        print(f"ERROR: VQGAN model {model_name} checkpoint failed to download! Please contact model host or find a new one.")
        raise FileNotFoundError(f"VQGAN {model_name} checkpoint not found")
        
    VQGAN_MODEL, VQGAN_IS_GUMBEL = load_vqgan_model(vqgan_config, vqgan_checkpoint)
    VQGAN_MODEL = VQGAN_MODEL.to(device)
    VQGAN_NAME  = model_name

  @staticmethod
  def free_vqgan():
    global VQGAN_MODEL
    VQGAN_MODEL = None
