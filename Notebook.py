#this library is designed for use with google colab runtimes.
#This file defines utility functions for use with notebooks.

#https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook():
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    elif shell == 'Shell':
      return True   # Google Colab
    else:
      print("DEGBUG: unknown shell type:",shell)
      return False
  except NameError:
    return False    # Probably standard Python interpreter

def change_tqdm_color():
  if not is_notebook():
    return

  from IPython import display
  from IPython.display import HTML
  def set_css_in_cell_output():
    display.display(HTML('''
      <style>
          .jupyter-widgets {color: #d5d5d5 !important;}
          .widget-label {color: #d5d5d5 !important;}
      </style>
    '''))
  get_ipython().events.register('pre_run_cell', set_css_in_cell_output)

if is_notebook():
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

def get_tqdm():
  return tqdm

def get_last_file(directory, pattern):
  import os, re
  def key(f):
    index = re.match(pattern, f).group('index')
    return 0 if index == '' else int(index)
  files = [f for f in os.listdir(directory) if re.match(pattern, f)]
  if len(files) == 0:
    return None, None
  files.sort(key=key)
  index = key(files[-1])
  return files[-1], index

def get_next_file(directory, pattern, templates):
  import os, re
  files = [f for f in os.listdir(directory) if re.match(pattern, f)]
  if len(files) == 0:
    return templates[0], 0
  def key(f):
    index = re.match(pattern, f).group('index')
    return 0 if index == '' else int(index)
  files.sort(key=key)
  n = len(templates)-1
  for i, f in enumerate(files):
    index = key(f)
    if i != index:
      return (templates[0],0) if i == 0 else (re.sub(pattern, lambda m:f"{m.group('pre')}{i}{m.group('post')}", templates[min(i,n)]), i)
  return re.sub(pattern, lambda m:f"{m.group('pre')}{i+1}{m.group('post')}", templates[min(i,n)]), i+1


def make_hbox(im, fig):
  #https://stackoverflow.com/questions/51315566/how-to-display-the-images-side-by-side-in-jupyter-notebook/51316108
  import io
  import ipywidgets as widgets
  from ipywidgets import Layout
  with io.BytesIO() as buf:
    im.save(buf, format="png")
    buf.seek(0)
    wi1 = widgets.Image(value=buf.read(), format='png', layout = Layout(border='0',margin='0',padding='0'))
  with io.BytesIO() as buf:
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    wi2 = widgets.Image(value=buf.read(), format='png', layout = Layout(border='0',margin='0',padding='0'))
  return widgets.HBox([wi1, wi2], layout = Layout(border='0',margin='0',padding='0', align_items='flex-start'))

def load_settings(settings_string, random_seed = True):
  import json, random
  from bunch import Bunch
  params = Bunch(json.loads(settings_string))
  if random_seed or params.seed is None:
    params.seed = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)
    print("using seed:", params.seed)
  return params

def write_settings(settings_dict, f):
  import json
  json.dump(settings_dict, f)
  f.write('\n\n')
  params = settings_dict
  scenes = [(params.scene_prefix + scene + params.scene_suffix).strip() for scene in params.scenes.split('||') if scene]
  for i,scene in enumerate(scenes):
    frame = i * params.steps_per_scene/params.save_every
    f.write(str(f'{frame:.2f}: {scene}'.encode('utf-8', 'ignore')))
    f.write('\n')

def save_settings(settings_dict, path):
  with open(path, 'w+') as f:
    write_settings(settings_dict, f)

def save_batch(settings_list, path):
  from bunch import Bunch
  with open(path, 'w+') as f:
    for batch_index, settings_dict in enumerate(settings_list):
      f.write(f'batch_index: {batch_index}')
      f.write('\n')
      write_settings(Bunch(settings_dict), f)
      f.write('\n\n')


CLIP_MODEL_NAMES = None
def load_clip(params):
  from pytti import Perceptor
  global CLIP_MODEL_NAMES
  if CLIP_MODEL_NAMES is not None:
    last_names = CLIP_MODEL_NAMES
  else:
    last_names = []
  CLIP_MODEL_NAMES = []
  if params.RN50x4:
    CLIP_MODEL_NAMES.append("RN50x4")
  if params.RN50:
    CLIP_MODEL_NAMES.append("RN50")
  if params.ViTB32:
    CLIP_MODEL_NAMES.append("ViT-B/32")
  if params.ViTB16:
    CLIP_MODEL_NAMES.append("ViT-B/16")
  if last_names != CLIP_MODEL_NAMES or Perceptor.CLIP_PERCEPTORS is None:
    if CLIP_MODEL_NAMES == []:
      Perceptor.free_clip()
      raise RuntimeError("Please select at least one CLIP model")
    Perceptor.free_clip()
    print("Loading CLIP...")
    Perceptor.init_clip(CLIP_MODEL_NAMES)
    print("CLIP loaded.")

def get_frames(path):
  """reads the frames of the mp4 file `path` and returns them as a list of PIL images"""
  import imageio, subprocess
  from PIL import Image
  from os.path import exists as path_exists
  if not path_exists(path+'_converted.mp4'):
    print(f'Converting {path}...')
    subprocess.run(['ffmpeg', '-i', path, path+'_converted.mp4'])
    print(f'Converted {path} to {path}_converted.mp4.')
    print(f'WARNING: future runs will automatically use {path}_converted.mp4, unless you delete it.')
  vid = imageio.get_reader(path+'_converted.mp4',  'ffmpeg')
  n_frames = vid._meta['nframes']
  print(f'loaded {n_frames} frames. for {path}')
  return vid

def build_loss(weight_name, weight, name, img, pil_target):
  from pytti.LossAug import LOSS_DICT
  weight_name, suffix = weight_name.split('_', 1)
  if weight_name == 'direct':
    Loss = type(img).get_preferred_loss()
  else:
    Loss = LOSS_DICT[weight_name]
  out = Loss.TargetImage(f"{weight_name} {name}:{weight}", img.image_shape, pil_target)
  out.set_enabled(pil_target is not None)
  return out

def format_params(params, *args):
  return [params[x] for x in args]

rotoscopers = []

def clear_rotoscopers():
  global rotoscopers
  rotoscopers = []

def update_rotoscopers(frame_n):
  global rotoscopers
  for r in rotoscopers:
    r.update(frame_n)

from PIL import Image

class Rotoscoper:
  def __init__(self, video_path, target = None, thresh = None):
    global rotoscopers
    if video_path[0] == '-':
      video_path = video_path[1:]
      inverted = True
    else:
      inverted = False
    
    self.frames = get_frames(video_path)
    self.target = target
    self.inverted = inverted
    rotoscopers.append(self)
  def update(self, frame_n):
    if self.target is None:
      return
    mask_pil = Image.fromarray(self.frames.get_data(frame_n)).convert('L')
    self.target.set_mask(mask_pil, self.inverted)

