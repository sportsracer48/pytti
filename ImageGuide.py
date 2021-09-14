from torch import optim, nn
from pytti.Notebook import tqdm
from pytti import *
import pandas as pd
import math

from labellines import labelLines
from adjustText import adjust_text


class DirectImageGuide():
  """
  Image guide that uses an optimizer and torch autograd to optimize an image representation
  Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun)
  image_rep: (DifferentiableImage) image representation
  embedder: (Module)               image embedder
  optimizer: (Class)               optimizer class to use. Defaults to Adam
  all other arguments are passed as kwargs to the optimizer.
  """
  def __init__(self, image_rep, embedder, optimizer = None, lr = None, **optimizer_params):
    self.image_rep = image_rep
    self.embedder = embedder
    if lr is None:
      lr = image_rep.lr
    optimizer_params['lr']=lr
    if optimizer is None:
      self.optimizer = optim.Adam(image_rep.parameters(), **optimizer_params)
    else:
      self.optimizer = optimizer
    self.dataframe = None

  def run_steps(self, n_steps, 
                      prompts, image_prompts, interp_prompts, loss_augs, 
                      stop = -math.inf, interp_steps = 0, 
                      i_offset = 0, skipped_steps = 0):
    """
    runs the optimizer
    prompts: (ClipPrompt list) list of prompts
    n_steps: (positive integer) steps to run
    returns: the number of steps run
    """
    for i in tqdm(range(n_steps)):
      losses = self.train(i+skipped_steps,
                          prompts, image_prompts, interp_prompts, loss_augs, 
                          interp_steps = interp_steps)
      self.update(i+i_offset, losses, i+skipped_steps)
      if losses['TOTAL'] <= stop:
        break
    return i+1

  def clear_dataframe(self):
    self.dataframe = None

  def plot_losses(self, ax1, ax2):
    def plot_dataframe(df, ax, color_dict = {}, remove_total = False):
      keys = list(df)
      keys.sort(reverse=True, key = lambda k:df[k].iloc[-1])
      ax.clear()
      df[keys].plot(ax=ax, legend=False)
      #ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
      ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                      bottom=True, top=False, left=True, right=False)
      last_x = df.last_valid_index()
      lines = ax.get_lines()
      for l in lines:
        label = l.get_label()
        if label in color_dict:
          l.set_color(color_dict[label])

      colors = [l.get_color() for l in lines]
      labels = [l.get_label() for l in lines]
      if remove_total:
        [l.remove() for l in lines if l.get_label() == 'TOTAL']
      ax.relim()
      ax.autoscale_view()
        
      texts = labelLines(ax.get_lines(), align = False)
      #print(texts)
      adjust_text(texts, text_from_points=False, ax=ax)
      return dict(zip(labels, colors))

    df = self.dataframe
    if df is None or len(df.index) < 2:
      return False
    rel_loss = (df-df.iloc[0])
    color_dict = plot_dataframe(rel_loss, ax1, remove_total = True)
    ax1.set_ylabel('Relative Loss')
    ax1.set_xlabel('Step')
    plot_dataframe(df, ax2, color_dict = color_dict)
    ax2.set_ylabel('Absoulte Loss')
    ax2.set_xlabel('Step')
    return True

  def update(self, i, losses, stage_i):
    """
    update hook called ever step
    """
    pass

  def train(self, i, prompts, image_prompts, interp_prompts, loss_augs, interp_steps = 0):
    """
    steps the optimizer
    promts: (ClipPrompt list) list of prompts
    """
    self.optimizer.zero_grad()
    z = self.image_rep.decode_training_tensor()
    losses = []
    if self.embedder is not None:
      image_embeds, offsets, sizes = self.embedder(self.image_rep, input = z)

    if i < interp_steps:
      t = i/interp_steps
      interp_losses = [prompt(format_input(image_embeds, self.embedder, prompt),
                              format_input(offsets, self.embedder, prompt),
                              format_input(sizes, self.embedder, prompt))*(1-t) for prompt in interp_prompts]
    else:
      t = 1
      interp_losses = [0] 

    for prompt in prompts:
      losses.append(prompt(format_input(image_embeds, self.embedder, prompt),
                           format_input(offsets, self.embedder, prompt),
                           format_input(sizes, self.embedder, prompt))*t)
    for prompt in image_prompts:
      losses.append(prompt(format_input(image_embeds, self.embedder, prompt),
                           format_input(offsets, self.embedder, prompt),
                           format_input(sizes, self.embedder, prompt)))
    for aug in loss_augs:
      losses.append(aug(format_input(z, self.image_rep, aug)))
    image_loss = self.image_rep.image_loss()
    for img_loss in image_loss:
      losses.append(img_loss(self.image_rep))

    loss = sum(losses)+sum(interp_losses)
    loss.backward()
    self.optimizer.step()
    self.image_rep.update()
    loss_dict = {str(prompt):float(loss) for prompt, loss in zip(prompts+image_prompts+loss_augs+image_loss+["TOTAL"],losses+[loss])}
    if self.dataframe is None:
      self.dataframe = pd.DataFrame(loss_dict, index=[i])
      self.dataframe.index.name = 'Step'
    else:
      self.dataframe = self.dataframe.append(pd.DataFrame(loss_dict, index=[i]), ignore_index=False)
      self.dataframe.index.name = 'Step'
    return loss_dict
