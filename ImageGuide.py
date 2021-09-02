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

  def run_steps(self, n_steps, prompts, loss_augs, stop = -math.inf):
    """
    runs the optimizer
    prompts: (ClipPrompt list) list of prompts
    n_steps: (positive integer) steps to run
    """
    for i in tqdm(range(n_steps)):
      losses = self.train(i, prompts, loss_augs)
      self.update(i, losses)
      if losses['TOTAL'] <= stop:
        break

  def plot_losses(self, ax1, ax2):
    def plot_dataframe(df, ax):
      keys = list(df)
      keys.sort(reverse=True, key = lambda k:df[k].iloc[-1])
      ax.clear()
      df[keys].plot(ax=ax, legend=False)
      #ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
      ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                      bottom=True, top=False, left=True, right=False)
      last_x = df.last_valid_index()
      texts = labelLines(ax.get_lines(), align = False)
      #print(texts)
      adjust_text(texts, text_from_points=False, ax=ax)

    df = self.dataframe
    rel_loss = (df-df.iloc[0]).drop('TOTAL', axis=1)
    plot_dataframe(rel_loss, ax1)
    ax1.set_ylabel('Relative Loss')
    ax1.set_xlabel('Step')
    plot_dataframe(df, ax2)
    ax2.set_ylabel('Absoulte Loss')
    ax2.set_xlabel('Step')

  def update(self, i, losses):
    """
    update hook called ever step
    """
    pass

  def train(self, i, prompts, loss_augs):
    """
    steps the optimizer
    promts: (ClipPrompt list) list of prompts
    """
    self.optimizer.zero_grad()
    z = self.image_rep.decode_training_tensor()
    losses = []
    if self.embedder is not None:
      image_embeds = self.embedder(self.image_rep, input=z)
    for prompt in prompts:
      losses.append(prompt(format_input(image_embeds, self.embedder, prompt)))
    for aug in loss_augs:
      losses.append(aug(format_input(z, self.image_rep, aug)))
    loss = sum(losses)
    loss.backward()
    self.optimizer.step()
    self.image_rep.update()
    loss_dict = {str(prompt):float(loss) for prompt, loss in zip(prompts+loss_augs+["TOTAL"],losses+[loss])}
    if self.dataframe is None:
      self.dataframe = pd.DataFrame(loss_dict, index=[i])
      self.dataframe.index.name = 'Step'
    else:
      self.dataframe = self.dataframe.append(pd.DataFrame(loss_dict, index=[i]), ignore_index=False)
      self.dataframe.index.name = 'Step'
    return loss_dict
