from torch import optim
from tqdm.notebook import tqdm
from pytti import *


class DirectImageGuide():
  """
  Image guide that uses an optimizer and torch autograd to optimize an image representation
  Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun)
  image_rep: (DifferentiableImage) image representation
  embedder: (Module)               image embedder
  optimizer: (Class)               optimizer class to use. Defaults to Adam
  all other arguments are passed as kwargs to the optimizer.
  """
  def __init__(self, image_rep, embedder, tv_weight = 0.15, optimizer = optim.Adam, lr = None, weight_decay = 0.0, **optimizer_params):
    self.image_rep = image_rep
    self.embedder = embedder
    if lr is None:
      lr = image_rep.lr
    optimizer_params['lr']=lr
    optimizer_params['weight_decay']=weight_decay
    self.optimizer = optimizer(image_rep.parameters(), **optimizer_params)
    self.tv_weight = tv_weight

  def run_steps(self, prompts, n_steps):
    """
    runs the optimizer
    prompts: (ClipPrompt list) list of prompts
    n_steps: (positive integer) steps to run
    """
    for i in tqdm(range(n_steps)):
      losses = self.train(prompts, i)
      self.update(i, losses)

  def update(self, i, losses):
    """
    update hook called ever step
    """
    pass

  def train(self, prompts, i):
    """
    steps the optimizer
    promts: (ClipPrompt list) list of prompts
    """
    self.optimizer.zero_grad()
    z = self.image_rep.decode_training_tensor()
    losses = []
    image_embeds = self.embedder(self.image_rep, input=z)
    for prompt in prompts:
        losses.append(prompt(format_input(image_embeds, self.embedder, prompt)))
    losses.append(tv_loss(z)*self.tv_weight)
    loss = sum(losses)
    loss.backward()
    self.optimizer.step()
    self.image_rep.update()
    return ', '.join(f"{str(prompt)}:{float(loss):.3}" for prompt, loss in zip(prompts+["TV LOSS","TOTAL"],losses+[loss]))
