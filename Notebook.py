

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