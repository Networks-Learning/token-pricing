import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from transformers import AutoTokenizer
import torch

def get_fig_dim(width, fraction=1, aspect_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    aspect_ratio: float, optional
            Aspect ratio of the figure

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        # If not specified, set the aspect ratio equal to the Golden ratio (https://en.wikipedia.org/wiki/Golden_ratio)
        aspect_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    font_serif: string, optional
		Set the desired font family
    mathtext_font: float, optional
    	Set the desired math font family
    font_size: int, optional
    	Set the large font size
    small_font_size: int, optional
    	Set the small font size
    usetex: boolean, optional
        Use tex for strings
    """

    if small_font_size is None:
        small_font_size = font_size

    params = {
        'backend': 'ps',
        'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
            
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        
        # Optionally set a smaller font size for legends and tick labels
        'legend.fontsize': small_font_size,
        'legend.title_fontsize': small_font_size,
        'xtick.labelsize': small_font_size,
        'ytick.labelsize': small_font_size,
        
        'text.usetex': usetex,    
        'font.family' : 'serif',
        'font.serif' : font_serif,
        'mathtext.fontset' : mathtext_font
    }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)



def is_in_voc(string, model_name="meta-llama/Llama-3.2-1B"):
    """Tests whether a target string is in a model vocabulary"""
# Load the tokenizer for LLaMA 3.2-1B

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the vocabulary: token -> ID

    vocab = tokenizer.get_vocab()

    # Reverse vocab: ID -> token string (for decoding)
    inv_vocab = {v: k for k, v in vocab.items()}

    # Go through all token strings, decode them, and compare
    found_matches = []
    for token, token_id in vocab.items():
        decoded = tokenizer.decode([token_id])
        if decoded == string:
            found_matches.append((token_id, token))

    if found_matches:
        print("Found exact match(es):")
        for token_id, token_str in found_matches:
            print(f"  Token ID: {token_id}, Token string: {repr(token_str)}")
    else:
        print("Not a token in the vocabulary.")
        
def sort_tensors(x,y):
    """
    Sorts and aggregates values in tensors.
    """


    unique_x = torch.unique(x)

    # Initialize an empty tensor to accumulate the summed y values
    sorted_y = torch.zeros_like(unique_x, dtype=torch.float64)

    # Sum the corresponding y values for each unique x
    for i, ux in enumerate(unique_x):
        sorted_y[i] = y[x == ux].sum()

    return unique_x, sorted_y