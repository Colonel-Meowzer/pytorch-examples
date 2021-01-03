from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch

def list_files(path): return glob.glob(path)



# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language

# Read a file and split into lines
def read_and_encode_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def get_categories(path_glob):
    all_categories = []
    category_lines = {}
    for filename in list_files(path_glob):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_and_encode_lines(filename)
        category_lines[category] = lines
    return category_lines, all_categories


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

category_lines, all_categories = get_categories("data/names/*.txt")
n_categories = len(all_categories)

