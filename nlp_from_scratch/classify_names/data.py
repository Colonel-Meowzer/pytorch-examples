
import random
import torch
from util import line_to_tensor


def category_from_output(output, categories, n):
    top_n, top_i = output.topk(n)
    category_i = top_i[0].item()
    return categories[category_i], category_i


def _random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(categories, data):
    category = _random_choice(categories)
    line = _random_choice(data[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor
