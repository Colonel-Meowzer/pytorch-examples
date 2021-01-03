#!/usr/bin/env python

import pytest
from data import category_from_output, _random_choice, random_training_example
from util import line_to_tensor, all_letters, n_letters, all_categories, n_categories
from model import RNN
import torch

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


def test_Given_RNNoutput_When_categoryfromoutput_Then_returnTopcategories():
    input = line_to_tensor("Albert")
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input[0], hidden)

    assert category_from_output(output, all_categories, 1)[0] in all_categories

def test_Given_categorylist_When_randomchoice_Then_returncategory():
    ls = ["foo", "bar", "baz"]
    res = _random_choice(ls)
    assert res in ls
    assert isinstance(res, str)

def test_Given_categoriesWithData_When_randomTrainingSample_Then_returnSample():
    categories = ["persons", "places", "things"]

    data = {
        "persons": ["Bob", "Janet"],
        "places": ["Morocco", "Barcelona"],
        "things": ["dog", "mug"]
    }

    category, line, category_tensor, line_tensor = random_training_example(categories, data)

    assert category in categories
    

