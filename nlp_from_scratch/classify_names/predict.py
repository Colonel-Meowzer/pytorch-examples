from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import time
import math

import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import RNN
from util import list_files, unicode_to_ascii, line_to_tensor, letter_to_tensor, all_letters, n_letters, all_categories, n_categories, category_lines
from data import category_from_output, random_training_example


#hidden = torch.zeros(1, n_hidden)

#output, next_hidden = rnn(input[0], hidden)
#print(output)


#for i in range(10):
#    category, line, category_tensor, line_tensor = random_training_example()
#    print('category =', category, '/ line =', line)


criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

# TODO: create a class which inherits RNN to store and train data OR use partial function.
def train(category_tensor, line_tensor, criterion, learning_rate, rnn):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def main():
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    categories = all_categories
    data = category_lines

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example(categories, data)
        output, loss = train(category_tensor, line_tensor, criterion, learning_rate, rnn)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, categories, 1)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)

    # Confusion Matrix

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Just return an output given a line
    def evaluate(line_tensor):
        hidden = rnn.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(categories, data)
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output, categories, 1)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


if __name__ == '__main__':
    main()

