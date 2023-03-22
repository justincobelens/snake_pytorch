# import matplotlib.pyplot as plt
# from IPython import display
# import numpy as np
# import torch
#
# plt.ion()
#
# def plot(scores, mean_scores, train_loss_values):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.gcf()
#
#     plt.subplots(2, sharex=True)
#     axs = plt.axes
#
#     plt.title('Training...')
#
#     axs[0].set(ylabel='Score')
#     axs[0].plot(scores)
#     axs[0].plot(mean_scores)
#     # ax1.ylim(ymin=0)
#     axs[0].text(len(scores)-1, scores[-1], str(scores[-1]))
#     axs[0].text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
#
#
#     axs[1].set(xlabel='Number of Games', ylabel='Loss')
#     axs[1].plot(np.array(torch.tensor(train_loss_values).numpy()))

import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
import numpy as np
import torch

plt.ion()


def create_plots():
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Training...')

    axs[0].set(ylabel='Score')
    axs[1].set(xlabel='Number of Games', ylabel='Loss')

    return fig, axs


def update_plots(fig, axs, scores, mean_scores, train_loss_values):
    clear_output(wait=True)

    axs[0].clear()
    axs[1].clear()

    axs[0].set(ylabel='Score')
    axs[0].plot(scores, label='Scores')
    axs[0].plot(mean_scores, label='Mean Scores')
    axs[0].legend(loc='upper left')
    axs[0].text(len(scores) - 1, scores[-1], str(scores[-1]))
    axs[0].text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    axs[1].set(xlabel='Number of Games', ylabel='Loss')
    axs[1].plot(np.array(torch.tensor(train_loss_values).numpy()), label='Loss')
    axs[1].legend(loc='upper left')

    display.display(fig)
