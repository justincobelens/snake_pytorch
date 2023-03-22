import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch

plt.ion()

def plot(scores, mean_scores, train_loss_values):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.gcf()

    plt.subplots(2, sharex=True)
    axs = plt.axes

    plt.title('Training...')

    axs[0].set(ylabel='Score')
    axs[0].plot(scores)
    axs[0].plot(mean_scores)
    # ax1.ylim(ymin=0)
    axs[0].text(len(scores)-1, scores[-1], str(scores[-1]))
    axs[0].text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))


    axs[1].set(xlabel='Number of Games', ylabel='Loss')
    axs[1].plot(np.array(torch.tensor(train_loss_values).numpy()))

