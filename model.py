import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MODELS_DIR_PATH = Path('models')
MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

# TODO: Add something to prevent overwriting older models by accident


class LinearQnet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        FILE_PATH = MODELS_DIR_PATH / Path(file_name)
        torch.save(self.state_dict(), f=FILE_PATH)

    def load(self, file_name='model.pth'):
        FILE_PATH = MODELS_DIR_PATH / Path(file_name)

        if FILE_PATH.exists():
            self.load_state_dict(torch.load(FILE_PATH))
            # self.train()

    # TODO: fix checkpoint save and load
    def save_checkpoint(self, epoch, optimizer_state_dict, loss, file_name='checkpoint.tar'):
        FILE_PATH = MODELS_DIR_PATH / Path(file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state_dict.state_dict(),
            'loss': loss
        }, f=FILE_PATH)

    # TODO: fix checkpoint save and load
    def load_checkpoint(self, file_name='checkpoint.tar'):
        FILE_PATH = MODELS_DIR_PATH / Path(file_name)

        if FILE_PATH.exists():
            checkpoint = torch.load(FILE_PATH)
            return checkpoint
        else:
            return None


class QTrainer:
    def __init__(self, model, lr, gamma, checkpoint=None):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        # TODO: checkpoint doesnt work
        if checkpoint:
            self.model = self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']

            # self.model.train()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)  # redefine as a tuple with 1 value

        # 1: predicted Q values with current state
        # self.model.train()
        pred = self.model(state)

        # 2: Q_nes = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()

        self.optimizer.step()

        return loss
