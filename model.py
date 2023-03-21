import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class LinearQnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = Path('model')
        model_folder_path.mkdir(parents=True, exist_ok=True)
        file_path = model_folder_path / Path(file_name)
        torch.save(self.state_dict(), f=file_path)

    def load(self, file_name='model.pth'):
        model_folder_path = Path('model')
        file_path = model_folder_path / Path(file_name)

        if file_path.exists():
            self.load_state_dict(torch.load(file_path))
            self.eval()

    # TODO: fix checkpoint save and load
    def save_checkpoint(self, epoch, optimizer_state_dict, loss, file_name='checkpoint.tar'):
        model_folder_path = Path('model')
        model_folder_path.mkdir(parents=True, exist_ok=True)

        file_path = model_folder_path / Path(file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state_dict.state_dict(),
            'loss': loss
        }, f=file_path)

    # TODO: fix checkpoint save and load
    def load_checkpoint(self, file_name='checkpoint.tar'):
        model_folder_path = Path('model')
        file_path = model_folder_path / Path(file_name)

        if file_path.exists():
            checkpoint = torch.load(file_path)
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

            self.model.eval()

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