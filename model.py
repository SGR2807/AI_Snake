import torch as tch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(saanp, input_layer_size, hidden_size, output_layer_size):
        super().__init__()
        saanp.linear1 = nn.Linear(input_layer_size, hidden_size)
        saanp.linear2 = nn.Linear(hidden_size, output_layer_size)

    def forward(saanp, x):
        x = F.relu(saanp.linear1(x))
        x = saanp.linear2(x)
        return x

    def save(saanp, file_name='model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        tch.save(saanp.state_dict(), file_name)


class QTrainer:
    def __init__(saanp, model, lr, gamma):
        saanp.lr = lr
        saanp.model = model
        saanp.gamma = gamma
        saanp.optimizer = optim.Adam(model.parameters(), lr=saanp.lr)
        saanp.criterion = nn.MSELoss()

    def train_step(saanp, shape, action, profit, agla_state, done):
        shape = tch.tensor(shape, dtype=tch.float)
        agla_state = tch.tensor(agla_state, dtype=tch.float)
        profit = tch.tensor(profit, dtype=tch.float)
        action = tch.tensor(action, dtype=tch.long)


        if len(shape.shape) == 1:
            # (1, x)
            #update profit
            #update action
            shape = tch.unsqueeze(shape, 0)
            agla_state = tch.unsqueeze(agla_state, 0)
            profit = tch.unsqueeze(profit, 0)
            action = tch.unsqueeze(action, 0)
            done = (done, )

        # 1: predicted Q values with current shape
        pred = saanp.model(shape)

        aim = pred.clone()
        for idx in range(len(done)):
            Q_new = profit[idx]
            if not done[idx]:
                Q_new = profit[idx] + saanp.gamma * tch.max(saanp.model(agla_state[idx]))

            aim[idx][tch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        saanp.optimizer.zero_grad()
        loss = saanp.criterion(aim, pred)
        loss.backward()

        saanp.optimizer.step()
