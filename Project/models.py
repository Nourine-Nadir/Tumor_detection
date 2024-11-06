import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)

class Naive_net(nn.Module):
    def __init__(self,
                 lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims,
                 fc3_dims,
                 n_output):
        super(Naive_net, self).__init__()

        self.fc1 = nn.Linear(input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.fc4 = nn.Linear(fc2_dims, fc2_dims)
        self.fc5 = nn.Linear(fc2_dims, fc3_dims)
        self.fc6 = nn.Linear(fc3_dims, n_output)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.apply(weights_init_)

        self.loss= nn.CrossEntropyLoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.99 ** epoch, (lr*1e-1 / lr))
        )

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f'device used: {self.device}')
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.FloatTensor(x).to(self.device)
        elif isinstance(x, T.Tensor):
            x = x.to(self.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout2(x)
        x = F.relu(self.fc5(x))
        x = F.softmax(self.fc6(x), dim=-1)

        return x

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]