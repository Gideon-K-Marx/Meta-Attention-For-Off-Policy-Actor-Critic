import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action
        nn.init.kaiming_normal_(self.l3.weight)

    def forward(self, x):
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Actor_Feature(nn.Module):
    def __init__(self, state_dim):
        super(Actor_Feature, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.kaiming_normal_(self.l2.weight)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.kaiming_normal_(self.l2.weight)
        nn.init.kaiming_normal_(self.l3.weight)
        nn.init.kaiming_normal_(self.l4.weight)
        nn.init.kaiming_normal_(self.l5.weight)
        nn.init.kaiming_normal_(self.l6.weight)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def features(self, x, u):
        xu = torch.cat([x, u], 1)
        f1 = F.relu(self.l1(xu))
        f1 = F.relu(self.l2(f1))

        return f1

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

# Meta-Critic of Twin Delayed Deep Deterministic Policy Gradient (TD3_MC)
class Meta_Critic(nn.Module):
    def __init__(self, hidden_dim):
        super(Meta_Critic, self).__init__()
        self.fc1 = nn.Linear(hidden_dim,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.functional.softplus(self.fc3(x))
        return torch.mean(x)

# Meta-Attention of Twin Delayed Deep Deterministic Policy Gradient (TD3_MATT or TD3_MAT4)
class Meta_Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Meta_Attention, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = x * 2
        return x
