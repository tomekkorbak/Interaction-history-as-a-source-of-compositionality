import torch
import torch.nn as nn


class Sender(nn.Module):
    def __init__(self, n_hidden, vision_module):
        super(Sender, self).__init__()
        self.vision = vision_module
        self.fc = nn.Linear(25, n_hidden)

    def forward(self, input):
        with torch.no_grad():
            embedding = self.vision.embedd(input)
        return self.fc(embedding)


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features*4)
        self.fc2_1 = nn.Linear(n_features*4, n_features)
        self.fc2_2 = nn.Linear(n_features*4, n_features)

    def forward(self, input, _):
        hidden = torch.nn.functional.leaky_relu(self.fc1(input))
        return self.fc2_1(hidden), self.fc2_2(hidden)
