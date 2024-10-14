import torch
class FCNet(torch.nn.Linear):
  def __init__(self, n_hidden_neurons):
    super(FCNet,self).__init__(28*28, 10)
    self.fc1 = torch.nn.Linear(28*28, n_hidden_neurons)
    self.act1 = torch.nn.Tanh()
    self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

  def forward(self, X):
    X = self.fc1(X)
    X = self.act1(X)
    X = self.fc2(X)
    return X