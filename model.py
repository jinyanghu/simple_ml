import torch
from torch.utils.data import DataLoader
import torch.nn as nn      

class Logistics(nn.Module):
  def __init__(self):
    super(Logistics , self).__init__()
    self.neuron = 128
    self.input_size = 27
    self.output_size = 4
    self.logtics = nn.Sequential(
        nn.Linear(self.input_size, self.output_size),
        )
  
  def forward(self , x):
    x = self.logtics(x)
    return x

class DNN(nn.Module):
  def __init__(self):
    super(DNN , self).__init__()
    self.neuron = 512
    self.input_size = 27
    self.output_size = 4
    self.drop_rate = 0.4
    self.dnn = nn.Sequential(
        nn.Linear(self.input_size , self.neuron),
        nn.BatchNorm1d(self.neuron),
        nn.Dropout(self.drop_rate),
        nn.ReLU(),
        
        nn.Linear(self.neuron , self.neuron),
        nn.BatchNorm1d(self.neuron),
        nn.Dropout(self.drop_rate),
        nn.ReLU(),

        nn.Linear(self.neuron , self.neuron),
        nn.BatchNorm1d(self.neuron),
        nn.Dropout(self.drop_rate),
        nn.ReLU(),


        nn.Linear(self.neuron , self.output_size),
        nn.ReLU()
        )

  def forward(self, x):
    x = self.dnn(x)
    return x



class CNN(nn.Module):
  def __init__(self):
    super(CNN , self).__init__()
    self.neuron = 512
    self.input_size = 27
    self.output_size = 4
    self.cnn = nn.Sequential(
        nn.Conv1d(1,8,3,stride=1,padding=1),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        nn.Conv1d(8,16,3,stride=1,padding=1),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(16,32,3,stride=1,padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        )
    
    self.fc = nn.Sequential(nn.Linear(864 , self.neuron),
                            nn.BatchNorm1d(self.neuron),
                            nn.Dropout(0.5),
                            nn.ReLU(),
                            
                            nn.Linear(self.neuron,self.neuron),
                            nn.BatchNorm1d(self.neuron),
                            nn.Dropout(0.5),
                            nn.ReLU(),
                            
                            nn.Linear(self.neuron , self.output_size),
                            nn.ReLU(),
                            )

  def forward(self, x):
    x = self.cnn(x)
    #print(x.shape)
    x = x.flatten(1)
    #print(x.shape)
    x = self.fc(x)
    return x


