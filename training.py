from pathlib import Path
import numpy as np
import os 
from data_process import *
import sklearn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import * 

class Data_load:
  
  def __init__(self , path , data_type='train'):
    self.load_data = Open_data(path , data_type=data_type)[:]
    self.attribute = np.array(self.load_data[0] , dtype = np.float64)
    self.label = np.array(self.load_data[1] , dtype = np.int32)
    self.all_normalize([2 , 4 , 5])

  def normalize(self, col):  
    col_mean = np.transpose(self.attribute)[col].mean()
    col_std  = np.transpose(self.attribute)[col].std()
    for i in range(len(self.attribute)): 
      self.attribute[i][col] = (self.attribute[i][col] - col_mean) / col_std

  def all_normalize(self , col_list):
    for idx in col_list:
      self.normalize(idx)  

  def __getitem__(self, key):
    return self.attribute[key] , self.label[key]      
  
  def __len__(self):
    return len(self.attribute)

def split_train_val(data_attr , data_label , data_type='train' ,divider=20):
  data_num = len(data_attr)
  idx_tr = []
  idx_val = []
  for i in range(data_num):
    if i % divider == 0:
      idx_val.append(i)
    else:
      idx_tr.append(i)
  
  if data_type == 'train':
    return data_attr[idx_tr] , data_label[idx_tr]
  
  if data_type == 'val':
    return data_attr[idx_val] , data_label[idx_val]
      
      
  

training_set = Data_load('Train.csv' , 'train')[:]
testing_set = Data_load('Test.csv' , 'test')[:]

train_attr , train_label = split_train_val(training_set[0] , training_set[1] , data_type='train')
val_attr , val_label = split_train_val(training_set[0] , training_set[1] , data_type='val')


test_data , ground_truth = testing_set[0] , testing_set[1]
test_data = torch.tensor(test_data , dtype=torch.float32)

device = torch.cuda.is_available()

def make_dataset(data_attr , data_label , reshape_size = False):
  combine_data = []
  if reshape_size:
    data_attr = data_attr.reshape((len(data_attr) , 1 , 27))  
  
  for i in range(len(data_attr)):    
    combine_data.append((torch.tensor(data_attr[i] , dtype = torch.float32), torch.tensor(data_label[i] , dtype = torch.long)))
  return combine_data

#train_set = make_dataset(train_attr , train_label , True)
#val_set = make_dataset(val_attr , val_label , True)

class Train_nn:
  
  def __init__(self , train_set,  
               val_set,
               model_path,
               batch_size=128,
               model = 'log',
               learning_rate = 1e-5,
               weight_decay = 1e-7,
               epoch = 100):
      
      
    self.train_loader = DataLoader(train_set , batch_size=batch_size , shuffle=True)
    self.val_loader = DataLoader(val_set, batch_size=batch_size , shuffle=False)
    
    self.model_path = model_path
    self.device = self.get_device()
    
    if model == 'log':
      self.model = Logistics().to(self.device)
    elif model == 'dnn':
      self.model = DNN().to(self.device)
    elif model == 'cnn':
      self.model = CNN().to(self.device)  
    
    self.criteria = nn.CrossEntropyLoss()
    self.lr = learning_rate
    self.epoch = epoch
    self.optimizer = torch.optim.Adam(self.model.parameters() , lr=self.lr , weight_decay=weight_decay)
  
  def get_device(self):  
    return 'cuda' if torch.cuda.is_available() else 'cpu'
      
  def train(self):
      best_acc = 0.0
      for epoch in range(self.epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        self.model.train()
        for i , data in enumerate(self.train_loader):
          inputs , labels = data
          inputs , labels = inputs.to(self.device) , labels.to(self.device)
          self.optimizer.zero_grad()
          outputs = self.model(inputs)
          batch_loss = self.criteria(outputs , labels)
          _ , train_pred = torch.max(outputs , 1)
          batch_loss.backward()
          self.optimizer.step()
          
          train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
          train_loss += batch_loss.item()  
        
        if len(val_set) > 0:
          self.model.eval()
          with torch.no_grad():
            for i , data in enumerate(self.val_loader):
              inputs , labels = data 
              inputs , labels = inputs.to(self.device), labels.to(self.device)
              outputs = self.model(inputs)
              batch_loss = self.criteria(outputs , labels)
              _ , val_pred = torch.max(outputs , 1)
            
              val_acc += (val_pred.cpu() == labels.cpu()).sum()
              val_loss += batch_loss.item()
          
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc {:3.6f} loss: {:3.6f}'.format(
                epoch+1 , self.epoch , train_acc/len(train_set) , train_loss/len(self.train_loader) , val_acc/len(val_set) , val_loss/len(self.val_loader)))
          
            
            if val_acc > best_acc:
              best_acc = val_acc
              torch.save(self.model.state_dict(), self.model_path)
              print('saving model with acc {:3f}'.format(best_acc/len(val_set) , val_loss/len(self.val_loader)))
        
        else:
          print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc {:3.6f} loss: {:3.6f}'.format(
                epoch+1 , self.epoch , train_acc/len(train_set) , train_loss/len(self.train_loader) , val_acc/len(val_set) , val_loss/len(self.val_loader)))
        
        if len(val_attr) == 0:
          torch.save(self.model.state_dict(), self.model_path)
          print('saving model at last epoch')
      
class Test_nn:
  
  def __init__(self , test_attr  , model_path , out_name , model='log' , batch_size=128):
    self.test_loader = DataLoader(test_attr , batch_size= batch_size , shuffle=False)
    self.device = self.get_device()
    self.predict_name = out_name
    
    if model == 'log':
      self.model = Logistics().to(self.device)
    elif model == 'dnn':
      self.model = DNN().to(self.device)
    elif model == 'cnn':
      self.model = CNN().to(self.device)  
    
    self.model.load_state_dict(torch.load(model_path))
  
  def predict(self):
    predict = []
    self.model.eval()
    with torch.no_grad():
      for i , data in enumerate(self.test_loader):
        inputs = data
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        _ , test_pred = torch.max(outputs , 1)
        
        for y in test_pred.cpu().numpy():
          predict.append(y)
    
    
    with open(self.predict_name , 'w') as f:
      f.write('Class\n')
      for i in predict:
        print(i , file=f)

  def get_device(self):  
    return 'cuda' if torch.cuda.is_available() else 'cpu'      


train_set = make_dataset(train_attr , train_label , False)
val_set = make_dataset(val_attr , val_label , False)


logit = Train_nn(
    train_set,val_set,
    model_path='logit_model.pt',
    model='log' , epoch=500)

logit.train()

logit_test = Test_nn(test_data, model_path='logit_model.pt', 
                     out_name='predict_logit.csv',model='log')
logit_test.predict()

print('Finished Logistics!')

dnn = Train_nn(train_set , val_set,
               model_path='cnn_model.pt',
               model = 'dnn' , epoch=500)

dnn.train()

dnn_test = Test_nn(test_data , model_path='dnn_model.pt',
                   out_name='predict_dnn.csv' , model = 'dnn')
dnn_test.predict()

print('Finished DNN!')

train_set = make_dataset(train_attr , train_label , True)
val_set = make_dataset(val_attr , val_label , True)
test_data = test_data.reshape([len(test_data) , 1, 27])

cnn = Train_nn(train_set, val_set, 
             model_path='cnn_model.pt',
             model='cnn',epoch=500)
cnn.train()

cnn_test = Test_nn(test_data, model_path='cnn_model.pt',
                   out_name='predict_cnn.csv' , model = 'cnn')
cnn_test.predict()

print('Finished CNN!')