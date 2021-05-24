import os
from pathlib import Path
import numpy as np
import csv

class Open_data:
  
  def __init__(self,file_name , data_type = 'train'):
    self.data = []
    self.label_dict = {'A' : 0 , 'B' : 1 , 'C' : 2 , 'D' : 3}
    self.id_class = []
    self.read_data(file_name)
    self.columns = self.data[0]
    self.data.pop(0)
    self.extract_class(-1)
    #self.label_to_one_hot()
    self.binarize()
    self.to_one_hot(4)
    self.to_one_hot(6)
    self.to_one_hot(8)
    self.clean_useless_col([4,6,8])
    self.complete_missing_value(5 , specific_value=1.0 ,use_value=True)
    self.complete_missing_value(4)
  
  def read_data(self , name):
    cur_dir = str(Path.cwd())
    file_path =  cur_dir + '/' + name
    with open(file_path, 'r') as out_file:
      file = csv.reader(out_file)
      for line in file:
        line.pop(0)  
        self.data.append(line)  
    out_file.close()
    
  def binarize(self):
    for l in self.data:
      l[0] = 1 if l[0] == 'Male' else 0
      l[1] = 1 if l[1] == 'Yes' else 0
      l[3] = 1 if l[3] == 'Yes' else 0
  
  def to_one_hot(self, col):
    col_item = {}
    append_counter = 0
    for l in self.data:
      if l[col] not in col_item:
        col_item[l[col]] = append_counter
        append_counter += 1
    
    append_list = [0 for i in range(len(col_item))]
    for i in range(len(self.data)):
      real_append = append_list.copy()
      real_append[col_item[self.data[i][col]]] = 1
      self.data[i] = self.data[i] + real_append

    self.columns += list(col_item.keys())

  def label_to_one_hot(self):
    one_hot = [0 for i in range(len(self.label_dict))]
    for i in range(len(self.id_class)):
      intend_one_hot = one_hot.copy()
      intend_one_hot[self.id_class[i] - 1] = 1
      self.id_class[i] = intend_one_hot

  def clean_useless_col(self , del_col):
    for l in self.data:
      del_counter = 0
      for col in del_col:
        l.pop(col - del_counter)
        del_counter += 1
    
    del_col_counter = 0
    for col in del_col:
      self.columns.pop(col - del_col_counter)
      del_col_counter += 1

  def complete_missing_value(self , col , specific_value=0.0, use_value=False):
     complete_value = specific_value if use_value else None    
     if complete_value == None:
        col_sum = 0
        valid_counter = 0
        for l in self.data:
          if l[4] != '':
            col_sum += float(l[4])
            valid_counter += 1
        
        mean = round(col_sum / valid_counter , 1)
        complete_value = mean
            
     for l in self.data:
       if l[col] == '':
         l[col] = complete_value
  
  def extract_class(self , col):
    for l in self.data:
      self.id_class.append(self.label_dict[l[col]])
      l.pop(-1)
    self.columns.pop(-1)  
        
  def show(self , row):
    for i in range(row):
      print(self.data[i])    
  
  def __getitem__(self , key):
    return self.data[key] , self.id_class[key]    
  
  def create_csv(self , write_name , file_type='attr'):
    with open(str(Path.cwd()) + '/' + write_name , 'w') as in_file:
      title_str = 'class'
      if file_type == 'attr':
        title_str = ''  
        for col_name in self.columns:
          title_str += col_name + ','
        title_str = title_str[:-1]
      
      if file_type == 'attr':
        data_file = self.data
      elif file_type == 'label':
        data_file = self.id_class  
            
      print(title_str  , file=in_file)
      for row in data_file:
        append_row = str(row).replace("'","")
        append_row = append_row[1:-1].replace(' ','')
        
        print(append_row if file_type == 'attr' else row , file=in_file)
        



train_set = Open_data('Train.csv')
train_set.create_csv('Train_data.csv' , 'attr')
train_set.create_csv('Train_label.csv' , 'label')

test_set = Open_data('Test.csv')
test_set.create_csv('Test_data.csv' , 'attr')
test_set.create_csv('Test_label.csv' , 'label')

#print(len(test.columns) , len(test.data[0]))
#print(test.id_class[:10])
#test.show(10)
#print(test[:10])

    