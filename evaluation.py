from sklearn.metrics import confusion_matrix
from pathlib import Path
import csv
import numpy as np

ground_truth = []
with open('Test_label.csv' , 'r') as f:
  #f = csv.reader(f)
  for l in f:
    l = l.strip()
    ground_truth.append(l)
f.close()

ground_truth.pop(0)
ground_truth = np.array(ground_truth , dtype = np.int32)

log_predict = []
dnn_predict = []
cnn_predict = []

with open('predict_logit.csv' , 'r') as f1:
  with open('predict_dnn.csv' , 'r') as f2:
    with open('predict_cnn.csv' , 'r') as f3:
      
      for l in f1:
        l = l.strip()
        log_predict.append(l)
      for l2 in f2:
        l2 = l2.strip()
        dnn_predict.append(l2)
      for l3 in f3:
        l3 = l3.strip()
        cnn_predict.append(l3)

f1.close()
f2.close()
f3.close()


log_predict.pop(0)
dnn_predict.pop(0)
cnn_predict.pop(0)

log_predict = np.array(log_predict , dtype=np.int32)
dnn_predict = np.array(dnn_predict , dtype=np.int32)
cnn_predict = np.array(cnn_predict , dtype=np.int32)

log_conf = confusion_matrix(ground_truth, log_predict)
print('Logistics regression confusion matrix:')
print(log_conf)
print('Ratin expression: ')
print(np.round(log_conf[:]/np.sum(log_conf) , 3))
total_acc = 0
for i in range(len(log_conf)):
  total_acc += log_conf[i][i]
total_acc /= np.sum(log_conf)   
print("Total presicion: {:3.3f}".format(total_acc))

print("***Precision respectively: ***")
print('A: {:3.3f}'.format(log_conf[0][0]/np.sum(log_conf[0])))
print('B: {:3.3f}'.format(log_conf[1][1]/np.sum(log_conf[1])))
print('C: {:3.3f}'.format(log_conf[2][2]/np.sum(log_conf[2])))
print('D: {:3.3f}'.format(log_conf[3][3]/np.sum(log_conf[3])))


print('***Recall respectively: ***')
print('A: {:3.3f}'.format(log_conf[0][0]/np.sum(log_conf , 0)[0]))
print('B: {:3.3f}'.format(log_conf[1][1]/np.sum(log_conf , 0)[1]))
print('C: {:3.3f}'.format(log_conf[2][2]/np.sum(log_conf , 0)[2]))
print('D: {:3.3f}'.format(log_conf[3][3]/np.sum(log_conf , 0)[3]))


print('*** F1-score respectively: ***')
f1_list = ['A' , 'B' , 'C' , 'D']
for i in range(len(log_conf)):
  p = log_conf[i][i]/np.sum(log_conf[i])
  r = log_conf[i][i]/np.sum(log_conf , 0)[i]
  f = 2 * p * r / (p + r)
  print(f1_list[i] + ': ' + '{:3.3f}'.format(f))     

dnn_conf = confusion_matrix(ground_truth, dnn_predict)
print('DNN confusion matrix:')
print(dnn_conf)
print('Ratin expression: ')
print(np.round(dnn_conf[:]/np.sum(dnn_conf) , 3))
total_acc = 0
for i in range(len(dnn_conf)):
  total_acc += dnn_conf[i][i]
total_acc /= np.sum(dnn_conf)   
print("Total presicion: {:3.3f}".format(total_acc))

print("***Precision respectively: ***")
print('A: {:3.3f}'.format(dnn_conf[0][0]/np.sum(dnn_conf[0])))
print('B: {:3.3f}'.format(dnn_conf[1][1]/np.sum(dnn_conf[1])))
print('C: {:3.3f}'.format(dnn_conf[2][2]/np.sum(dnn_conf[2])))
print('D: {:3.3f}'.format(dnn_conf[3][3]/np.sum(dnn_conf[3])))

print('***Recall respectively: ***')
print('A: {:3.3f}'.format(dnn_conf[0][0]/np.sum(dnn_conf , 0)[0]))
print('B: {:3.3f}'.format(dnn_conf[1][1]/np.sum(dnn_conf , 0)[1]))
print('C: {:3.3f}'.format(dnn_conf[2][2]/np.sum(dnn_conf , 0)[2]))
print('D: {:3.3f}'.format(dnn_conf[3][3]/np.sum(dnn_conf , 0)[3]))

print('*** F1-score respectively: ***')
f1_list = ['A' , 'B' , 'C' , 'D']
for i in range(len(log_conf)):
  p = dnn_conf[i][i]/np.sum(dnn_conf[i])
  r = dnn_conf[i][i]/np.sum(dnn_conf , 0)[i]
  f = 2 * p * r / (p + r)
  print(f1_list[i] + ': ' + '{:3.3f}'.format(f))  


cnn_conf = confusion_matrix(ground_truth, cnn_predict)
print('CNN confusion matrix:')
print(cnn_conf)
print('Ratin expression: ')
print(np.round(cnn_conf[:]/np.sum(cnn_conf) , 3))
total_acc = 0
for i in range(len(cnn_conf)):
  total_acc += cnn_conf[i][i]
total_acc /= np.sum(cnn_conf)   
print("Total presicion: {:3.3f}".format(total_acc))
print("***Precision respectively: ***")
print('A: {:3.3f}'.format(cnn_conf[0][0]/np.sum(cnn_conf[0])))
print('B: {:3.3f}'.format(cnn_conf[1][1]/np.sum(cnn_conf[1])))
print('C: {:3.3f}'.format(cnn_conf[2][2]/np.sum(cnn_conf[2])))
print('D: {:3.3f}'.format(cnn_conf[3][3]/np.sum(cnn_conf[3])))

print('***Recall respectively: ***')
print('A: {:3.3f}'.format(cnn_conf[0][0]/np.sum(cnn_conf , 0)[0]))
print('B: {:3.3f}'.format(cnn_conf[1][1]/np.sum(cnn_conf , 0)[1]))
print('C: {:3.3f}'.format(cnn_conf[2][2]/np.sum(cnn_conf , 0)[2]))
print('D: {:3.3f}'.format(cnn_conf[3][3]/np.sum(cnn_conf , 0)[3]))

print('*** F1-score respectively: ***')
f1_list = ['A' , 'B' , 'C' , 'D']
for i in range(len(log_conf)):
  p = cnn_conf[i][i]/np.sum(cnn_conf[i])
  r = cnn_conf[i][i]/np.sum(cnn_conf , 0)[i]
  f = 2 * p * r / (p + r)
  print(f1_list[i] + ': ' + '{:3.3f}'.format(f))  




        