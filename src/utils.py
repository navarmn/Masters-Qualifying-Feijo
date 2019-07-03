import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from os.path import join


def percentage_confusion_matrix(confMat):
    return np.around((confMat / np.sum(confMat,axis=1)[:,None])*100,2)


def print_metrics(y_true, y_hat):
   conf_mat = confusion_matrix(y_true, y_hat)
   acc = accuracy_score(y_true, y_hat)

   print('Confusion Matrix:')
   print(percentage_confusion_matrix(conf_mat))
   print('Accuracy: {}'.format(acc))

   pass 
   


def save_dict(d, folder):
   for feat in d.keys():
      for clf in d[feat].keys():
         df = pd.DataFrame(d[feat][clf]['conf'])
         df.to_csv(join(folder, 'conf_' + feat + '_' + clf + '.csv'))
         
