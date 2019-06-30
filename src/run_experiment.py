import pandas as pd
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import *
from sklearn.model_selection import StratifiedKFold
import transforms
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DATA_FOLDER = '../data/csv'
MODELS_FOLDER = '../models'
PIPELINE_BASE_NAME = 'transformer_pipeline'
CLASSIFIER_BASE_NAME = 'clf'

DATA = {
         'FOURIER': 'v000_SCIG_SC_SENSORA_FOURIER',
         'HOS': 'v000_SCIG_SC_SENSORA_HOS',
         'SCM': 'v000_SCIG_SC_SENSORA_SCM',
      }

LABELS = {'FOURIER': 'v000_SCIG_SC_SENSORC_FOURIER_010_labels_chunk_90'}

SEED = 1879287912
MT_RUNS = 2

CLASSIFIERS = {'mlp', 'svm', 'knn','naive_bayes'}

from config import *


conf = np.zeros((7,7))

for feature_set in DATA.keys():
    logging.info('Feature set: {}'.format(feature_set))
    data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[feature_set]) + '.csv')
        
    pipeline = joblib.load(os.path.join(MODELS_FOLDER, PIPELINE_BASE_NAME + '_{}.pkl'.format(feature_set)))

    X = data

    y = transforms.GetLables().fit_transform(data)
    # logging.info(features.head())
    
    skf = StratifiedKFold(n_splits=MT_RUNS, random_state=SEED)


    for classifier in CLASSIFIERS:
        logging.info('classifier: {}'.format(classifier))
        clf_name = CLASSIFIER_BASE_NAME + '_{}_{}.pkl'.format(feature_set, classifier)
        logging.info('Loading classifier: {}'.format(clf_name))
        clf = joblib.load(os.path.join(MODELS_FOLDER, clf_name))
        logging.info('Classifiers loaded')
        
        it = 0
        acc = []
        f1 = []
        recall = []
        precision = []

        for train_index, test_index in skf.split(X, y):
            it += 1
            # X_test = X[test_index]
            X_test = X.loc[test_index]
            X_test = pipeline.transform(X_test)
            y_test = y[test_index]

            logging.info('======================')
            logging.info('Monte Carlo run: {}'.format(it))

            logging.info('Transforming data...'.format(classifier))
            logging.info('Predicting with classifier: {}'.format(classifier))
            y_hat = clf.predict(X_test)

            logging.info('Calculating metrics... {}'.format(classifier))
            print_metrics(y_test, y_hat)
            acc.append(accuracy_score(y_test, y_hat))
            f1.append(f1_score(y_test, y_hat, average='weighted'))
            precision.append(precision_score(y_test, y_hat, average='weighted'))
            recall.append(recall_score(y_test, y_hat, average='weighted'))
            conf += percentage_confusion_matrix(confusion_matrix(y_test, y_hat))/2

            # RESULTS[feature_set]['acc'].append(acc)
            # RESULTS[feature_set]['f1'].append(f1)
            # RESULTS[feature_set]['precision'].append(precision)
            # RESULTS[feature_set]['recall'].append(recall)
            # RESULTS[feature_set]['conf'] = conf

        RESULTS[feature_set][classifier]['acc'] = np.mean(100*acc)
        RESULTS[feature_set][classifier]['f1'] = np.mean(100*f1)
        RESULTS[feature_set][classifier]['precision'] = np.mean(100*precision)
        RESULTS[feature_set][classifier]['recall'] = np.mean(100*recall)

        RESULTS_std[feature_set]['acc'] = np.std(100*acc)
        RESULTS_std[feature_set]['f1'] = np.std(100*f1)
        RESULTS_std[feature_set]['precision'] = np.std(100*precision)
        RESULTS_std[feature_set]['recall'] = np.std(100*recall)

        RESULTS_conf[feature_set]['conf'] = conf


# Save to dataframe

df_results = pd.DataFrame(
    index=['FOURIER', 'HOS', 'SCM'],
    columns=['acc', 'f1', 'precision', 'recall'],
    data = [
        RESULTS['FOURIER']['mlp'], 
        RESULTS['HOS']['mlp'], 
        RESULTS['SCM']['mlp']
        ],
)

print(df_results)
