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

DATA_FOLDER = '../data/csv'
MODELS_FOLDER = '../models'
PIPELINE_BASE_NAME = 'transformer_pipeline'
CLASSIFIER_BASE_NAME = 'clf'

DATA = 
      {
         'FOURIER': 'v000_SCIG_SC_SENSORC_FOURIER',
         'HOS': 'v000_SCIG_SC_SENSORC_HOS',
         'SCM': 'v000_SCIG_SC_SENSORC_SCM',
      }

LABELS = {'FOURIER': 'v000_SCIG_SC_SENSORC_FOURIER_010_labels_chunk_90'}

SEED = 1879287912
MT_RUNS = 2

CLASSIFIERS = {'mlp', 'svm', 'knn','naive_bayes'}


for feature_set in DATA.keys():
    logging.info('Feature set: {}'.format(feature_set))
    data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[feature_set]) + '.csv')
        
    pipeline = joblib.load(os.path.join(MODELS_FOLDER, PIPELINE_BASE_NAME + '_{}.pkl'.format(feature_set)))

    features = pipeline.transform(data)
    # logging.info(features.head())

    X_test = features.values
    y_test = labels.values.reshape(-1)

    for classifier in CLASSIFIERS:
        logging.info('classifier: {}'.format(classifier))
        clf_name = CLASSIFIER_BASE_NAME + '_{}_{}.pkl'.format(feature_set, classifier)
        logging.info('Loading classifier: {}'.format(clf_name))
        clf = joblib.load(os.path.join(MODELS_FOLDER, clf_name))
        logging.info('Classifiers loaded')

        logging.info('Predicting with classifier: {}'.format(classifier))
        y_hat = clf.predict(X_test)

        logging.info('Calculating metrics... {}'.format(classifier))
        print_metrics(y_test, y_hat)

