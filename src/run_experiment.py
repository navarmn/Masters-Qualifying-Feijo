import pandas as pd
import os
import sys
import time
import logging
filename_log = '.experiment.log'
if os.path.exists(os.path.join(os.path.dirname(__file__), filename_log)):
    os.remove(os.path.join(os.path.dirname(__file__), filename_log))
logging.basicConfig(level=logging.DEBUG, filename=filename_log)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse

from utils import *
from sklearn.model_selection import StratifiedKFold
import transforms
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import *



def run(args):
    for feature_set in DATA.keys():
        logging.debug('************************************************')
        logging.debug('Feature set: {}'.format(feature_set))
        data = pd.read_csv(os.path.join(DATA_FOLDER, DATA[feature_set]) + '.csv')
            
        pipeline = joblib.load(os.path.join(MODELS_FOLDER, PIPELINE_BASE_NAME + '_{}.pkl'.format(feature_set)))

        X = data
        y = transforms.GetLables().fit_transform(data)
        # logging.debug(features.head())
        
        skf = StratifiedKFold(n_splits=args['runs'], random_state=args['seed'], shuffle=True)
        # skf = StratifiedKFold(n_splits=args['runs'])
        
        for classifier in CLASSIFIERS:
            logging.debug('classifier: {}'.format(classifier))
            clf_name = CLASSIFIER_BASE_NAME + '_{}_{}.pkl'.format(feature_set, classifier)
            logging.debug('Loading classifier: {}'.format(clf_name))
            clf = joblib.load(os.path.join(MODELS_FOLDER, clf_name))
            logging.debug('Classifiers loaded')
            
            it = 0
            conf = np.zeros((7,7))
            acc = []
            f1 = []
            recall = []
            precision = []
            time_array = []
            time_array_feat = []

            for train_index, test_index in skf.split(X, y):
                it += 1
                # X_test = X[test_index]
                X_test = X.loc[test_index]
                logging.debug('Test data \n {}'.format(X_test))
                logging.debug('Transforming data...'.format(classifier))
                start_feat = time.time()
                X_test = pipeline.transform(X_test)
                end_feat = time.time()
                time_array_feat.append(end_feat - start_feat)
                y_test = y[test_index]

                logging.debug('======================')
                logging.debug('Monte Carlo run: {}'.format(it))

                logging.debug('Predicting with classifier: {}'.format(classifier))
                start = time.time()
                y_hat = clf.predict(X_test)
                end = time.time()
                time_array.append(end - start)

                logging.debug('Calculating metrics... {}'.format(classifier))
                print_metrics(y_test, y_hat)
                acc.append(accuracy_score(y_test, y_hat))
                f1.append(f1_score(y_test, y_hat, average='weighted'))
                precision.append(precision_score(y_test, y_hat, average='weighted'))
                recall.append(recall_score(y_test, y_hat, average='weighted'))
                # conf += percentage_confusion_matrix(confusion_matrix(y_test, y_hat))/2
                conf += confusion_matrix(y_test, y_hat)

                # RESULTS[feature_set]['acc'].append(acc)
                # RESULTS[feature_set]['f1'].append(f1)
                # RESULTS[feature_set]['precision'].append(precision)
                # RESULTS[feature_set]['recall'].append(recall)
                # RESULTS[feature_set]['conf'] = conf

            logging.debug('{} results:'.format(classifier))
            logging.debug('ACC \n {}'.format(acc))
            logging.debug('Confusion matrix \n {}'.format(percentage_confusion_matrix(conf/MT_RUNS)))

            RESULTS[feature_set][classifier]['acc'] = 100*np.mean(acc)
            RESULTS[feature_set][classifier]['f1'] = 100*np.mean(f1)
            RESULTS[feature_set][classifier]['precision'] = 100*np.mean(precision)
            RESULTS[feature_set][classifier]['recall'] = 100*np.mean(recall)
            RESULTS[feature_set][classifier]['test_time_clf'] = 1000*np.mean(time_array)
            RESULTS[feature_set][classifier]['test_time_feat'] = 1000*np.mean(time_array_feat)
            

            RESULTS_STD[feature_set][classifier]['acc'] = 100*np.std(acc)
            RESULTS_STD[feature_set][classifier]['f1'] = 100*np.std(f1)
            RESULTS_STD[feature_set][classifier]['precision'] = 100*np.std(precision)
            RESULTS_STD[feature_set][classifier]['recall'] = 100*np.std(recall)
            RESULTS_STD[feature_set][classifier]['test_time_clf'] = 1000*np.std(time_array)
            RESULTS_STD[feature_set][classifier]['test_time_feat'] = 1000*np.std(time_array_feat)

            RESULTS_conf[feature_set][classifier]['conf'] = percentage_confusion_matrix(conf/MT_RUNS)



    # Save to dataframe

    df_results = pd.DataFrame(
        index=INDEX_DF,
        columns=['acc', 'f1', 'precision', 'recall', 'test_time_clf', 'test_time_feat'],
        data = DATA_DF,
    )


    df_results_std = pd.DataFrame(
        index=INDEX_DF,
        columns=['acc', 'f1', 'precision', 'recall', 'test_time_clf', 'test_time_feat'],
        data = DATA_DF_STD,
    )

    print(df_results.round(args['round']))


    df_results.round(args['round']).to_csv(os.path.join(args['output'], 'results_acc.csv'))
    df_results_std.round(args['round']).to_csv(os.path.join(args['output'], 'results_std.csv'))

    save_dict(RESULTS_conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=MT_RUNS, type=int)
    parser.add_argument('--round', default=ROUND, type=int)
    parser.add_argument('--output', default=RESULTS_FOLDER, type=str)
    parser.add_argument('--seed', default=SEED, type=str)
    args = vars(parser.parse_args())    
    #
    if not os.path.exists(args['output']):
        os.system('mkdir {}'.format(args['output']))
    #
    run(args)