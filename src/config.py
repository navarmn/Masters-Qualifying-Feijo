RESULTS = {
    'FOURIER': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
    'HOS': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
    'SCM': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
}

RESULTS_STD = {
    'FOURIER': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
    'HOS': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
    'SCM': {
        'mlp':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'svm':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'knn':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        },
        'naive_bayes':{
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'test_time_clf': [],
            'test_time_feat': [],
        }
    },
}

RESULTS_conf = {
    'FOURIER': {
        'mlp':{
            'conf': [],
        },
        'svm':{
            'conf': [],
        },
        'knn':{
            'conf': [],
        },
        'naive_bayes':{
            'conf': [],
        },
    },
    'HOS': {
        'mlp':{
            'conf': [],
        },
        'svm':{
            'conf': [],
        },
        'knn':{
            'conf': [],
        },
        'naive_bayes':{
            'conf': [],
        },
    },
    'SCM': {
        'mlp':{
            'conf': [],
        },
        'svm':{
            'conf': [],
        },
        'knn':{
            'conf': [],
        },
        'naive_bayes':{
            'conf': [],
        }
    }
}


INDEX_DF = [
    'FOURIER_mlp', 'HOS_mlp', 'SCM_mlp',
    'FOURIER_svm', 'HOS_svm', 'SCM_svm',
    'FOURIER_knn', 'HOS_knn', 'SCM_knn',
    'FOURIER_naive_bayes', 'HOS_naive_bayes', 'SCM_naive_bayes',    
    ]


DATA_DF = [
    RESULTS['FOURIER']['mlp'], RESULTS['HOS']['mlp'], RESULTS['SCM']['mlp'],
    RESULTS['FOURIER']['svm'], RESULTS['HOS']['svm'], RESULTS['SCM']['svm'],
    RESULTS['FOURIER']['knn'], RESULTS['HOS']['knn'], RESULTS['SCM']['knn'],
    RESULTS['FOURIER']['naive_bayes'], RESULTS['HOS']['naive_bayes'], RESULTS['SCM']['naive_bayes'],
    ]

DATA_DF_STD = [
    RESULTS_STD['FOURIER']['mlp'], RESULTS_STD['HOS']['mlp'], RESULTS_STD['SCM']['mlp'],
    RESULTS_STD['FOURIER']['svm'], RESULTS_STD['HOS']['svm'], RESULTS_STD['SCM']['svm'],
    RESULTS_STD['FOURIER']['knn'], RESULTS_STD['HOS']['knn'], RESULTS_STD['SCM']['knn'],
    RESULTS_STD['FOURIER']['naive_bayes'], RESULTS_STD['HOS']['naive_bayes'], RESULTS_STD['SCM']['naive_bayes'],
    ]

