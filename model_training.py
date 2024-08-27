from _utils.model_training_helper import TrainerUtils
from evaluation import Evaluator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import pandas as pd
import datetime
import time
import re

EX_GERSTI = ['emotion', 'id', 'source']
EX_SEMEVAL = ['engl_text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'emotion']
EX_ISEAR = ['Prior_Emotion', 'fear', 'disgust', 'joy', 'shame', 'guilt', 'sadness', 'anger', 'emotion']
EX_GNE = ['engl_text', 'emotion', 'other_emotions', 'reader_emotions']

class Trainer:
    """
    Core class for model training
    :param language: en or de, depending on language to use for training
    :param dataset: if true, uses locally cached dataset
    :param dataset: dataset_id to load respective dataset
    :param resample: if true, resample labels for equal class distribution
    :param base_emo: if true, filters dataset to 4 base emotions + no_emotion
    :param label_merge: if true, reduces label number based on EDA results,
     only works on base_emo:False & dataset:GNE_SEM_GER
    :param classifier: chosen model (svm, rf & gradient boosting)
    :param log: if true, logs and plots results
    :param tuning: if true, hyperparameter tuning is performed
    :param feature_set: definition of feature set to use
    :param configs: configuration for NeuralNetwork, otherwise ignored
    :return:
    """
    def __init__(self, language, use_cache, dataset, feature_set, hypertuning=False, resample=False, base_emo=True,
                 label_merge=False, classifier='svm', log=False, tuning=False, configs=None, nn_sweep=False,
                 param_grid=None):
        self.language = language
        self.use_cache = use_cache
        self.dataset = dataset
        self.resample = resample
        self.base_emo = base_emo
        self.label_merge = label_merge
        self.classifier = classifier
        self.log = log
        self.tuning = tuning
        self.configs = configs
        self.feature_set = feature_set
        self.hypertuning = hypertuning
        self.nn_sweep = nn_sweep
        self.param_grid = param_grid
        self.model_id = None
        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.features = None
        self.Utils = TrainerUtils(self)
        self.run()

    def run(self):
        self.load_data()
        if self.hypertuning:
            self.hyperparam_tuning()
        else:
            self.single_execution()

    def load_data(self):
        if self.label_merge:
            self.base_emo = False
            self.dataset = 'GNE_SEM_GER'
        if self.use_cache:
            self.X_train, self.X_val, self.Y_train, self.Y_val, self.features = self.Utils.load_cached_dataset()
        else:
            self.X_train, self.X_val, self.Y_train, self.Y_val, self.features = self.Utils.data_prep()
        model_id = f'{datetime.datetime.now().strftime("%y%m%d%H%M%S")}_{self.classifier}_{self.features}'
        self.model_id = re.sub(r'\s+', '_', model_id)

    def single_execution(self):
        train_pred, Y_pred, train_proba, val_proba, classes, model = self.Utils.trigger_training()
        if self.log:
            train = pd.concat([self.Y_train.reset_index(drop=True), pd.Series(train_pred, name='prediction'),
                               pd.DataFrame(train_proba)], axis=1)
            val = pd.concat([self.Y_val.reset_index(drop=True), pd.Series(Y_pred, name='prediction'),
                             pd.DataFrame(val_proba)], axis=1)
            train.to_csv(f'data/predictions/{self.model_id}_train.csv')
            val.to_csv(f'data/predictions/{self.model_id}_test.csv') #changed for test
            Evaluator(self.dataset, self.log, self.classifier, self.X_train, self.X_val, self.Y_train, train_pred,
                      self.Y_val, Y_pred, self.features, self.tuning, train_proba, val_proba, classes, self.model_id,
                      self.base_emo, self.label_merge, self.language)


    def hyperparam_tuning(self):
        model = self.Utils.trigger_training()
        print('Starting grid_search ...')
        start = time.time()
        scoring_metrics = {'accuracy': make_scorer(accuracy_score), 'f1_score': make_scorer(f1_score, average='weighted')}
        grid_search = GridSearchCV(model, self.param_grid, scoring=scoring_metrics, refit='f1_score',
                                   return_train_score=True, n_jobs=-1)
        grid_search.fit(self.X_train, self.Y_train)
        print(f'Finished grid_search in {time.time() - start} seconds.')
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Accuracy:", grid_search.best_score_)
        val_accuracy = accuracy_score(self.Y_val, grid_search.predict(self.X_val))
        print("Val Set Accuracy:", val_accuracy)
        val_f1_score = grid_search.score(self.X_val, self.Y_val)
        print("Validation Set F1 Score:", val_f1_score)
        if self.log:
            self.Utils.log_dataframe_to_html(pd.DataFrame(grid_search.cv_results_),
                                  f'Results/experiments/grid_search_{self.language}.html')


if __name__ == "__main__":
    Trainer(language='en',
            use_cache=True,
            dataset='GNE_SEM_GER',
            classifier='svm',
            base_emo=True,
            hypertuning=False,
            configs=None,
            log=True,
            feature_set={'ada': True,
                         'pca': False,
                         'n_pca': 300,
                         'tf_idf': False,
                         'linguistics': False,
                         'nrc': True,
                         'n_tfidf': 1000},
            param_grid={
                'estimator__C': [5, 10, 15], #0.01, 1, 20, 50
                'estimator__kernel': ['rbf']
            })
    # train(language='en',
    #       use_cache=True,
    #       dataset='GNE_SEM_GER',
    #       resample=False,
    #       base_emo=True,
    #       label_merge=False,
    #       classifier='nn',
    #       log=True,
    #       feature_set={'ada': True,
    #                    'pca': False,
    #                    'n_pca': 300,
    #                    'tf_idf': False,
    #                    'linguistics': False,
    #                    'nrc': True,
    #                    'n_tfidf': 1000},
    #       configs={'num_classes': 5,
    #                'batch_size': 64,
    #                'lr': 0.0001,
    #                'epochs': 50,
    #                'hl1_size': 1024,
    #                'hl2_size': 256,
    #                'hl3_size': 64,
    #                'use_third_hidden_layer': True,
    #                'optimizer': 'adam'})

    # hyperparam_tuning(dataset='GNE_SEM_GER',
    #       resample=False,
    #       base_emo=True,
    #       label_merge=False,
    #       classifier='svm',
    #       log=True,
    #       language='en',
    #       use_cache=True,
    #       feature_set={'ada': True,
    #                    'pca': False,
    #                    'n_pca': 300,
    #                    'tf_idf': False,
    #                    'linguistics': False,
    #                    'nrc': False,
    #                    'n_tfidf': 1000},
    #       param_grid={
    #                     'estimator__C': [0.01, 0.01, 0.1, 0.5, 1, 20, 50, 100],
    #                     'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #                     'estimator__gamma': ['scale', 0.01],
    #             })



# for svm:
# param_grid={
#               'estimator__C': [0.01,0.01, 0.1, 0.5, 1, 20, 50, 100],
#               'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#               'estimator__gamma': ['scale', 0.01],
#       }) # best result: C=20, gamma=scale, kernel =rbf -> f1:.63 -> result hyperparamter tuning: no observable increase

# for rf:
# param_grid = {
# 'estimator__n_estimators': [50, 100, 200, 300],
# 'estimator__max_depth': [5, 10, 15, 20],
# 'estimator__max_features': ['sqrt', 'log2']}