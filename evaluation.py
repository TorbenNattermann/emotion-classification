import datetime
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from _utils.evaluation_helper import Plotter


class Evaluator:
    """
    Class performs evaluation of trained model and logs results
    """

    def __init__(self, dataset, log, classifier, X_train, X_val, Y_train, train_pred, Y_val, Y_pred, features, tuning,
                 train_proba, val_proba, classes, model_id, base_emo, label_merge, language):
        self.dataset = dataset
        self.log = log
        self.classifier = classifier
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.train_pred = train_pred
        self.Y_val = Y_val
        self.Y_pred = Y_pred
        self.features = features
        self.tuning = tuning
        self.train_proba = train_proba
        self.val_proba = val_proba
        self.classes = classes
        self.base_emo = base_emo
        self.label_merge = label_merge
        self.language = language
        if self.base_emo:
            self.model_id = f'{model_id}_base'
        else:
            self.model_id = f'{model_id}_full'
            if self.label_merge:
                self.model_id = f'{model_id}_merge'
        self.run()

    def run(self):
        print(f'-----\nEvaluation Results for model {self.model_id}')
        if self.Y_val.shape[1] == 1 or self.Y_train.shape[1] == 1:
            Y_val_np = np.array(self.Y_val).reshape(-1,)
            Y_train_np = np.array(self.Y_train).reshape(-1,)
        acc_train = accuracy_score(Y_train_np, self.train_pred)
        f1_train = f1_score(Y_train_np, self.train_pred, average='weighted')
        precision_train = precision_score(Y_train_np, self.train_pred, average='weighted')
        recall_train = recall_score(Y_train_np, self.train_pred, average='weighted')
        roc_auc_train = roc_auc_score(Y_train_np, self.train_proba, average='weighted', multi_class='ovr')
        confusion_train = confusion_matrix(Y_train_np, self.train_pred)

        accuracy = accuracy_score(Y_val_np, self.Y_pred)
        f1 = f1_score(Y_val_np, self.Y_pred, average='weighted')
        _precision = precision_score(Y_val_np, self.Y_pred, average='weighted')
        _recall = recall_score(Y_val_np, self.Y_pred, average='weighted')
        roc_auc_val = roc_auc_score(Y_val_np, self.val_proba, average='weighted', multi_class='ovr')
        confusion = confusion_matrix(Y_val_np, self.Y_pred)
        agg_train = pd.concat([pd.DataFrame(self.Y_train).reset_index(), pd.DataFrame(self.train_proba)], axis=1)
        agg_val = pd.concat([pd.DataFrame(self.Y_val).reset_index(), pd.DataFrame(self.Y_pred)], axis=1)
        train_local = {}

        for index, emotion in enumerate(self.classes):
            true = np.where(agg_train['emotion'] == emotion, 1, 0)
            pred = np.where(pd.DataFrame(self.train_pred)[0] == emotion, emotion, 'other')
            pred_proba = self.train_proba[:, index]
            _string = self.compute_class_scores(true, pred, pred_proba)
            print(f'Scores on train for {emotion}: ', _string)
            train_local[f'<strong>{emotion}</strong>'] = _string
        train_local = '<br>'.join([f'"{key}": "{value}"' for key, value in train_local.items()])
        val_local = {}
        for index, emotion in enumerate(self.classes):
            true = np.where(agg_val['emotion'] == emotion, 1, 0)
            pred = np.where(pd.DataFrame(self.Y_pred)[0] == emotion, emotion, 'other')
            pred_proba = self.val_proba[:, index]
            _string = self.compute_class_scores(true, pred, pred_proba)
            val_local[f'<strong>{emotion}</strong>'] = _string
            print(f'Scores on val for {emotion}: ', _string)
        val_local = '<br>'.join([f'"{key}": "{value}"' for key, value in val_local.items()])
        print(f'Train: Acc = {acc_train}, F1 = {f1_train}, Precision = {precision_train}, Recall = {recall_train}, '
              f'AUC = {roc_auc_train}')
        print(f'Val: Accuracy: {accuracy}, F1: {f1}, Precision = {_precision}, Recall = {_recall}, AUC = {roc_auc_val}')
        if self.log:
            if not self.tuning:
                data_dict = {'confusion_train': confusion_train,
                             'confusion_val': confusion,
                             'X_train': self.X_train,
                             'X_val': self.X_val,
                             'Y_train': self.Y_train if self.Y_train.shape[1] != 1 else Y_train_np,
                             'Y_val': self.Y_val if self.Y_val.shape[1] != 1 else Y_val_np,
                             'train_pred': self.train_pred,
                             'val_pred': self.Y_pred,
                             'train_proba': self.train_proba,
                             'val_proba': self.val_proba
                             }
                Plotter(model_id=self.model_id, classes=self.classes, data=data_dict, language=self.language)
                self.log_experiment_html(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         data=f'{self.dataset} dataset',
                                         samples=f'{len(self.X_train) + len(self.X_val)}',
                                         features=self.features,
                                         n_features=self.X_train.shape[1],
                                         model=self.classifier,
                                         label=f'{self.Y_train.emotion.unique()}',
                                         n_label=f'{self.Y_train.nunique()} classes',
                                         train_global=f'<strong>F1</strong>: {f1_train.round(4)};'
                                                      f'<br><strong style="color: #FF0000;">Accuracy</strong>: {acc_train.round(4)};<br>'
                                                      f'<strong style="color: #FF0000;">Precision</strong> = 'f'{precision_train.round(4)};<br>'
                                                      f'<strong>Recall</strong> = {recall_train.round(4)},'
                                                      f'<br><strong>AUC</strong>: 'f'{roc_auc_train.round(4)}',
                                         val_global=f'<strong>F1</strong>: {f1.round(4)};'
                                                    f'<br><strong style="color: #FF0000;">Accuracy</strong>: {accuracy.round(4)};'
                                                    f'<br><strong style="color: #FF0000;">Precision</strong> = 'f'{_precision.round(4)};'
                                                    f'<br><strong>Recall</strong> = {_recall.round(4)},'
                                                    f'<br><strong>AUC</strong>: {roc_auc_val.round(4)}',
                                         train_local=train_local,
                                         val_local=val_local,
                                         )
            else:
                self.log_hyperparam_tuning_html()

    def compute_class_scores(self, true, pred, pred_proba):
        roc = roc_auc_score(true, pred_proba).round(4)
        pred_binary = np.where(pred == 'other', 0, 1)
        f1 = f1_score(true, pred_binary).round(4)
        precision = precision_score(true, pred_binary).round(4)
        recall = recall_score(true, pred_binary).round(4)
        return (f'<span style="color: #006400;">ROC</span>: {roc}, <span style="color: #006400;">F1</span>: {f1}, '
                f'<span style="color: #FF0000;">Precision</span>: {precision}, <span style="color: '
                f'#006400;">Recall</span>: {recall}')

    def log_hyperparam_tuning_html(self):
        pass

    def log_experiment_html(self, date, data, samples, features, n_features, model, label, n_label, train_global,
                            val_global,
                            train_local, val_local):
        if isinstance(train_local, dict):
            train_local = json.dumps(train_local)
        if isinstance(val_local, dict):
            val_local = json.dumps(val_local)
        entry = f"""
        <tr>
            <td>{date}</td>
            <td>{data}</td>
            <td>{samples}</td>
            <td>{features}</td>
            <td>{n_features}</td>
            <td>{model}</td>
            <td>{label}</td>
            <td>{n_label}</td>
            <td>{train_global}</td>
            <td>{val_global}</td>
            <td>{train_local}</td>
            <td>{val_local}</td>
        </tr>
        """

        if self.base_emo:
            path = f'Results/experiments/logs_base_{self.language}.html'
        else:
            path = f'Results/experiments/logs_full_{self.language}.html'
            if self.label_merge:
                path = 'Results/experiments/combined_merge.html'
        with open(path, 'a') as html_file:
            html_file.write(entry)
