import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay, average_precision_score, precision_recall_curve, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import pandas as pd
import numpy as np


class Plotter:

    def __init__(self, model_id, classes, data, language):
        self.language = language
        self.model_id = model_id
        self.path = f'Results/img/{self.language}/{self.model_id}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.classes = classes
        self.data = data
        self.run()

    def run(self):
        self.heatmaps(mode='train', data=self.data['confusion_train'], true=self.data['Y_train'], pred=self.data['train_pred'])
        self.heatmaps(mode='val', data=self.data['confusion_val'], true=self.data['Y_train'], pred=self.data['train_pred'])
        self.tsne_visualization(mode='train', x=self.data['X_train'],
                                y=self.data['Y_train'], pred=self.data['train_pred'])
        self.tsne_visualization(mode='val', x=self.data['X_val'],
                                y=self.data['Y_val'], pred=self.data['val_pred'])
        self.roc_auc_curves(mode='train', y=self.data['Y_train'], proba=self.data['train_proba'])
        self.roc_auc_curves(mode='val', y=self.data['Y_val'], proba=self.data['val_proba'])
        self.precision_recall_curve(mode='train', y=self.data['Y_train'], proba=self.data['train_proba'])
        self.precision_recall_curve(mode='val', y=self.data['Y_val'], proba=self.data['val_proba'])

    def heatmaps(self, mode, data, true, pred):
        if len(self.classes) > 10:
            plt.figure(figsize=(13, 10))
            plt.tight_layout()
        else:
            plt.figure(figsize=(10, 8))
        # f1_scores = [f1_score(true, pred, labels=[i], average='micro') for i in range(data.shape[0])]
        # for i in range(data.shape[0]):
        #     data[i, i] = f1_scores[i]

        mapper = {'negative_anticipation_including_pessimism': 'pessimism',
                  'positive_anticipation_including_optimism': 'optimism'}
        classes = [mapper[item] if item in mapper else item for item in self.classes]
        heatmap = sns.heatmap(data, xticklabels=classes,
                              yticklabels=classes, annot=True)
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.title(f'{self.model_id}: Confusion Matrix {mode}')
        heatmap.get_figure().savefig(f'{self.path}/{self.model_id}_{mode}_hm.png')
        plt.clf()

    def tsne_visualization(self, mode, x, y, pred):
        tsne = TSNE(n_components=2).fit_transform(x)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        if isinstance(y, np.ndarray):
            Y_numeric = pd.Series(y).astype('category').cat.codes
            colormap_true = dict(zip(sorted(pd.Series(Y_numeric).unique()), sorted(pd.Series(y).unique())))
        else:
            Y_numeric = y.astype('category').cat.codes
            colormap_true = dict(zip(sorted(Y_numeric.unique()), sorted(y.unique())))
        scatter_true = axes[0].scatter(tsne[:, 0], tsne[:, 1], c=Y_numeric)
        axes[0].set_title(f't-SNE {mode} True')
        pred_numeric = pd.Series(pred).astype('category').cat.codes
        scatter_pred = axes[1].scatter(tsne[:, 0], tsne[:, 1], c=pred_numeric)
        axes[1].set_title(f't-SNE {mode} Pred')
        colormap_pred = dict(zip(sorted(pd.Series(pred_numeric).unique()),
                                 sorted(pd.Series(pred).unique())))
        fig.colorbar(scatter_true, ax=axes[0], label='Emotion')
        fig.colorbar(scatter_pred, ax=axes[1], label='Emotion')
        if colormap_pred == colormap_true:
            text = f'mapping: {colormap_true}'
        else:
            text = f'true mapping: {colormap_true}, pred mapping: {colormap_pred}'
        fig.text(0.5, 0.02, text, ha='center', va='bottom', wrap=True, fontsize=12)
        fig.suptitle('t-SNE Visualizations for TRAIN')
        plt.savefig(f'{self.path}/{self.model_id}_{mode}_tsne.png')
        plt.clf()

    def roc_auc_curves(self, mode, y, proba):
        # unique = y_train.unique()
        Y_bin = label_binarize(y, classes=self.classes)
        n_classes = Y_bin.shape[1]
        emotion_dict = dict(zip(range(n_classes), self.classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw = 2
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # micro average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc["micro"] = auc(all_fpr, mean_tpr)

        colors = cycle(plt.cm.tab10.colors[:n_classes])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(emotion_dict[i], roc_auc[i]))
        plt.plot(all_fpr, mean_tpr, color='deeppink', linestyle='--', lw=2,
                 label='Micro-average ROC curve (AUC = {0:0.2f})'.format(roc_auc["micro"]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{mode} ROC for {self.model_id}')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.path}/{self.model_id}_roc_auc_{mode}.png')
        plt.clf()

    def precision_recall_curve(self, mode, y, proba):
        mapper = {'negative_anticipation_including_pessimism': 'pessimism',
                  'positive_anticipation_including_optimism': 'optimism'}
        classes = [mapper[item] if item in mapper else item for item in self.classes]
        Y_bin = label_binarize(y, classes=classes)
        n_classes = Y_bin.shape[1]
        emotion_dict = dict(zip(range(n_classes), classes))
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_bin[:, i], proba[:, i])
            average_precision[i] = average_precision_score(Y_bin[:, i], proba[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            Y_bin.ravel(), proba.ravel())
        average_precision["micro"] = average_precision_score(Y_bin, proba, average="micro")
        colors = cycle(plt.cm.tab10.colors[:n_classes])
        _, ax = plt.subplots(figsize=(10, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)

        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"])
        display.plot(ax=ax, label="Micro-average precision-recall", color='deeppink', linestyle='--', )

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i])
            display.plot(ax=ax, label=f"Precision-recall for class {emotion_dict[i]}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title(f"{mode} Precision-Recall curve for {self.model_id}")
        plt.savefig(f'{self.path}/{self.model_id}_pre_rec_{mode}.png')
        plt.clf()
