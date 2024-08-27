import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay, average_precision_score, precision_recall_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.patches as mpatches

def roc_auc_curves(mode, models):
    for model in models:
        Y_bin = label_binarize(model['y'], classes=sorted(model['y'].unique()))
        n_classes = Y_bin.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw = 2
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], model['proba'].iloc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # micro average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_mean = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2,
                 label=f"Micro-average ROC curve for {model['name']} (AUC = {roc_auc_mean:0.2f})")



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{mode} ROC for test')
    plt.legend(loc="lower right")
    plt.savefig(f'../Results/img/summary_plots/test2.png')
    plt.clf()


def precision_recall_curves(models, title):
    # Create a plot outside the loop
    fig, ax = plt.subplots(figsize=(10, 8))

    # Iterate over each model and plot the micro-averaged precision-recall curve
    for model in models:
        Y_bin = label_binarize(model['y'], classes=sorted(model['y'].unique()))
        n_classes = Y_bin.shape[1]

        precision = dict()
        recall = dict()

        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(Y_bin[:, i], model['proba'].iloc[:, i])
            precision[i] = prec
            recall[i] = rec

        # A "micro-average": quantifying score on all classes jointly
        y_true_flat = Y_bin.ravel()
        y_score_flat = model['proba'].values.ravel()
        prec_micro, rec_micro, _ = precision_recall_curve(y_true_flat, y_score_flat)

        # Plot the micro-averaged precision-recall curve for the current model
        display = PrecisionRecallDisplay(
            recall=rec_micro,
            precision=prec_micro,
            average_precision=average_precision_score(y_true_flat, y_score_flat, average="micro"),
            prevalence_pos_label=Counter(y_true_flat)[1] / len(y_true_flat),
        )

        display.plot(ax=ax, label=f'P-R curve for {model["name"]}') #
        lines = ax.lines

        # Set custom colors for the lines
        # for line, color in zip(lines, colormap):
        #     line.set_color(color)

    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # Set the title and legend
    _ = ax.set_title(f"Micro-averaged Precision-Recall Curve over {title}")
    _ = ax.legend()


    plt.savefig(f'../Results/img/summary_plots/{title}.png')
    plt.clf()

if __name__ == '__main__':
    m1 = pd.read_csv('../data/predictions/.csv')
    m2 = pd.read_csv('../data/predictions/.csv')
    m3 = pd.read_csv('../data/predictions/.csv')
    m4 = pd.read_csv('../data/predictions/.csv')
    m5 = pd.read_csv('../data/predictions/.csv')
    m6 = pd.read_csv('../data/predictions/.csv')
    #m7 = pd.read_csv('../data/predictions/240215173411_svm_ada_embeddings_val_m7.csv')
    m8 = pd.read_csv('../data/predictions/.csv')

    # m9 = pd.read_csv('../data/predictions/240215172756_svm_linguistics_val_m9.csv')
    # m10 = pd.read_csv('../data/predictions/240214143610_rf_linguistics_val_m10.csv')
    # m11 = pd.read_csv('../data/predictions/240215172554_svm_nrc_val_m11.csv')
    # m12 = pd.read_csv('../data/predictions/240214215525_rf_nrc_val_m12.csv')
    # m13 = pd.read_csv('../data/predictions/240215171340_svm_tfidf_1000_dims_val_m13.csv')
    # m14 = pd.read_csv('../data/predictions/240214220757_rf_tfidf_1000_dims_val_m14.csv')
    # m15 = pd.read_csv('../data/predictions/240215170058_svm_ada_embeddings_val_m15.csv')
    # m16 = pd.read_csv('../data/predictions/240214222607_rf_ada_embeddings_val_m16.csv')

    # m17 = pd.read_csv('../data/predictions/240215124831_svm_ada_embeddings&tfidf_1000_dims_val_m17.csv')
    # m18 = pd.read_csv('../data/predictions/240215121037_svm_ada_embeddings&nrc_val_m18.csv')
    # m19 = pd.read_csv('../data/predictions/240215133940_svm_ada_embeddings&tfidf_1000_dims&nrc_val_m19.csv')
    # m20 = pd.read_csv('../data/predictions/240215135036_svm_tfidf_1000_dims&nrc_val_m20.csv')

    # m21 = pd.read_csv('../data/predictions/240215135753_svm_ada_embeddings&tfidf_1000_dims_val_m21.csv')
    # m22 = pd.read_csv('../data/predictions/240215154041_svm_ada_embeddings&nrc_val_m22.csv')
    # m23 = pd.read_csv('../data/predictions/240215155717_svm_ada_embeddings&tfidf_1000_dims&nrc_val_m23.csv')
    # m24 = pd.read_csv('../data/predictions/240215164934_svm_tfidf_1000_dims&nrc_val_m24.csv')
    #
    mh2 = pd.read_csv('../data/predictions/240215194800_svm_ada_embeddings&nrc_val_hyperfull.csv')
    mh1 = pd.read_csv('../data/predictions/240215193558_svm_ada_embeddings&nrc_val_hyperbase.csv')
    m25 = pd.read_csv('../data/predictions/240215183758_nn_ada_embeddings&nrc_val_m25astral.csv')
    m26 = pd.read_csv('../data/predictions/240215185623_nn_ada_embeddings&nrc_val_m26_b.csv')

    models = list()
    models.append({'y': mh1.emotion, 'proba': mh1.iloc[:, 3:], 'name': 'SVM_ADA_NRC_BASE'})
    models.append({'y': mh2.emotion, 'proba': mh2.iloc[:, 3:], 'name': 'SVM_ADA_NRC_COMPLETE'})
    models.append({'y': m25.emotion, 'proba': m25.iloc[:, 3:], 'name': 'NN_ADA_NRC_BASE'})
    models.append({'y': m26.emotion, 'proba': m26.iloc[:, 3:], 'name': 'NN_ADA_NRC_COMPLETE'})
    # models.append({'y': m24.emotion, 'proba': m24.iloc[:, 3:], 'name': 'SVM_NRC_TFIDF'})
    # models.append({'y': m14.emotion, 'proba': m14.iloc[:, 3:], 'name': 'RF_TFIDF'})
    # models.append({'y': m15.emotion, 'proba': m15.iloc[:, 3:], 'name': 'SVM_ADA'})
    # models.append({'y': m16.emotion, 'proba': m16.iloc[:, 3:], 'name': 'RF_ADA'})

    #roc_auc_curves(mode='val', models=models)
    #roc_auc_curves(mode='val', y=self.data['Y_val'], proba=self.data['val_proba'])
    precision_recall_curves(models=models, title='1st iteration base')
    #precision_recall_curve(mode='val', y=self.data['Y_val'], proba=self.data['val_proba'])
