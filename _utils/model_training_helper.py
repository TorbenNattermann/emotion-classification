from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineer
from nn_training import train_nn
from nn_sweep import Sweeper
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

EX_GERSTI = ['emotion', 'id', 'source']
EX_SEMEVAL = ['engl_text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'emotion']
EX_ISEAR = ['Prior_Emotion', 'fear', 'disgust', 'joy', 'shame', 'guilt', 'sadness', 'anger', 'emotion']
EX_GNE = ['engl_text', 'emotion', 'other_emotions', 'reader_emotions']

class TrainerUtils:
    def __init__(self, main_class_instance):
        self.instance = main_class_instance

    def data_prep(self):
        if self.instance.language == 'en':
            EX_SEMEVAL.remove('engl_text')
            EX_GNE.remove('engl_text')
        if self.instance.dataset == 'GERSTI':
            X, Y = DataPreprocessor().preprocess_data(language=self.instance.language, dataset='GerSti',
                                                      exclusion=EX_GERSTI, base_emo=self.instance.base_emo)

        if self.instance.dataset == 'SEMEVAL':
            X, Y = DataPreprocessor().preprocess_data(language=self.instance.language, dataset='SemEval2007',
                                                      exclusion=EX_SEMEVAL, base_emo=self.instance.base_emo)
        if self.instance.dataset == 'ISEAR':
            X, Y = DataPreprocessor().preprocess_data(language=self.instance.language, dataset='deISEAR',
                                                      exclusion=EX_ISEAR, base_emo = self.instance.base_emo)

        if self.instance.dataset == 'GNE':
            X, Y = DataPreprocessor().preprocess_data(language=self.instance.language, dataset='GNE', exclusion=EX_GNE,
                                                      base_emo=self.instance.base_emo)
        if self.instance.dataset == 'SEM_GER':
            X, Y = DataPreprocessor().merge_SEMGER(language=self.instance.language, base_emo=self.instance.base_emo,
                                                   exclusion=[EX_GERSTI, EX_SEMEVAL])
        if self.instance.dataset == 'GNE_SEM_GER':
            X, Y = DataPreprocessor().merge_GNESEMGER(language=self.instance.language, base_emo=self.instance.base_emo,
                                                      exclusion=[EX_GERSTI, EX_SEMEVAL, EX_GNE],
                                                      label_merge=self.instance.label_merge, resample=self.instance.resample)
        print(f'Dataset {self.instance.dataset} loaded\n ------')
        features = ''
        if self.instance.feature_set['ada']:
            features += 'ada_embeddings&'
            if self.instance.feature_set['pca']:
                n_pca = self.instance.feature_set['n_pca']
                X_ = self.pca_decomp(X.drop(['text', 'fasttext'], axis=1), n_pca)
                X = pd.concat([X[['text', 'fasttext']].reset_index(drop=True), pd.DataFrame(X_)], axis=1)
                features = features[:-1]  # remove trailing &
                features += f'_{n_pca}_dims&'
        elif not self.instance.feature_set['ada']:
            X = X[['text', 'fasttext']]
        X, features = FeatureEngineer(X, self.instance.feature_set, features, self.instance.language).main()
        X.drop('fasttext', axis=1, inplace=True)
        X.columns = X.columns.astype(str)  # convert to string for model learning
        X.drop(['tokens', 'text'], axis=1, inplace=True)  # remove required cleartext for feature engineering
        print(f'------\nNumber labels: {Y.nunique()}\n Labels: {Y.unique()}')
        # test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # val split
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        # store test data locally
        X_test.to_csv(f'data/test_set/X_test_{self.instance.language}.csv', index=False)
        Y_test.to_csv(f'data/test_set/Y_test_{self.instance.language}.csv', index=False)
        return X_train, X_val, Y_train, Y_val, features

    def load_cached_dataset(self):
        print('Loading dataset from local cache.')
        base_emotions = ['fear', 'joy', 'sadness', 'anger', 'no emotion']
        if self.instance.feature_set['pca'] or self.instance.dataset != 'GNE_SEM_GER' or self.instance.resample or self.instance.label_merge:
            print('Requirements for cache not met, loading new dataset.')
            return self.data_prep()
        X_train = pd.read_csv(f'data/dataset_cache/X_train_{self.instance.language}.csv')
        Y_train = pd.read_csv(f'data/dataset_cache/Y_train_{self.instance.language}.csv')
        df = pd.concat([X_train, Y_train], axis=1)
        if self.instance.base_emo:
            df = df[df['emotion'].isin(base_emotions)]
            X_train = df.drop(['emotion'], axis=1)
            Y_train = df[['emotion']]
            # changes for test evaluation
            X_val = pd.read_csv(f'data/dataset_cache/test_sets/X_test_{self.instance.language}.csv')
            Y_val = pd.read_csv(f'data/dataset_cache/test_sets/Y_test_{self.instance.language}.csv')
            df = pd.concat([X_val, Y_val], axis=1)
            df = df[df['emotion'].isin(base_emotions)]
            X_val = df.drop(['emotion'], axis=1)
            Y_val = df[['emotion']]
        features = ''
        concat_train = pd.DataFrame()
        concat_test = pd.DataFrame() # adaption for test
        if self.instance.feature_set['ada']:
            features += 'ada&'
            concat_train = pd.concat([concat_train, X_train.iloc[:, :1536]], axis=1)
            concat_test = pd.concat([concat_test, X_val.iloc[:, :1536]], axis=1)
        if self.instance.feature_set['tf_idf']:
            features += f'tfidf_{self.instance.feature_set["n_tfidf"]}_dims&'
            concat_train = pd.concat([concat_train, X_train.iloc[:, 1536:2536]], axis=1)
            concat_test = pd.concat([concat_test, X_val.iloc[:, 1536:2536]], axis=1)
        if self.instance.feature_set['linguistics']:
            features += f'linguistics&'
            concat_train = pd.concat([concat_train, X_train.iloc[:, 2536:2540]], axis=1)
            concat_test = pd.concat([concat_test, X_val.iloc[:, 2536:2540]], axis=1)
        if self.instance.feature_set['nrc']:
            features += f'nrc&'
            concat_train = pd.concat([concat_train, X_train.iloc[:, 2540:]], axis=1)
            concat_test = pd.concat([concat_test, X_val.iloc[:, 2540:]], axis=1)
        if features[-1:] == '&':
            features = features[:-1]
        # changes for test evaluation
        #X_train, X_val, Y_train, Y_val = train_test_split(concat, Y_train, test_size=0.2, random_state=42)
        return concat_train, concat_test, Y_train, Y_val, features

    @staticmethod
    def pca_decomp(data, components):
        print(f'Dimension reduced to {components}\n ------')
        pca = PCA()
        result = pca.fit_transform(data)
        exp_matrix = pca.explained_variance_ratio_[:components]
        var_exp = np.cumsum(exp_matrix)
        print(f'Explained Variance on train_set: {var_exp[-1]}')
        reduced = result[:, :components]
        return reduced

    @staticmethod
    def log_dataframe_to_html(df, file_path):
        # Convert the DataFrame to an HTML table
        html_table = df.to_html(index=False)

        # Save the HTML table to the file in append mode
        with open(file_path, 'a') as html_file:
            # If this is the first table, write the opening HTML tags
            if html_file.tell() == 0:
                html_file.write('<html><body>\n')
            # Write the table
            html_file.write(html_table)

    def trigger_training(self):
        if self.instance.classifier == 'svm':
            print('Model: SVM\n ------')
            scaler_train = StandardScaler()
            scaler_val = StandardScaler()
            self.instance.X_train = scaler_train.fit_transform(self.instance.X_train)
            self.instance.X_val = scaler_val.fit_transform(self.instance.X_val)
            model = OneVsRestClassifier(SVC(probability=True, random_state=42, C=5, kernel='rbf'))
            if not self.instance.hypertuning:
                model = model.fit(self.instance.X_train, self.instance.Y_train)
        if self.instance.classifier == 'rf':
            print('Model: RF\n ------')
            model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
            if not self.instance.hypertuning:
                model = model.fit(self.instance.X_train, self.instance.Y_train)
        if self.instance.classifier == 'gb':
            print('Model: Gradient Boosting\n ------')
            model = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))
            if not self.instance.hypertuning:
                model = model.fit(self.instance.X_train, self.instance.Y_train)
        if self.instance.classifier == 'nn':
            if self.instance.nn_sweep:
                Sweeper(self.instance.X_train, self.instance.Y_train, self.instance.X_val, self.instance.Y_val, self.instance.configs)
            else:
                train_pred, Y_pred, train_proba, val_proba, classes = train_nn(self.instance.X_train, self.instance.Y_train, self.instance.X_val, self.instance.Y_val,
                                                                               self.instance.configs, self.instance.language, self.instance.model_id)
        else:
            if not self.instance.hypertuning:
                train_pred = model.predict(self.instance.X_train)
                train_proba = model.predict_proba(self.instance.X_train)
                Y_pred = model.predict(self.instance.X_val)
                val_proba = model.predict_proba(self.instance.X_val)
                classes = model.classes_
        if not self.instance.hypertuning:
            return train_pred, Y_pred, train_proba, val_proba, classes, model
        return model
