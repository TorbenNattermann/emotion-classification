import boto3
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from _utils.ada_inference import infere_ada_002
from _utils.data_upload import upload_to_bucket
import glob
from io import BytesIO
from feature_engineering import FeatureEngineer
from tensorflow.keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt


CLASSMAPPING_5 = {'anger': 0, 'fear': 1, 'joy': 2, 'no emotion': 3, 'sadness': 4}
CLASSMAPPING_12 = {'anger': 0, 'annoyance': 1, 'disgust': 2, 'fear': 3, 'guilt': 4, 'joy': 5, 'neg. surprise': 6,
                   'pessimism': 7, 'no emotion': 8, 'pos. surprise': 9, 'optimism': 10, 'sadness': 11}
@staticmethod
def concat_batches(name):
    file_paths = sorted(glob.glob(f'data/{name}/{name}*.pkl'))
    dfs = [pd.read_pickle(file) for file in file_paths]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.to_pickle(f'data/{name}/{name}.pkl')
@staticmethod
def get_barplot(df, column, title):
    plt.close('all')
    sns.countplot(df, x=column)
    plt.xticks(rotation=45)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title}.png')
@staticmethod
def get_combi(df, column, title, xlabel, sort=False, filt=False, i=10, labels=None, **kwargs):
    """
    Plotting method to generate combi_plot, containing information on ctr performance and frequencies
    df: dataframe used for plotting
    column: column to perform analysis on
    title: Plot title
    xlabel: x-axis label
    sort: if True, sort category by ctr
    filt: if True, only include top i entries
    i: top i amount
    labels: dict that stores the index : representation mapping
    """
    plt.close('all')
    ctr = len(df.loc[df['label'] == 1]) / len(df)
    #plt.figure(figsize=(20, 15))
    if filt:
        filter_array = df[column].value_counts().iloc[:i].index.tolist()
        df = df[df[column].isin(filter_array)].copy()
    df_prep = df.groupby(by=column)[['label']].count().reset_index()
    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    if sort:
        sns.barplot(data=df, x=column, y='label', edgecolor=".5", facecolor=(0, 0, 0, 0), ax=ax,
                    order=df[f'{column}'].value_counts().index)
        sns.pointplot(data=df_prep, x=column, y='label', color='b', ax=ax_twin, markers='p', linestyles='dotted',
                      order=df[f'{column}'].value_counts().index)
    else:
        sns.barplot(data=df, x=column, y='label', edgecolor=".5", facecolor=(0, 0, 0, 0), ax=ax)
        sns.pointplot(data=df_prep, x=column, y='label', color='b', ax=ax_twin, markers='p', linestyles='dotted')
    ax.axhline(ctr, color='red')
    ax.set_xlabel(xlabel)
    if labels:
        ax.set_xticklabels(list(labels.values())[:i], rotation=90)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('CTR with global benchmark', color='r')
    ax_twin.set_ylabel('Total Impressions', color='b')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f'{title}.png')
    # return fig

class TransferReal:


    def __init__(self):
        self.bucket_name = 'ba-torben-nattermann'
        self.key = 'RealData/'
        self.s3_client = boto3.client('s3')
        self.dataset = self.load_s3_dataset()
        #self.embedd_referrer()
        #self.embeddings = self.load_from_s3()
        self.main()


    def main(self):
        # X = self.feature_engineering()
        X = pd.read_pickle('data/referrer/RealData_features.pkl')
        X_preds, X_preds_certain = self.load_ml_model(X)
        m_pred, m_pred_cert = self.merge_datasets(X_preds, X_preds_certain)
        get_combi(df=m_pred, column='prediction', title='CTR and Total Impressions for Complete Emotions', xlabel='predictions', sort=True) # adapt
        #get_combi(df=m_pred_cert, column='prediction', title='Test2', xlabel='emotions', sort=True)
        print('Counts for Impressions')
        print(m_pred['prediction'].value_counts())
        get_barplot(df=m_pred, column='prediction', title='Total Impressions for Complete Emotions') # adapt
        X_preds = pd.DataFrame(X_preds).rename(columns={0: 'prediction'})
        print('Counts for Referrer')
        print(X_preds['prediction'].value_counts())
        get_barplot(df=X_preds, column='prediction', title='Predicted Class Distribution for Complete Emotions') # adapt
        #get_barplot(df=m_pred_cert, column='prediction', title='Test4', xlabel='emotions', ylabel='Total Occurance')


    def load_s3_dataset(self):
        # Initialize Boto3 S3 client


        # List objects in the specified S3 bucket folder
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.key)

        # Initialize an empty list to store datasets
        datasets = []

        # Iterate over the objects in the folder
        for obj in response.get('Contents', []):
            # Extract the file key (path) for each object
            file_key = obj['Key']

            # Download the file from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)

            # Check if the file size is greater than 0
            file_size = response['ContentLength']
            if file_size == 0:
                print(f"File {file_key} is empty. Skipping...")
                continue

            # Read the downloaded file as a DataFrame
            try:
                data = response['Body'].read()
                file_obj = pa.BufferReader(data)
                table = pq.read_table(file_obj)
                df = table.to_pandas()
                datasets.append(df)
            except pd.errors.EmptyDataError:
                print(f"File {file_key} contains no data. Skipping...")
                continue
            except pd.errors.ParserError:
                print(f"Unable to parse file {file_key}. Skipping...")
                continue

        # Concatenate all datasets into a single DataFrame

        return pd.concat(datasets, ignore_index=True)

    def embedd_referrer(self):
        batch_size = 50
        data = self.dataset.groupby('primary_referrer_id')['attr_page_title'].max().reset_index()
        for batch_idx in range(0, len(data), batch_size):
            batch = data.iloc[batch_idx:batch_idx + batch_size].copy()
            batch['ada_embedding'] = infere_ada_002(batch.attr_page_title.tolist())
            file_path = f'data/referrer/referrer{batch_idx}.pkl'
            batch.to_pickle(file_path)
            print(f'Success in processing batch from {batch_idx} to {batch_idx + batch_size}')
        concat_batches('referrer')
        upload_to_bucket(file_path=f'data/referrer/referrer.pkl', object_key=f'referrer/ref_embeddings.pkl')

    def unpack_embeddings(self, df):
        return pd.concat(
            [df.drop('ada_embedding', axis=1), df['ada_embedding'].apply(pd.Series).add_prefix('ada_')], axis=1)

    def load_from_s3(self):
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key='referrer/ref_embeddings_cleaned.pkl')
        pickle_content = response['Body'].read()
        with BytesIO(pickle_content) as bio:
            df = pd.read_pickle(bio)
        return df

    def feature_engineering(self):
        print('Data loaded, starting Feature Engineering')
        X = self.unpack_embeddings(self.embeddings)
        X = X.rename(columns={'attr_page_title': 'text'})
        X, features = FeatureEngineer(data=X, feature_set={'ada': True,
                       'pca': False,
                       'n_pca': 300,
                       'tf_idf': False,
                       'linguistics': False,
                       'nrc': True,
                       'n_tfidf': 1000}, features='').main()
        X.drop('fasttext', axis=1, inplace=True)
        X.columns = X.columns.astype(str)  # convert to string for model learning
        X.drop(['tokens', 'text'], axis=1, inplace=True)  # remove required cleartext for feature engineering
        X.to_pickle('RealData_features.pkl')
        return X

    def load_ml_model(self, X):
        ref_id = X['primary_referrer_id']
        X = X.drop(['primary_referrer_id'], axis=1)

        checkpoint_filepath = f'Results/checkpoints/ckpt_model_basic.h5' # adapt
        model = models.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(1566,)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'), # adapt
            layers.Dropout(0.2), # adapt
            layers.Dense(5, activation='softmax')  # adapt
        ])
        model.load_weights(checkpoint_filepath)
        X_proba = pd.DataFrame(model.predict(X)).set_index(ref_id)
        X_preds = pd.DataFrame(X_proba).idxmax(axis=1).map({v: k for k, v in CLASSMAPPING_5.items()}) #adapt
        max_index = pd.DataFrame(X_proba).idxmax(axis=1)
        condition = pd.DataFrame(X_proba).max(axis=1) >= 0.6
        X_preds_certain = max_index[condition].map({v: k for k, v in CLASSMAPPING_5.items()}) # adapt
        print('Predictions loaded')
        return X_preds, X_preds_certain

    def merge_datasets(self, X_preds, X_preds_certain):
        inner_preds = pd.merge(X_preds.reset_index(), self.dataset, on='primary_referrer_id', how='inner')
        inner_preds = inner_preds.drop(['attr_page_description'], axis=1).set_index('primary_referrer_id')
        inner_preds = inner_preds.rename(columns={0: 'prediction'})
        inner_preds_certain = pd.merge(X_preds_certain.reset_index(), self.dataset, on='primary_referrer_id', how='inner')
        inner_preds_certain = inner_preds_certain.drop(['attr_page_description'], axis=1).set_index('primary_referrer_id')
        inner_preds_certain = inner_preds_certain.rename(columns={0: 'prediction'})
        return inner_preds, inner_preds_certain







    # function to load models
    # load ref_embeddings from s3, perform model inference, discard embeddings
    # implement differentiation max_value / certainty
    # join emotion to self.dataset
    # evaluation

if __name__ == "__main__":
    TransferReal()
