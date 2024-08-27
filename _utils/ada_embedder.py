import boto3
import json
import translators as ts
import pandas as pd
from ada_inference import infere_ada_002
from data_upload import upload_to_bucket
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
from io import StringIO, BytesIO
import time

s3 = boto3.client('s3')
BUCKET_NAME = 'ba-torben-nattermann'
GERSTI = 'GerSti'
SEMEVAL = 'SemEval2007'
ISEAR = 'deISEAR'
GNE = 'GNE'


def load_from_s3(object_key):
    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
    pickle_content = response['Body'].read()
    with BytesIO(pickle_content) as bio:
        df = pd.read_pickle(bio)
    return df

def translate(text, from_language='en', to_language='de', translator='bing'):
    translation = ts.translate_text(query_text=text, translator=translator, from_language=from_language,
                                    to_language=to_language)
    return translation


def concat_batches(name):
    file_paths = sorted(glob.glob(f'data/{name}/{name}*.pkl'))
    dfs = [pd.read_pickle(file) for file in file_paths]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.to_pickle(f'data/{name}/{name}.pkl')


def max_vote(row, threshold):
    max_value = row.max()
    if max_value >= threshold:
        return row.idxmax()
    else:
        return 'no emotion'


class DataLoaderGerSti:
    """
    Class used to preform ada embedding on GerSti dataset
    """

    def load_GerSti(self):
        object_key = f'{GERSTI}/data.jsonl'
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        jsonl_content = response['Body'].read().decode('utf-8')
        data = []
        keys = ['id', 'source', 'text']
        for line in jsonl_content.splitlines():
            entry = json.loads(line)
            cleaned = {key: entry[key] for key in keys}
            cleaned['emotion'] = entry['gold']['emotion']
            data.append(cleaned)
        return data

    def GerSti_formathandler(self):
        batch_size = 50
        raw = self.load_GerSti()
        df = pd.DataFrame(raw)
        resampled_df = df.groupby('emotion').apply(lambda x: x.sample(n=min(230, len(x)), random_state=42)).set_index(
            'id').sort_values(by='id') # remove some of the neutral entries
        for batch_idx in range(0, len(resampled_df), batch_size):
            batch = resampled_df.iloc[batch_idx:batch_idx + batch_size].copy()
            batch['ada_embedding'] = infere_ada_002(batch.text.tolist())
            file_path = f'{GERSTI}{batch_idx}.pkl'
            batch.to_pickle(file_path)
            print(f'Success in processing batch from {batch_idx} to {batch_idx + batch_size}')
        concat_batches(GERSTI)
        upload_to_bucket(file_path=f'{GERSTI}.pkl', object_key=f'{GERSTI}/cleaned.pkl')


    def translate_GerSti(self):
        """
        :return: writes translation to csv
        """
        df = pd.DataFrame(self.load_GerSti())
        df = df.groupby('emotion').apply(lambda x: x.sample(n=min(230, len(x)), random_state=42)).set_index(
            'id').sort_values(by='id')
        # with open('../data/GerSti/skipped.json', 'r') as json_file:
        #     skipped = json.load(json_file)
        # df = df.loc[skipped]
        batch_size = 50
        # Iterate through the DataFrame rows and perform translations
        skipped = []
        for batch_idx in range(0, len(df), batch_size):
            result_data = []
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            for index, row in tqdm(batch[['text']].iterrows(), total=len(batch)):
                id_value = index
                original = row['text']
                # Perform translations
                try:
                    bing = translate(original, from_language='de', to_language='en', translator='bing')
                    result_data.append(
                        {'id': id_value, 'text': original, 'bing': bing})
                except:
                    print('1st error')
                    time.sleep(1.5)
                    try:
                        bing = translate(original, from_language='de', to_language='en', translator='bing')
                        result_data.append(
                            {'id': id_value, 'text': original, 'bing': bing})
                    except:
                        print('2nd error, skipping case')
                        skipped.append(id_value)
            print(f'translation complete for batch{batch_idx}, store to csv\n------')
            # Create a DataFrame from the list of dictionaries
            result_df = pd.DataFrame(result_data)

            # Write the result DataFrame to a CSV file
            result_df.to_csv(f'../data/GerSti/GerStiTranslation_retry.csv', index=False)
        with open('../data/GerSti/skipped.json', 'w') as fp:
            json.dump(skipped, fp)

    def GerSti_retranslation(self):
        """
        :return: writes retranslation to csv
        """
        df = pd.read_csv('../data/GerSti/GerStiTranslation.csv')
        with open('../data/GerSti/skipped.json', 'r') as json_file:
            skipped = json.load(json_file)
        df = df.loc[skipped]
        batch_size = 50
        # Iterate through the DataFrame rows and perform translations
        skipped = []
        for batch_idx in range(0, len(df), batch_size):
            result_data = []
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            for index, row in tqdm(batch[['bing']].iterrows(), total=len(batch)):
                id_value = index
                original = row['bing']
                # Perform translations
                try:
                    bing = translate(original, from_language='en', to_language='de', translator='bing')
                    result_data.append(
                        {'id': id_value, 'retranslation': bing})
                except:
                    print('1st error')
                    time.sleep(1.5)
                    try:
                        bing = translate(original, from_language='en', to_language='de', translator='bing')
                        result_data.append(
                            {'id': id_value, 'retranslation': bing})
                    except:
                        print('2nd error, skipping case')
                        skipped.append(id_value)
            print(f'translation complete for batch{batch_idx}, store to csv\n------')
            # Create a DataFrame from the list of dictionaries
            result_df = pd.DataFrame(result_data)

            # Write the result DataFrame to a CSV file
            result_df.to_csv(f'../data/GerSti/GerStiRetranslation_retry1_{batch_idx}.csv', index=False)
        with open('../data/GerSti/skipped.json', 'w') as fp:
            json.dump(skipped, fp)

class DataLoaderSemEval2007:
    """
    Class used to preform ada embedding on SemEval2007 dataset
    """

    @staticmethod
    def xml_to_df(file_path):
        root = ET.parse(file_path)
        data = {'id': [], 'text': []}
        for instance in root.findall('instance'):
            instance_id = instance.get('id')
            instance_text = instance.text
            data['id'].append(instance_id)
            data['text'].append(instance_text)
            df = pd.DataFrame(data)
            df['id'] = df['id'].astype(int)
        return df


    def txt_to_df(self, file_path):
        with open(file_path, 'r') as file:
            data_list = [line.strip().split() for line in file]
        df = pd.DataFrame(data_list, columns=['id', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']).astype(
            int)
        df['emotion'] = df.drop('id', axis=1).apply(max_vote, axis=1, threshold=30)
        return df

    def load_SemEval2007(self):
        X1 = self.xml_to_df('/home/tnattermann/Documents/Thesis/data/SemEval2007/AffectiveText.test/affectivetext_test.xml')
        X2 = self.xml_to_df('/home/tnattermann/Documents/Thesis/data/SemEval2007/AffectiveText.trial/affectivetext_trial.xml')
        y1 = self.txt_to_df('/home/tnattermann/Documents/Thesis/data/SemEval2007/AffectiveText.test/affectivetext_test.emotions.gold')
        y2 = self.txt_to_df('/home/tnattermann/Documents/Thesis/data/SemEval2007/AffectiveText.trial/affectivetext_trial.emotions.gold')
        X_con = pd.concat([X1, X2]).sort_values(by='id').set_index('id')
        Y_con = pd.concat([y1, y2], axis=0).sort_values(by='id').set_index('id')
        print('dataset loaded')
        return pd.merge(X_con, Y_con, left_index=True, right_index=True)

    def filter_embeddings(self, threshold=0.85):
        """
        :param threshold: minimal required cosine similarity between original and retranslation to include sample
        :return: only "good" translations, above defined threshold
        """
        sims = pd.read_csv('../data/SemEval2007/SemEvalCosSim.csv')  # computed bert cos similarities
        trans = pd.read_csv('../data/SemEval2007/SemEvalTranslation.csv')  # bing translations
        data = self.load_SemEval2007()  # preprocessed dataset with english text and labels
        total = sims.O_RT.shape[0]
        selected = np.where(threshold < sims.O_RT)[0]  # only filter with good translation (above cos-sim threshold)
        print(f'Selected Percentage (Cos-Sim > {threshold}): {len(selected) / total}%')
        print('Mean Cos_Sim after filter:', sims.iloc[selected].O_RT.mean())
        dataset = pd.concat([trans, data.reset_index().drop(['id', 'text'], axis=1)], axis=1)  # concatenate sets
        return dataset.iloc[selected].set_index('id')  # only return filtered indices

    def SemEval2007_formathandler(self):
        batch_size = 50
        #df = self.filter_embeddings()
        df = self.load_SemEval2007()

        for batch_idx in range(0, len(df), batch_size):
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            batch['ada_embedding'] = infere_ada_002(batch.text.tolist())  # embedding of german bing translation
            file_path = f'../data/{SEMEVAL}/{SEMEVAL}_{batch_idx}_en.pkl'
            batch.to_pickle(file_path)
            print(f'Success in processing batch from {batch_idx} to {batch_idx + batch_size}')
        concat_batches(SEMEVAL)
        upload_to_bucket(file_path=f'../data/{SEMEVAL}/{SEMEVAL}.pkl', object_key=f'{SEMEVAL}/cleaned_en.pkl')
        print('-----\nUpload to S3 completed')

    def translate_SemEval2007(self):
        """
        :return: writes translation to csv
        """
        df = self.load_SemEval2007()
        result_data = []
        # Iterate through the DataFrame rows and perform translations
        for index, row in tqdm(df[['text']].iterrows(), total=len(df)):
            id_value = index
            original = row['text']

            # Perform translations
            bing = translate(original, from_language='en', to_language='de', translator='bing')
            result_data.append(
                {'id': id_value, 'text': original, 'bing': bing})
        print('translation complete, store to csv\n------')
        # Create a DataFrame from the list of dictionaries
        result_df = pd.DataFrame(result_data)

        # Write the result DataFrame to a CSV file
        result_df.to_csv('SemEvalTranslation.csv', index=False)

    def SemEval2007_retranslation(self):
        """
        :return: writes retranslation to csv
        """
        df = pd.read_csv('../data/SemEval2007/SemEvalTranslation.csv')
        result_data = []
        for text in tqdm(df['bing']):
            trans = translate(text, from_language='de', to_language='en', translator='bing')
            result_data.append({'retranslation': trans})
        print('retranslation complete, store to csv\n------')
        result_df = pd.DataFrame(result_data)
        concat = pd.concat([df, result_df], axis=1)
        concat.to_csv('SemEvalRetranslation.csv', index=False)

class DataLoaderdeISEAR:
    """
    Class used to preform ada embedding on ISEAR dataset
    """

    def load_deISEAR(self):
        object_key = 'deISEAR/deISEAR.tsv'
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        tsv_content = response['Body'].read().decode('utf-8')
        tsv_df = pd.read_csv(StringIO(tsv_content), sep='\t')
        df = tsv_df[['Sentence', 'Prior_Emotion', 'Angst', 'Ekel', 'Freude', 'Scham', 'Schuld', 'Traurigkeit', 'Wut']]

        eng = ['fear', 'joy', 'sadness', 'anger', 'disgust', 'shame', 'guilt']
        emotions = ['Angst', 'Freude', 'Traurigkeit', 'Wut', 'Ekel', 'Scham', 'Schuld']
        column_mapping = dict(zip(emotions, eng))
        df.rename(columns=column_mapping, inplace=True)

        df['emotion'] = df.loc[:, eng].apply(max_vote, axis=1, threshold=3)
        return df

    def deISEAR_formathandler(self):
        batch_size = 50
        df = self.load_deISEAR()
        for batch_idx in range(0, len(df), batch_size):
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            batch['ada_embedding'] = infere_ada_002(batch.Sentence.tolist())
            file_path = f'data/{ISEAR}/{ISEAR}_{batch_idx}.pkl'
            batch.to_pickle(file_path)
            print(f'Success in processing batch from {batch_idx} to {batch_idx + batch_size}')
        concat_batches(ISEAR)
        upload_to_bucket(file_path=f'data/{ISEAR}/{ISEAR}.pkl', object_key=f'{ISEAR}/cleaned.pkl')


class DataLoaderGNE:
    """
    Class used to preform ada embedding on GNE dataset
    """

    def load_GNE(self):
        object_key = 'GoodNewsEveryone/raw.tsv'
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        tsv_content = response['Body'].read().decode('utf-8')
        tsv_df = pd.read_csv(StringIO(tsv_content), sep='\t')
        tsv_df = tsv_df[['headline', 'dominant_emotion', 'other_emotions', 'reader_emotions']]
        # more preprocessing! -> remove columns i dont want to store
        return tsv_df

    def filter_embeddings(self, threshold=0.85):
        sims = pd.read_csv('../data/GNE/GNECosSim.csv')  # computed bert cos similarities
        trans = pd.read_csv('../data/GNE/GNETranslation.csv')  # bing translations
        data = self.load_GNE()  # preprocessed dataset with english text and labels

        total = sims.O_RT.shape[0]
        selected = np.where(threshold < sims.O_RT)[0]  # only filter with good translation (above cos-sim threshold)
        print(f'Selected Percentage (Cos-Sim > {threshold}): {len(selected) / total}%')
        print('Mean Cos_Sim after filter:', sims.iloc[selected].O_RT.mean())
        dataset = pd.concat([trans, data.reset_index().drop(['index', 'headline'], axis=1)], axis=1)  # concatenate sets
        return dataset.iloc[selected]  # only return filtered indices

    def GNE_formathandler(self):
        batch_size = 50
        #df = self.filter_embeddings()
        df = self.load_GNE()
        for batch_idx in range(0, len(df), batch_size):
            batch = df.iloc[batch_idx:batch_idx + batch_size].copy()
            batch['ada_embedding'] = infere_ada_002(batch.headline.tolist())  # embedding of german bing translation
            file_path = f'../data/{GNE}/{GNE}_{batch_idx}_en.pkl'
            batch.to_pickle(file_path)
            print(f'Success in processing batch from {batch_idx} to {batch_idx + batch_size}')
        concat_batches(GNE)
        upload_to_bucket(file_path=f'data/{GNE}/{GNE}.pkl', object_key=f'{GNE}/cleaned_en.pkl')
        print('-----\nUpload to S3 completed')



if __name__ == "__main__":
    #DataLoaderGNE().filter_embeddings()
    #DataLoaderGNE().GNE_formathandler()
    #df2 = DataLoaderGNE().load_GNE()
    #d = pd.read_pickle('../data/GNE/GNE_0_en.pkl')
    #df = DataLoaderGNE().load_GNE()
    #concat_batches('GNE')
    #upload_to_bucket(file_path=f'../data/{GNE}/{GNE}.pkl', object_key=f'{GNE}/cleaned_en.pkl')
    #df2 = DataLoaderSemEval2007().load_SemEval2007()
    #df = load_from_s3(f'GNE/cleaned.pkl')
    #df2 = load_from_s3('GerSti/cleaned.pkl')
    #df = DataLoaderGerSti().load_GerSti()
    #translate('Übersetze mich bitte', from_language='de', to_language='en', translator='deepl')
    #DataLoaderGerSti().GerSti_retranslation()
    #df = pd.read_csv('../data/GerSti/GerStiTranslation.csv')
    #concat_batches('GerSti')
    df1 = pd.read_csv('../data/GerSti/GerStiTranslation.csv')
    df2 = pd.read_csv('../data/GerSti/GerStiRetranslation.csv')
    concat = pd.concat([df1, df2], axis=1).drop(columns=['id'])
    concat.to_csv('../data/GerSti/GerStiTrans_full.csv', index=False)
    test = pd.read_csv('../data/GerSti/GerStiTrans_full.csv')
    print('Finished')

    #DataLoaderSemEval2007().SemEval2007_formathandler()
