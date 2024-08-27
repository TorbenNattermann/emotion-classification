import boto3
from io import BytesIO, StringIO
import pandas as pd

s3 = boto3.client('s3')
BUCKET_NAME = 'ba-torben-nattermann'
GERSTI = 'GerSti'
SEMEVAL = 'SemEval2007'
GNE = 'GNE'


def load_txt_from_s3(object_key):
    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
    text_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(text_content), sep='\t')
    return df


class DataPreprocessor:

    def load_from_s3(self, object_key):
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        pickle_content = response['Body'].read()
        with BytesIO(pickle_content) as bio:
            df = pd.read_pickle(bio)
        return df

    def unpack_embeddings(self, df):
        return pd.concat(
            [df.drop('ada_embedding', axis=1), df['ada_embedding'].apply(pd.Series).add_prefix('ada_')], axis=1)

    def preprocess_data(self, language, dataset, exclusion, base_emo=False):
        if language == 'de':
            df = self.load_from_s3(f'{dataset}/cleaned.pkl')
        if language == 'en':
            df = self.load_from_s3(f'{dataset}/cleaned_en.pkl')
        if base_emo:
            base_emotions = ['fear', 'joy', 'sadness', 'anger', 'no emotion']
            df = df[df['emotion'].isin(base_emotions)]
        unpacked_df = self.unpack_embeddings(df)
        # handle inconsistency in dataset naming
        if 'Sentence' in unpacked_df.columns:
            unpacked_df.rename(columns={'Sentence': 'text'}, inplace=True)
        elif 'bing' in unpacked_df.columns:
            unpacked_df.rename(columns={'text': 'engl_text'}, inplace=True)
            unpacked_df.rename(columns={'bing': 'text'}, inplace=True)
        X = unpacked_df.drop(exclusion, axis=1)
        Y = unpacked_df['emotion']
        print(f'Dataset size: {len(X)}')
        return X, Y

    def merge_SEMGER(self, language, exclusion, base_emo=False):
        print('-----\nLoading GerSti with properties:')
        ger = self.preprocess_data(language=language, dataset=GERSTI, exclusion=exclusion[0],
                                   base_emo=base_emo)
        print('-----\nLoading SemEval2007 with properties:')
        sem = self.preprocess_data(language=language, dataset=SEMEVAL,
                                   exclusion=exclusion[1], base_emo=base_emo)
        sem0 = pd.concat([sem[0], sem[1]], axis=1)
        ger0 = pd.concat([ger[0], ger[1]], axis=1)
        if not base_emo:
            # replacement_dict = {'neg. surprise': 'surprise', 'pos. surprise': 'surprise'}  # combine surprise categories
            # ger0['emotion'] = ger0['emotion'].replace(replacement_dict)
            exclude_emotions = ['other', 'shame', 'hope']  # exclude non overlapping/small categories
            ger0 = ger0[~ger0['emotion'].isin(exclude_emotions)]
        merge = pd.concat([sem0, ger0], axis=0)
        # merge = merge[~merge['emotion'].isin(['disgust'])]  # exclude disgust, only 22 cases in merge
        print(f'Merged Dataset Size: {merge.shape[0]}')
        X = merge.drop('emotion', axis=1)
        Y = merge['emotion']
        return X, Y

    def merge_GNESEMGER(self, language, base_emo, exclusion, label_merge, resample):
        loading_semger = self.merge_SEMGER(language=language, base_emo=base_emo, exclusion=exclusion[:2])
        print('-----\nLoading GNE with properties:')
        loading_gne = self.preprocess_data(language=language, dataset=GNE, exclusion=exclusion[2],
                                           base_emo=base_emo)
        semger = pd.concat([loading_semger[0], loading_semger[1]], axis=1)
        gne = pd.concat([loading_gne[0], loading_gne[1]], axis=1)
        if not base_emo:
            replacement_dict = {'love_including_like': 'joy',
                                'positive_surprise': 'pos. surprise',
                                'negative_surprise': 'neg. surprise'}  # include like in joy
            gne['emotion'] = gne['emotion'].replace(replacement_dict)

        merge = pd.concat([semger, gne], axis=0)
        exclude_emotions = ['pride', 'shame', 'trust', 'surprise']  # exclude small
        merge = merge[~merge['emotion'].isin(exclude_emotions)]
        if label_merge:
            replacement_dict = {'annoyance': 'neg. surprise',
                                'disgust': 'anger',
                                'guilt': 'neg. surprise',
                                'negative_anticipation_including_pessimism': 'neg. surprise',
                                'positive_anticipation_including_optimism': 'joy'}
            merge['emotion'] = merge['emotion'].replace(replacement_dict)
            replacement_dict2 = {'neg. surprise': 'surprise',
                                 'pos. surprise': 'joy'}
            merge['emotion'] = merge['emotion'].replace(replacement_dict2)
            if resample:
                merge = (merge.groupby('emotion').apply(lambda x: x.sample(n=min(1100, len(x)), random_state=42))
                         .reset_index(drop=True))
        # merge = merge[~merge['emotion'].isin(['no emotion'])]
        print(f'Merged Dataset Size: {merge.shape[0]}')
        X = merge.drop('emotion', axis=1)
        Y = merge['emotion']
        return X, Y


if __name__ == "__main__":
    DataPreprocessor().preprocess_SemEval2007(base_emo=False)
