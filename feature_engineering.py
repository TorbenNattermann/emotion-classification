from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import spacy
from nltk.corpus import stopwords
import boto3
from io import BytesIO, StringIO
from _utils.feature_engineering_helper import (filter_emotion_words, get_index_mask, count_words_in_list, get_word_index
, compute_cossim)
import time

s3 = boto3.client('s3')
BUCKET_NAME = 'ba-torben-nattermann'
NRC_EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness',
                'surprise', 'trust']
FASTTEXT = 'fasttext_de'


class FeatureEngineer:

    @staticmethod
    def count_words(text):
        return len(text.split())

    def __init__(self, data, feature_set, features, language):
        self.start = time.time()
        self.data = data
        self.feature_set = feature_set
        self.features = features
        self.language = language
        self.fasttext = FASTTEXT
        if self.language == 'en':
            self.fasttext = 'fasttext_en'
        self.n_tfidf = feature_set['n_tfidf']
        if self.language == 'de':
            self.stopwords = list(stopwords.words('german'))
            self.nlp = spacy.load('de_core_news_sm')
        if self.language == 'en':
            self.stopwords = list(stopwords.words('english'))
            self.nlp = spacy.load('en_core_web_sm')
        self.lex = self.load_from_s3('NRC/cleaned.pkl')
        self.nrc_dict = None


    def main(self):
        self.data['tokens'] = self.data['text'].apply(self.preprocess_text)
        if self.feature_set['tf_idf']:
            self.compute_tfidf()
            self.features += f'tfidf_{self.n_tfidf}_dims&'
        if self.feature_set['linguistics']:
            self.compute_linguistics()
            self.features += f'linguistics&'
        if self.feature_set['nrc']:
            self.preprocess_nrc()
            self.compute_nrc_lex()
            self.features += f'nrc&'
        if self.features[-1:] == '&':
            self.features = self.features[:-1]
        print(f'------\nFeature Engineering completed in {round(time.time() - self.start, 2)} seconds.\n'
              f'Number of features: {len(self.data.columns)-3}')  # includes token, fasttext, text
        return self.data, self.features

    def compute_tfidf(self):
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=self.stopwords, max_features=self.n_tfidf)
        X_tfidf = tfidf_vectorizer.fit_transform(self.data['tokens'])
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        self.data.reset_index(inplace=True, drop=True)
        self.data = pd.concat([self.data, tfidf_df], axis=1)

    def compute_linguistics(self):
        self.data['count_exclamation'] = self.data['text'].str.count('!')
        self.data['count_question'] = self.data['text'].str.count('\?')
        self.data['count_words'] = self.data['text'].apply(self.count_words)
        self.data['count_chars/word'] = self.data['text'].str.len() / self.data['count_words']

    def preprocess_nrc(self):
        if self.language == 'de':
            self.lex['tokens'] = self.lex['German Word'].apply(self.preprocess_text)
        if self.language == 'en':
            self.lex['tokens'] = self.lex['English Word'].astype('str').apply(self.preprocess_text)
        any_emotion = self.lex.drop(['English Word', 'German Word', 'tokens', 'positive', 'negative', 'fasttext_en', 'fasttext_de'],
                                    axis=1).sum(axis=1) != 0
        pures = self.lex.drop(['English Word', 'German Word', 'tokens', 'positive', 'negative', 'fasttext_en', 'fasttext_de'],
                              axis=1).sum(axis=1) == 1
        pures_indices = pures[pures == True].index
        any_emotion_indices = any_emotion[any_emotion == True].index
        self.lex['mixed_2'] = self.lex.index.isin(get_index_mask(filt=False, n=2, df=self.lex)).astype(int)
        self.lex['mixed_4'] = self.lex.index.isin(get_index_mask(filt=False, n=4, df=self.lex)).astype(int)
        self.lex['pure_emo'] = self.lex.index.isin(pures_indices).astype(int)
        self.lex['any_emo'] = self.lex.index.isin(any_emotion_indices).astype(int)

        self.nrc_dict = {emotion: filter_emotion_words(emotion=emotion, df=self.lex, fasttext=self.fasttext) for emotion in NRC_EMOTIONS}

    def compute_nrc_lex(self):
        for emotion in NRC_EMOTIONS:  # count features (specific emotions + positive/negative)
            self.data[f'count_{emotion}'] = self.data['tokens'].apply(count_words_in_list,
                                                                      args=(self.nrc_dict[emotion].index,))
            self.data[f'cosSim_{emotion}'] = self.data['fasttext'].apply(compute_cossim,
                                                                         args=(self.nrc_dict[emotion][self.fasttext],))

        for column in ['mixed_2', 'mixed_4', 'pure_emo', 'any_emo']:  # general emotion features
            self.data[column] = self.data['tokens'].apply(count_words_in_list,
                                                          args=(self.lex[self.lex[column] == 1].tokens.to_list(),))
            self.data[f'cosSim_{column}'] = self.data['fasttext'].apply(compute_cossim,
                                                                        args=(self.lex[self.lex[column] == 1][self.fasttext],))
        self.data['position_positive'] = self.data['tokens'].apply(get_word_index,
                                                                   args=(self.nrc_dict['positive'].index,))
        self.data['position_negative'] = self.data['tokens'].apply(get_word_index,
                                                                   args=(self.nrc_dict['negative'].index,))
        # compute cos-features on fasttext features

    def preprocess_text(self, text):
        # Convert to lowercase
        doc = self.nlp(text)
        special_characters = ['\u201E', '\u201C', '\u2013']
        processed_tokens = [token.lemma_ for token in doc if
                            not token.is_digit and token.text not in special_characters]
        tokens_without_stopwords = [token for token in processed_tokens if token not in self.stopwords]
        processed_text = ' '.join(tokens_without_stopwords)
        text_without_hyphens = processed_text.replace('-', ' ')
        lowercase = text_without_hyphens.lower()
        return self.clean_whitespace(lowercase)

    def load_from_s3(self, object_key):
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        pickle_content = response['Body'].read()
        with BytesIO(pickle_content) as bio:
            df = pd.read_pickle(bio)
        return df

    def clean_whitespace(self, input_string):
        cleaned_string = input_string.strip()
        cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
        return cleaned_string


if __name__ == "__main__":
    data = pd.DataFrame({
        'text': ['Dies ist ein Beispielsatz.',
                 'Ein weiteres Beispiel für einen Satz.',
                 'Noch ein Beispiel.']
    })
    data = FeatureEngineer(data).main()
    print(data.columns)
    print(data.head())
