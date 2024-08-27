from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import numpy as np
import pandas as pd
import string
import re
from nltk.corpus import stopwords

NRC_EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness',
                'surprise', 'trust']
g_stopwords = list(stopwords.words('german'))


class PlutchikHelper:

    def __init__(self, semger, variant, referrer, nrc):
        self.semger = semger
        self.variant = variant
        self.referrer = referrer
        self.nrc = nrc
        self.corpus = None
        self.total_word_count = None
        self.nrc_dict = None

    def preprocessor(self):
        # DATA LOADING
        semger_tok = pd.DataFrame()
        semger_tok['tokens'] = self.semger['text'].apply(self.preprocess_text_german)

        self.referrer['text'] = self.referrer['text'].replace('', np.nan)
        self.referrer = self.referrer.dropna(subset=['text'])[:10000]  # restrict manually
        referrer_tok = pd.DataFrame()
        referrer_tok['tokens'] = self.referrer['text'].apply(self.preprocess_text_german)

        self.variant['text'] = self.variant['text'].replace('', np.nan)
        self.variant = self.variant.dropna(subset=['text'])
        variant_tok = pd.DataFrame()
        variant_tok['tokens'] = self.variant['text'].apply(self.preprocess_text_german)
        self.corpus = referrer_tok.sample(n=50, random_state=42) # implementation for mini-corpus = 5000
        #self.corpus = pd.concat([semger_tok, referrer_tok, variant_tok], axis=0)

        # NRC LEXICON LOADING
        self.nrc['stemm'] = self.nrc['German Word'].apply(self.preprocess_text_german)
        self.total_word_count = self.corpus['tokens'].str.split().apply(len).sum()
        self.nrc_dict = {emotion: self.filter_emotion_words(emotion) for emotion in NRC_EMOTIONS}
        self.append_counts_nrc()
        # HELPER MATRIX COMPUTATION
        word_proba_m = self.word_proba_matrix(self.corpus)
        return self.nrc_dict, self.corpus, word_proba_m, self.total_word_count

    def filter_emotion_words(self, emotion):
        filtered_indices = self.nrc[self.nrc[emotion] == 1].index
        return pd.DataFrame(
            self.nrc.iloc[filtered_indices].stemm.drop_duplicates())  # stemm and drop resulting duplicates

    def word_proba_matrix(self, data):
        # Combine all texts into one corpus
        corpus = data['tokens'].str.cat(sep=' ')

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([corpus])

        # Convert the TF-IDF matrix to a DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # convert tf-idf
        return tfidf_df.div(tfidf_df.sum(axis=1), axis=0)

    def preprocess_text_german(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and digits
        text = re.sub(f"[{string.punctuation}\d]", "", text)
        # handle problematic punctuations separately ([„“–])
        special_characters = ['\u201E', '\u201C', '\u2013']
        pattern = f"[{''.join(map(re.escape, special_characters))}]"
        text = re.sub(pattern, "", text)

        # Tokenization using NLTK
        tokens = word_tokenize(text, language='german')

        # Remove stop words (both English and German)
        all_stop_words = set(stopwords.words('english')).union(g_stopwords)
        tokens = [token for token in tokens if token not in all_stop_words]

        # Stemming using NLTK SnowballStemmer for German
        stemmer = SnowballStemmer(language='german')
        tokens = [stemmer.stem(token) for token in tokens]

        # Join the tokens back into a string
        processed_text = ' '.join(tokens)

        return processed_text

    def word_proba(self, word_to_count):
        # Count the occurrences of the word in each text entry
        word_occurrences = self.corpus['tokens'].str.count(word_to_count)
        # Sum the occurrences to get the total frequency
        proba = word_occurrences.sum() / self.total_word_count
        return proba

    def append_counts_nrc(self):
        for key in self.nrc_dict.keys():
            self.nrc_dict[key]['word_count'] = self.nrc_dict[key]['stemm'].apply(self.word_proba)
            self.nrc_dict[key].reset_index(inplace=True)
            self.nrc_dict[key].drop('index', axis=1, inplace=True)
            self.nrc_dict[key].set_index('stemm', inplace=True)
