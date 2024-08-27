import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cossim(emb_text, emb_list):
    """
    Used to compute cos_sim between a datapoint and the mean of a category
    :param emb_text: single embedding of interest
    :param emb_list: list of embeddings
    :return: Cosine Similarity between mean of list and input
    """
    emb_list_r = emb_list.apply(lambda x: x[0])
    mean_emb = np.array(emb_list_r.tolist()).mean(axis=0)
    try:
        sim = cosine_similarity(np.array(mean_emb).reshape(1, -1), np.array(emb_text).reshape(1, -1))[0, 0]
        return sim
    except:
        return 0

def count_words_in_list(text, word_list):
    words = text.split()  # Split the text into words
    count = sum(word in word_list for word in words)  # Count occurrences in the specified list
    if len(words) != 0:
        return count / len(words)
    else:
        return 0


def get_index_mask(df, n, filt,  indices=None):
    if filt:
        mixed = df.iloc[indices].drop(['English Word', 'German Word', 'tokens', 'positive', 'negative', 'fasttext_en', 'fasttext_de'], axis=1).sum(
            axis=1) >= n
    else:
        mixed = df.drop(['English Word', 'German Word', 'tokens', 'positive', 'negative', 'fasttext_en', 'fasttext_de'], axis=1).sum(axis=1) >= n

    return mixed[mixed == True].index


def filter_emotion_words(emotion, df, fasttext):
    filtered_indices = df[df[emotion] == 1].index
    i_2 = get_index_mask(df=df, n=2, filt=True, indices=filtered_indices)
    i_4 = get_index_mask(df=df, n=4, filt=True, indices=filtered_indices)
    df = pd.DataFrame(df.iloc[filtered_indices][['tokens', fasttext]])
    df['mixed_2'] = df.index.isin(i_2).astype(int)
    df['mixed_4'] = df.index.isin(i_4).astype(int)
    df.drop_duplicates(subset=['tokens'], inplace=True)
    return df.set_index('tokens')


def get_word_index(text, check_list):
    splits = text.split()
    for index, word in enumerate(splits):
        if word in check_list:
            return index + 1
    return 0
