from data_preprocessor import DataPreprocessor, load_txt_from_s3
from _utils.pantheon_dataloader import fetch_pantheon
from _utils.plutchik_helper import PlutchikHelper
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import time
from concurrent.futures import ProcessPoolExecutor

EX_GERSTI = ['emotion', 'id', 'source']
EX_SEMEVAL = ['engl_text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'emotion']


class PlutchikScore:
    """
    Class performs PlutchikScore computation
    """
    def __init__(self, emotion):
        self.start_time = time.time()
        self.emotion = emotion
        self.semger = DataPreprocessor().merge_SEMGER(base_emo=False, exclusion=[EX_GERSTI, EX_SEMEVAL])[0]
        self.variant, self.referrer = fetch_pantheon()
        self.nrc = load_txt_from_s3('NRC/German_NRC.txt')
        self.nrc_dict = None
        self.total_word_count = None
        self.corpus = None
        self.word_proba_m = None
        self.PMI_storage = {}
        self.run()

    def run(self):
        self.nrc_dict, self.corpus, self.word_proba_m, self.total_word_count = (PlutchikHelper(self.semger, self.variant,
                                                                                         self.referrer, self.nrc)
                                                                                .preprocessor())
        print(f'Preprocessing finished, took {time.time()-self.start_time} seconds\n Starting with Score computation...')
        self.plutchik_score()
        self.corpus[:5].to_pickle('data/Plutchik_Score/Score_augmentation.pkl')
        print(f'Computation finished, took {time.time()-self.start_time} seconds')

    def plutchik_score(self):
        self.corpus[f'{self.emotion}_plutchik_max'] = 0
        self.corpus[f'{self.emotion}_plutchik_min'] = 0
        self.corpus[f'{self.emotion}_plutchik_mean'] = 0
        self.corpus[f'{self.emotion}_plutchik_max'] = self.corpus[f'{self.emotion}_plutchik_max'].astype(float)
        self.corpus[f'{self.emotion}_plutchik_min'] = self.corpus[f'{self.emotion}_plutchik_min'].astype(float)
        self.corpus[f'{self.emotion}_plutchik_mean'] = self.corpus[f'{self.emotion}_plutchik_mean'].astype(float)

        nrc = self.nrc_dict[self.emotion]  # stores P(ck) per emotion word # store already computed PMIs
        num_processes = 4

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit tasks to the pool asynchronously
            [executor.submit(self.compute_PMI, index, text, nrc) for index, text in enumerate(self.corpus.tokens[:8])]

            # Wait for all tasks to complete and retrieve results
            #results = [future.result() for future in futures]

        #docs = enumerate(tqdm(self.corpus.tokens[:5], desc="text level", unit="iteration"))
        #    executor = concurrent.futures.ProcessPoolExecutor(5)
        #    futures = [executor.submit(self.compute_PMI, (docs, nrc)) for group in _grouper(5, docs)]
             # parallelize this function

    def compute_PMI(self, index, text, nrc):
        PMI_scores = np.empty((0,))
        tokens = word_tokenize(text, language='german')[:10]  # consider first 10 words
        for token in tqdm(tokens, desc="word level", unit="iteration", leave=False):  # loop over tokens
            if token not in self.word_proba_m.columns: # set score to 1 if word not in corpus
                PMI_scores = np.append(PMI_scores, 1)
            elif token in self.PMI_storage.keys():  # reduce computational effort
                PMI_scores = np.append(PMI_scores, self.PMI_storage[token])
            else:
                p_token = self.word_proba_m[token][0]  # word probability
                PMI_subscore = np.empty((0,))
                count_valids = 0
                for emo in nrc.index:  # loop over emotion words
                    p_emo = nrc.loc[emo].iloc[0]
                    p_tXe = self.compute_proba_both(self.corpus.tokens, token, emo)
                    if (p_emo * p_token > 0.0000000001) & (p_tXe != 0):
                        count_valids += 1
                        # print('Included:', emo, token, p_emo, p_token, p_tXe)
                        PMI = np.log(p_tXe / (p_emo * p_token))  # formula for PMI
                        # print('Resulting PMI:',  PMI)
                        PMI_subscore = np.append(PMI_subscore, PMI)
                # exponent = 1 / len(nrc)  # account for removal of cases: 1/count_valids
                exponent = 1 / 10
                # base = np.prod(PMI_subscore)
                base = np.sum(PMI_subscore)  # rewrite formula, use sum instead of product
                PMI_computed = np.power(base, exponent)  # formula for wordXcategory multiplication
                print('PMI_computed: ', PMI_computed)
                print('valids: ', count_valids)
                self.PMI_storage[token] = PMI_computed
                self.PMI_scores = np.append(self.PMI_scores, PMI_computed)

        print('Results: ', np.max(self.PMI_scores), np.min(self.PMI_scores), np.mean(self.PMI_scores))
        self.corpus.loc[index, f'{self.emotion}_plutchik_max'] = np.max(PMI_scores)
        self.corpus.loc[index, f'{self.emotion}_plutchik_min'] = np.min(PMI_scores)
        self.corpus.loc[index, f'{self.emotion}_plutchik_mean'] = np.mean(PMI_scores)

    def compute_proba_both(self, data, A, B):
        # Example DataFrame with a 'text' column
        df = pd.DataFrame(data)
        # Create a new column indicating if both words A and B are present
        df['both_words_present'] = (data.str.contains(A) & data.str.contains(B)).astype(int)
        # Print the sum of the new column
        both_words_sum = df['both_words_present'].sum()
        # Delete the new column
        del df['both_words_present']
        return both_words_sum / len(df)


if __name__ == "__main__":
    PlutchikScore('fear')


