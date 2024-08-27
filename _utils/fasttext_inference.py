import pickle
import fasttext
import smart_open
import numpy as np
import datetime
from data_preprocessor import load_txt_from_s3, DataPreprocessor

S3_URL = 's3://pdm-pantheon-preprocessed/facebook_fasttext/wiki_word_vectors.pkl'

with smart_open.open(S3_URL, 'rb') as f:
    words_dict = pickle.load(f)

ZERO_WORD = np.zeros(300)
for v in words_dict.values():
    ZERO_WORD = np.zeros_like(v)
    break

def run():
    loader = DataPreprocessor()
    # GNE = loader.load_from_s3('GNE/cleaned.pkl')
    # SEMEVAL = loader.load_from_s3('SemEval2007/cleaned.pkl')
    GERSTI = loader.load_from_s3('GerSti/cleaned.pkl')
    #NRC = load_txt_from_s3('NRC/German_NRC.txt')
    # referrer = loader.load_from_s3('referrer/ref_embeddings.pkl')
    # start = datetime.datetime.now()
    # referrer['fasttext'] = referrer['attr_page_title'].apply(handler)
    # GNE['fasttext'] = GNE['text'].apply(handler)
    # SEMEVAL['fasttext'] = SEMEVAL['text'].apply(handler)
    # GERSTI['fasttext'] = GERSTI['text'].apply(handler)

    end = datetime.datetime.now()
    duration = end - start
    print('Fasttext computation duration: ', duration)
    # GNE.to_pickle('GNE_fasttext.pkl')
    # GERSTI.to_pickle('GERSTI_fasttext.pkl')
    # SEMEVAL.to_pickle('SEMEVAL_fasttext.pkl')
    referrer.to_pickle('referrer_fasttext.pkl')


def handler(input):
    #start = datetime.datetime.now()
    text = input.strip().lower()

    vectors = [
        words_dict[token] / np.linalg.norm(words_dict[token]) if token in words_dict else ZERO_WORD for token in fasttext.tokenize(text)
    ]

    if len(vectors) > 0:
        mean_vector = np.array(vectors).mean(axis=0)
    else:
        mean_vector = ZERO_WORD

    return mean_vector.tolist()



if __name__ == '__main__':
    run()


# import json
# import boto3
# FASTTEXT_ENCODER_LAMBDA = 'fasttext-encoder-lambda'
#
#
# def infere(input_text):
#     result = [0] * 300
#     response = boto3.client('lambda').invoke(
#         FunctionName=FASTTEXT_ENCODER_LAMBDA,
#         InvocationType='RequestResponse',
#         LogType='None',
#         Payload=json.dumps({
#             "text": input_text
#         }).encode())
#
#     if response['StatusCode'] == 200:
#         result = json.loads(response['Payload'].read())
#     return result
#

# if __name__ == "__main__":
#     texts = ["Andreas Scheuer: Maut operator contradicts the Minister of Transport"]
#     print(infere(texts))
