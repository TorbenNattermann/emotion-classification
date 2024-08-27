from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

DATASET = 'GerSti'

class TranslationEvaluation:
    """
    Class used to obtain Bert embeddings for original and retranslated text and compare to cosine similarities
    """

    # def load_bert(self):
    #     model_name = 'bert-base-uncased'
    #     tokenizer = BertTokenizer.from_pretrained(model_name)
    #     model = BertModel.from_pretrained(model_name)
    #     return tokenizer, model

    # def bert_embedder(self, text, tokenizer, model):
    #     # Tokenize the input text
    #     tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    #     # Forward pass through the BERT model to obtain embeddings
    #     with torch.no_grad():
    #         outputs = model(**tokens)
    #     # Extract the embeddings from the output
    #     embeddings = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling as an example
    #     # Convert the embeddings tensor to a numpy array
    #     return embeddings.numpy()


    def embedder(self):
        tokenizer, model = self.load_bert()
        df = pd.read_csv('../data/GNE/GNEFullTrans.csv')
        result_data = []
        print('Start embedding\n -----')
        for index, row in tqdm(df.iterrows()):
            original = self.bert_embedder(row['text'], tokenizer, model)[0]
            bing = self.bert_embedder(row['bing'], tokenizer, model)[0]
            retrans = self.bert_embedder(row['retranslation'], tokenizer, model)[0]
            result_data.append({
                'original': original,
                'bing_bert': bing,
                'retrans_bert': retrans
            })
        embs = pd.DataFrame(result_data)
        embs.to_csv('data/GNE/GNETransEmbeddings.csv', index=False)


    def evaluate_embedding(self, persist=False):
        embs = pd.read_csv('../data/GNE/GNETransEmbeddings.csv')
        embs['original'] = embs['original'].apply(self.convert_to_array)
        embs['bing_bert'] = embs['bing_bert'].apply(self.convert_to_array)
        embs['retrans_bert'] = embs['retrans_bert'].apply(self.convert_to_array)
        # O = original, G = german, RT = retranslation
        cos_sim_O_G = cosine_similarity(embs['original'].tolist(), embs['bing_bert'].tolist()).diagonal()
        cos_sim_G_RT = cosine_similarity(embs['bing_bert'].tolist(), embs['retrans_bert'].tolist()).diagonal()
        cos_sim_O_RT = cosine_similarity(embs['original'].tolist(), embs['retrans_bert'].tolist()).diagonal()
        print(cos_sim_O_RT.mean())
        df = pd.DataFrame([cos_sim_O_G, cos_sim_G_RT, cos_sim_O_RT]).T
        df.columns = ['O_G', 'G_RT', 'O_RT']
        df.to_csv('data/GNE/GNECosSim.csv')
        if persist:
            plt.hist(cos_sim_O_RT, bins=30, color='skyblue', edgecolor='black')
            plt.title('Cos Similarity Distribution for Original-Retrans')
            plt.xlabel('Cosinus Similarities')
            plt.ylabel('Frequency')
            plt.savefig('Results/img/GNE_histogram_O_RT.png')
            plt.clf()

            # Plot histogram for Original-German
            plt.hist(cos_sim_O_G, bins=30, color='skyblue', edgecolor='black')
            plt.title('Cos Similarity Distribution for Original-German')
            plt.xlabel('Cosine Similarities')
            plt.ylabel('Frequency')
            plt.savefig('Results/img/GNE_histogram_O_G.png')
            self.persist_experiment(cos_sim_O_RT)


    def convert_to_array(self, string_representation):
        elements = string_representation.strip('[]').split()
        return np.array([float(element) for element in elements])


    def persist_experiment(self, O_RT):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Open a text file for writing
        with open(f'Results/{DATASET}_CosSimilarity_en.md', 'w') as file:
            # Write the header
            file.write(f"# Experiment Report\n{'-' * 18}\n\n")

            # Write experiment name and date
            file.write(f"## Experiment Name\n{DATASET} Translation Evaluation\n\n")
            file.write(f"## Date\n{current_date}\n\n")

            # Write summary
            file.write(f"## Summary\nEvaluation of the translation of the {DATASET} english news headlines to german "
                       f"using the Bing Translation API. For evaluation, retranslation to english is performed and cosine "
                       f"similarities between original and retranslation are computed.\n\n")

            # Write results
            file.write("## Results\n")

            # Write single values
            file.write("### Cosine Similarities Original - Retranslation \n")
            file.write(f"- Mean: {O_RT.mean()}\n")
            file.write(f"- Variance: {O_RT.var()}\n\n")
            file.write(f"- min: {O_RT.min()}\n\n")

            # Write histogram plot
            file.write("### Histograms\n")
            file.write("![Histogram Plot]('Results/img/GNE_histogram_O_RT.png')\n")


if __name__ == "__main__":
    # TranslationEvaluation().evaluate_embedding(persist=True)
    data = pd.read_csv('../data/SemEval2007/SemEvalCosSim.csv')
    data2 = pd.read_csv('../data/GNE/GNECosSim.csv')
    print(data['O_RT'].mean())
    print(data2['O_RT'].mean())

