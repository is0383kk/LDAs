from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import textacy

max_features = 10000
#csv_name = "winemag-data-130k-v2"
#csv_name = "light_wine2" # 2文書
csv_name = "light_wine" # 10文書
#csv_name = "100_wine" # 100文書
#csv_name = "500_wine" # 500文書
#csv_name = "light_japanese" # 日本語テスト

def process(text):
    return textacy.preprocess.preprocess_text()


print('Verifying data')
for item in [csv_name + '.csv']:
    item = join('input_data', item)
    name = item.split('.')[0]
    if not isfile(item):
        raise ValueError('No input file "%s"' % item)

print('Loading data')
df = pd.read_csv(join('input_data', csv_name + '.csv'))

print('Cleaning data')
print(df["description"])
df['processed_text'] = df['description']
#df['processed_text'] = df['description'].apply(process)
print("df['processed_text']->\n"+ str(df['processed_text']))


print('Vectorising data')
vectorizer = CountVectorizer(stop_words='english', max_features=max_features, max_df=0.9)
#vectorizer = CountVectorizer(max_features=max_features)
term_document = vectorizer.fit_transform(df['processed_text'])
print("term_document\n"+str(term_document))
print("------------------")
print(vectorizer.vocabulary_)
#print(term_document.astype(np.float32))
"""
{'単語':index,'単語':index}
のように保存
"""

print('Saving data')
# TODO validation
length = term_document.shape[0]
harf_length = int(length/2)
print(harf_length)
print("------------------")
print(term_document.shape)
#print(term_document[int(harf_length):,:])
print(term_document)
save_npz(file='data/train.txt.npz', matrix=term_document[int(harf_length):, :].astype(np.float32))
save_npz(file='data/test.txt.npz', matrix=term_document[:int(harf_length), :].astype(np.float32))

with open('data/vocab.pkl', 'wb') as f:
    pickle.dump(vectorizer.vocabulary_, f)
"""
term_document
(文書インデックス,単語インデックス) 出現回数
"""
