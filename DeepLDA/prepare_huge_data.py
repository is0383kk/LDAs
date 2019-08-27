from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import textacy

max_features = 10000
#csv_name = "100_wine" # 100文書
csv_name = "500_wine" # 500文書
input_file = "huge_input_data"

def process(text):
    return textacy.preprocess.preprocess_text()


print('Verifying data')
for item in [csv_name + '.csv']:
    item = join( input_file , item)
    name = item.split('.')[0]
    if not isfile(item):
        raise ValueError('No input file "%s"' % item)

print('Loading data')
df = pd.read_csv(join( input_file , csv_name + '.csv'))

print('Cleaning data')
print(df["description"])
df['processed_text'] = df['description']
#df['processed_text'] = df['description'].apply(process)
print(df['processed_text'])

print('Vectorising data')
vectorizer = CountVectorizer(stop_words='english', max_features=max_features, max_df=0.9)
#vectorizer = CountVectorizer(max_features=max_features)

#print(vectorizer)
term_document = vectorizer.fit_transform(df['processed_text'])
print(term_document)
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
save_npz(file='npz_data/train.txt.npz', matrix=term_document[int(harf_length):, :].astype(np.float32))
save_npz(file='npz_data/test.txt.npz', matrix=term_document[:int(harf_length), :].astype(np.float32))

with open('npz_data/vocab.pkl', 'wb') as f:
    pickle.dump(vectorizer.vocabulary_, f)
"""
term_document
(文書インデックス,単語インデックス) 出現回数
"""
