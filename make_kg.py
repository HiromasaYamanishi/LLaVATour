from pandarallel import pandarallel
import spacy
import pandas as pd
import pickle

nlp = spacy.load('ja_ginza')

def f(token, i):
    if i == 5 or token.children is None:
        return []
    results = []
    for child in token.children:
        if child.pos_ == 'NOUN':
            results.append(child.text)
    for child in token.children:
        if len(results) > 2:
            break
        if child.pos_ != 'NOUN' and child != token:
            results.extend(f(child, i + 1))
    return results

def g(token):
    current = token
    count = 0
    while True:
        next_token = current.head
        if next_token == current:
            break
        if next_token.pos_ == 'NOUN':
            return next_token.text
        current = next_token
        count += 1
        if count == 10:
            break
    return None

def find_adj_noun_pairs(text):
    # Process the text
    doc = nlp(text)

    # Store the pairs
    pairs = []

    # Iterate through all tokens in the doc
    for token in doc:
        # Check if the token is an adjective
        if token.pos_ == 'ADJ':
            # Trace dependency links to find connected nouns
            result = f(token, 0)
            if g(token) is not None:
                result.append(g(token))
            pairs.extend([(token.text, r) for r in result])

    return pairs

df_review = pd.read_pickle(
    "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
)
df_review_tmp = df_review
pandarallel.initialize(progress_bar=True)
df_review_tmp['pairs'] = df_review_tmp['review'].parallel_apply(find_adj_noun_pairs)
with open('./data/pairs.pkl', 'wb') as f:
    pickle.dump(df_review_tmp['pairs'], f)

    