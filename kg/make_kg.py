from pandarallel import pandarallel
import spacy
import pandas as pd
import pickle
from collections import defaultdict

nlp = spacy.load('ja_ginza')

# def f(token, i):
#     if i == 5 or token.children is None:
#         return []
#     results = []
#     for child in token.children:
#         if child.pos_ == 'NOUN':
#             results.append(child.text)
#     for child in token.children:
#         if len(results) > 2:
#             break
#         if child.pos_ != 'NOUN' and child != token:
#             results.extend(f(child, i + 1))
#     return results

# def g(token):
#     current = token
#     count = 0
#     while True:
#         next_token = current.head
#         if next_token == current:
#             break
#         if next_token.pos_ == 'NOUN':
#             return next_token.text
#         current = next_token
#         count += 1
#         if count == 10:
#             break
#     return None

# def find_adj_noun_pairs(text):
#     # Process the text
#     doc = nlp(text)

#     # Store the pairs
#     pairs = []

#     # Iterate through all tokens in the doc
#     for token in doc:
#         # Check if the token is an adjective
#         if token.pos_ == 'ADJ':
#             # Trace dependency links to find connected nouns
#             result = f(token, 0)
#             if g(token) is not None:
#                 result.append(g(token))
#             pairs.extend([(token.text, r) for r in result])

#     return pairs

hiragana_stopwords = {chr(i) for i in range(12353, 12436)}  # ひらがなのUnicode範囲
hiragana_stopwords.update({'ゝ', 'ゞ', 'ー'})  # 特殊なひらがな拡張

# ストップワードのリストを定義
stopwords = list(hiragana_stopwords) + list({"は", "の", "が", "と", "に", "も", "で", "を", "ない", "なく"})
stopwords = stopwords + ["ところ", "お", "方", "こと", "ぶん", "たち", "お", "こと", "ため", "さん", "こう", "ずれ", "しょう", "ら", 'おり', "ゃ", 'なか', '後']

def get_compound(token):
    compound_parts = [token]
    current = token
    while True:
        compound_children = [child for child in current.children if child.dep_ in ["compound", 'nummod'] and child.i < current.i]
        if not compound_children:
            break
        compound_children.sort(key=lambda x: x.i)
        compound_parts = compound_children + compound_parts
        current = compound_children[0]
    
    compound = ''.join([t.lemma_ for t in compound_parts])
    return compound, compound_parts

def f(token, i, exclude_tokens, neighbor_depth, sentence):
    if i == neighbor_depth or token.children is None:
        return []
    results = []
    for child in token.children:
        if token.dep_ == "case" or child.text in stopwords:
            continue
        if child.pos_ in ['NOUN', 'PROPN'] and child not in exclude_tokens and child.sent == sentence:
            child_compound, _ = get_compound(child)
            results.append((child_compound, child.i))
    for child in token.children:
        if len(results) > 2:
            break
        if child.pos_ not in ['NOUN', 'PROPN'] and child != token and child.sent == sentence:
            results.extend(f(child, i + 1, exclude_tokens, neighbor_depth, sentence))
    return results

def f_adj(token, i, exclude_tokens, neighbor_depth, sentence):
    if i == neighbor_depth or token.children is None:
        return []
    results = []
    for child in token.children:
        if child.dep_ == "case" or child.text in stopwords:
            continue
        if child.pos_ == 'NOUN' and child not in exclude_tokens and child.sent == sentence:
            child_compound, _ = get_compound(child)
            results.append((child_compound, child.i))
    for child in token.children:
        if len(results) > 2:
            break
        if child.pos_ != 'NOUN' and child != token and child.sent == sentence:
            results.extend(f_adj(child, i + 1, exclude_tokens, neighbor_depth, sentence))
    return results

def g(token, exclude_tokens):
    current = token
    count = 0
    while True:
        next_token = current.head
        if next_token == current:
            break
        if next_token.pos_ in ['NOUN', 'PROPN'] and next_token not in exclude_tokens and next_token.sent == token.sent:
            next_compound, _ = get_compound(next_token)
            return next_compound, next_token.i
        current = next_token
        count += 1
        if count == 10:
            break
    return None, None

def g_adj(token, exclude_tokens, neighbor_depth):
    current = token
    count = 0
    while True:
        next_token = current.head
        if next_token == current:
            break
        if next_token.pos_ == 'NOUN' and next_token not in exclude_tokens and next_token.sent == token.sent:
            next_compound, _ = get_compound(next_token)
            return next_compound, next_token.i
        current = next_token
        count += 1
        if count == neighbor_depth:
            break
    return None, None

def find_noun_pairs(text, neighbor_depth=1):
    doc = nlp(text)
    pairs = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ['NOUN', 'PROPN'] and token.text not in stopwords:
                compound_token, components = get_compound(token)
                result = f(token, 0, components, neighbor_depth, sent)
                pairs.extend([(compound_token, token.i, r[0], r[1]) for r in result if r[0] != compound_token])
    final_noun_pairs = []
    for e1, p1, e2, p2 in pairs:
        if e1 not in stopwords and e2 not in stopwords:
            final_noun_pairs.append((e1, e2))
    return pairs


def find_adj_noun_pairs(text, neighbor_depth=3):
    doc = nlp(text)
    pairs = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == 'ADJ' and token.text not in stopwords:
                result = f_adj(token, 0, [], neighbor_depth, sent)
                head_noun, head_noun_i = g_adj(token, [], neighbor_depth)
                if head_noun is not None:
                    result.append((head_noun, head_noun_i))
                adj_compound, _ = get_compound(token)
                pairs.extend([(adj_compound, token.i, r[0], r[1]) for r in result if r[0] != adj_compound and nlp.vocab[r[0]].is_alpha])
    # 位置の近い形容詞のみを抽出
    closest_pairs = []
    for i, pair in enumerate(pairs):
        current_adj, adj_i, current_noun, noun_i = pair
        min_distance = float('inf')
        closest_pair = pair
        for j, other_pair in enumerate(pairs):
            if i != j and current_noun == other_pair[2] and doc[adj_i].sent == doc[other_pair[1]].sent:
                other_adj, other_adj_i, _, _ = other_pair
                distance = abs(adj_i - other_adj_i)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (other_adj, other_adj_i, current_noun, noun_i)
        closest_pairs.append(closest_pair)

    final_adj_pairs = []
    for e1, p1, e2, p2 in closest_pairs:
        if e1 not in stopwords and e2 not in stopwords:
            final_adj_pairs.append((e1, e2))
    return final_adj_pairs



def combine_pairs(noun_pairs, adj_pairs):
    combined = defaultdict(list)
    
    for pair in noun_pairs + adj_pairs:
        key = (pair[0], pair[1])
        value = (pair[2], pair[3])
        combined[key].append(value)
    
    result = []
    for key, values in combined.items():
        result.append((key[0], key[1], [v[0] for v in values], [v[1] for v in values]))
    
    return result

def process_sentence(sentence, adj_depth=1, noun_depth=1):
    noun_pairs = find_noun_pairs(sentence, noun_depth)
    adj_pairs = find_adj_noun_pairs(sentence, adj_depth)
    pairs = noun_pairs + adj_pairs
    final_adj_pairs = []
    for e1, p1, e2, p2 in adj_pairs:
        if e1 not in stopwords and e2 not in stopwords:
            final_adj_pairs.append((e1, e2))
    final_noun_pairs = []
    for e1, p1, e2, p2 in noun_pairs:
        if e1 not in stopwords and e2 not in stopwords:
            final_noun_pairs.append((e1, e2))
    return final_adj_pairs, final_noun_pairs
    final_pairs = []
    return noun_pairs + adj_pairs
    combined = combine_pairs(noun_pairs, adj_pairs)
    return combined

df_review = pd.read_pickle(
    "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
)
df_review_tmp = df_review
pandarallel.initialize(progress_bar=True)
df_review_tmp['pairs_adj'] = df_review_tmp['review'].parallel_apply(find_adj_noun_pairs)
df_review_tmp['pairs_noun'] = df_review_tmp['review'].parallel_apply(find_noun_pairs)
with open('../data/kg/pairs_adj.pkl', 'wb') as f:
    pickle.dump(df_review_tmp['pairs_adj'], f)

with open('../data/kg/pairs_noun.pkl', 'wb') as f:
    pickle.dump(df_review_tmp['pairs_noun'], f)

    