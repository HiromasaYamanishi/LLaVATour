import spacy
from tqdm import tqdm

class GinzaTokenizer:
    def __init__(self, model_name='ja_ginza_electra'):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        spacy.load('ja_ginza_electra')
        
    def tokenize(self, sentences, target_pos=['PROP','PROPN', 'NOUN', 'ADJ', 'VERB']):
        outputs, outputs_propn = [], []
        for sentence in tqdm(sentences):
            out, out_propn = [], []
            doc = self.nlp(sentence)
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ in target_pos:
                        if not len(token.lemma_):continue
                        out.append(token.lemma_)
                    if token.pos_ == 'PROPN':
                        out_propn.append(token.lemma_)
                    #print(token.lemma_, token.pos_)
                    # print(
                    #     token.i,
                    #     token.orth_,
                    #     token.lemma_,
                    #     token.norm_,
                    #     token.morph.get("Reading"),
                    #     token.pos_,
                    #     token.morph.get("Inflection"),
                    #     token.tag_,
                    #     token.dep_,
                    #     token.head.i,
                    # )
                    #out.append(token.lemma_)
            outputs.append(out)
            outputs_propn.append(out_propn)
            
        return outputs, outputs_propn