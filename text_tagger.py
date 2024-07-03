"""
Viib Limesurvey tekstid Stanzaga sobivasse vormi.
Eesmärk: tunnuste eraldamiseks on vaja, et sisend oleks sobival kujul.
Väljund -> üks tekst per rida -> [faili_id, [[sõne1, sõne1_lemma, sõne1_ POS, sõne_morf], [repeat], ...]]
"""
from collections import Counter
from pathlib import Path
import stanza
import jsonlines
import torch
torch.set_num_threads(2)
nlp = stanza.Pipeline('et', verbose=False, use_gpu=False)

lemma_counter_general = Counter()

def get_sent_len(txt):
    sentences = [sent for sent in txt.sentences]
    doc_tokens = []
    for i, sent in enumerate(txt.sentences, 1):
        sent_tokens = len([token.text for token in sent.tokens])
        doc_tokens.append(sent_tokens)
    average_sent_len = sum(doc_tokens)/len(sentences)

    return round(average_sent_len, 2)


def analyysi(doc):
    doc_lst = []
    txt = nlp(doc)
    avg_sent_len = get_sent_len(txt)
    wordinfo_counter = 0
    lemma_counter = Counter()
    for sent in txt.sentences:
        sent_words = [w for w in sent.words]
        for w in sent_words:
            token = w.text
            lemma = w.lemma
            lemma_counter[lemma] += 1
            lemma_counter_general[lemma] += 1
            pos = w.upos
            dep = w.deprel
            if w.feats is None:
                morf = '_'
            else:
                morf = w.feats

            word_info = [token, lemma, pos, morf, dep]
            wordinfo_counter += 1

            doc_lst.append(word_info)
    return avg_sent_len, doc_lst, wordinfo_counter, lemma_counter


def main():
    dir = 'data' # dir with files
    output_f = 'limesrurvey_tekstid_morfiga_vol_X' # name the output file
    paths = [p for p in Path(dir).glob('*.txt')]
    with jsonlines.open(output_f, mode='w') as writer:
        for path in paths:
            filename = path.stem
            with open(path) as fod:
                doc = ' '.join([line for line in fod.readlines()])
                analyys = analyysi(doc)
                print(filename, analyys[0], analyys[2], len(analyys[3]))#, analyys[3])
                # per rida: failinimi, keskmine lausete pikkus, sõnede arv, sõnede analüüsid, lemmade counter, lemmade arv
                # writer.write([filename, analyys[0], analyys[2], analyys[1], analyys[3], len(analyys[3])])

    # print(lemma_counter_general)


if __name__ == '__main__':
    # main()
    print('Boo')
