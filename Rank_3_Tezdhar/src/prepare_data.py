import csv
import string
import unicodedata
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize import sent_tokenize

from src.config import TRAIN_FILE, TEST_FILE, DATA_PATH, N_FOLDS, SEED, TARGET

all_letters = string.ascii_letters + " .,;'?!@#$="

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def get_drug_only_sentence(row):
    sentences = sent_tokenize(row["text"])
    if len(sentences) == 0:
        print("no data")
        return "neutral"
    idx = [i for i, sent in enumerate(sentences) if row["drug"].lower() in sent.lower()]
    if len(idx) == 0:
        print("No drug mention")
        return unicodeToAscii(" ".join(sentences))
    else:
        idx = idx[0]
    n = len(sentences)
    drug_sents = []
    if idx > 0:
        drug_sents.append(sentences[idx-1][-100:])
    drug_sents.append(sentences[idx])
    if idx < n-1:
        drug_sents.append(sentences[idx+1])
    if idx < n-2:
        drug_sents.append(sentences[idx+2])
    if idx < n-3:
        drug_sents.append(sentences[idx+3])
    return unicodeToAscii(" ".join(drug_sents))

def reframe_drug_question(drug):
    return "What do you think of {}?".format(drug)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE, encoding='utf-8')
    test = pd.read_csv(TEST_FILE, encoding='utf-8')
    train["drug_text"] = train.apply(get_drug_only_sentence, axis=1).astype(str)
    test["drug_text"] = test.apply(get_drug_only_sentence, axis=1).astype(str)

    train["drug"] = train["drug"].apply(reframe_drug_question)
    test["drug"] = test["drug"].apply(reframe_drug_question)

    cols  = ["unique_hash", "text", "drug_text", "drug", "sentiment"]
    cvlist = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, train[TARGET]))
    for fold in range(N_FOLDS):
        save_dir = Path(DATA_PATH) / "v2" / "fold_{}".format(fold)
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        tr_idx, val_idx = cvlist[fold]
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]
        tr[cols].to_csv(str(save_dir / "train.tsv"), index=False, encoding="utf-8")
        val[cols].to_csv(str(save_dir / "dev.tsv"), index=False, encoding="utf-8")
        test["sentiment"] = 2
        test[cols].to_csv(str(save_dir / "test.tsv"), index=False, encoding="utf-8")

