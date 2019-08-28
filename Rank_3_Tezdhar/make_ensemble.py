
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np
from copy import copy

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gmean

from src.config import SEED, TRAIN_FILE, TEST_FILE


def load_partial_numpy_arrays(model_path, k=1):
    test_preds = []
    val_preds = []
    y_partial = []
    for i, (tr_idx, val_idx) in enumerate(cvlist[:k]):
        vp = np.load(model_path+"/fold_{}/val_preds.npy".format(i))
        tp = np.load(model_path+"/fold_{}/text_preds.npy".format(i))
        print(vp.shape, tp.shape, y[val_idx].shape)
        val_preds.extend(vp)
        y_partial.extend(y[val_idx])
        test_preds.append(tp)
    test_preds = np.mean(test_preds, 0)
    return np.array(val_preds), test_preds, np.array(y_partial)


if  __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)


    y = train.sentiment.values
    cvlist = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, train["sentiment"]))

    b2_val_preds, b2_test_preds, b2_y = load_partial_numpy_arrays("models/bert_v3", k=3)
    print("Bert validation score", f1_score(b2_y, np.argmax(b2_val_preds, 1), average='macro'))

    x1_val_preds, x1_test_preds, x1_y = load_partial_numpy_arrays("models/xlnet_v2", k=3)
    print("xlnet v1 validation score", f1_score(x1_y, np.argmax(x1_val_preds, 1), average='macro'))
    
    x2_val_preds, x2_test_preds, x2_y = load_partial_numpy_arrays("models/xlnet_v3", k=3)

    avg_val_preds = 0.35*x2_val_preds + 0.3*b2_val_preds + 0.35*x1_val_preds
    avg_test_preds = 0.35*x2_test_preds + 0.3*b2_test_preds + 0.35*x1_test_preds

    print("Final validation score ": f1_score(x2_y, np.argmax(avg_val_preds, 1), average='macro'))

    sub = test[["unique_hash"]]
    sub["sentiment"] = np.argmax(avg_test_preds, 1)
    sub.to_csv("data/sub_e3.csv", index=False)
    sub.sentiment.value_counts(normalize=True)

