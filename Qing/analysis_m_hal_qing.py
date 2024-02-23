import pickle
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import os
import json
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv


def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled

def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    indices = [k for k in human_labels.keys()]
    unroll_preds = unroll_pred(preds, indices)
    if oneminus_pred:
        unroll_preds = [1.0-x for x in unroll_preds]
    unroll_labels = unroll_pred(human_labels, indices)
    assert len(unroll_preds) == len(unroll_labels)
    print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    return P, R

def print_AUC(P, R):
    print("AUC: {:.2f}".format(auc(R, P)*100))



if __name__ == "__main__":
    human_label_detect_False = {}
    human_label_detect_True = {}

    path = f"result/answer_synthetic_val_data_from_M_HalDetect.bin"
    with open(path, "rb") as f:
        responses = pickle.load(f)

    average_logprob_scores = {}  # average_logprob
    average_entropy_scores = {}  # lowest_logprob
    lowest_logprob_scores = {}  # average_entropy5
    highest_entropy_scores = {}  # highest_entropy5

    for idx, response in responses.items():
        question_id = response["question_id"]
        log_probs = response["logprobs"]
        combined_token_logprobs = log_probs["combined_token_logprobs"]
        combined_token_entropies = log_probs["combined_token_entropies"]
        labels = response["labels"]
        assert len(combined_token_logprobs) == len(combined_token_entropies) == len(labels) == len(response["sentences"]), "Unmatched numbers sentences."
        sentences_len = len(response["sentences"])

        average_logprob_sent_level = [None for _ in range(sentences_len)]
        lowest_logprob_sent_level = [None for _ in range(sentences_len)]
        average_entropy_sent_level = [None for _ in range(sentences_len)]
        highest_entropy_sent_level = [None for _ in range(sentences_len)]

        label_True_sent_level = [None for _ in range(sentences_len)]
        label_False_sent_level = [None for _ in range(sentences_len)]

        for i in range(sentences_len):
            sentence_log_probs = combined_token_logprobs[i]
            sentence_entropies = combined_token_entropies[i]
            label = labels[i]
            average_logprob = np.mean(sentence_log_probs)
            lowest_logprob = np.min(sentence_log_probs)
            average_entropy = np.mean(sentence_entropies)
            highest_entropy = np.max(sentence_entropies)

            average_logprob_sent_level[i] = average_logprob
            lowest_logprob_sent_level[i] = lowest_logprob
            average_entropy_sent_level[i] = average_entropy
            highest_entropy_sent_level[i] = highest_entropy

            if label == 'ACCURATE' or label == 'ANALYSIS':
                true_score = 1.0
                false_score = 0.0
            elif label == 'INACCURATE':
                true_score = 0.0
                false_score = 1.0

            label_True_sent_level[i] = true_score
            label_False_sent_level[i] =false_score


        average_logprob_scores[question_id] = average_logprob_sent_level
        lowest_logprob_scores[question_id] = lowest_logprob_sent_level
        average_entropy_scores[question_id] = average_entropy_sent_level
        highest_entropy_scores[question_id] = highest_entropy_sent_level
        human_label_detect_True[question_id] = label_True_sent_level
        human_label_detect_False[question_id] = label_False_sent_level

    # analysis true ratio:
    true_values = []
    for key, value in human_label_detect_True.items():
        true_values.extend(value)
    total_num = len(true_values)
    print("The total number of sentences is: {}; The ratio of true values is: {}".format(total_num, true_values.count(1.0) / total_num))

    # True
    # uncertainty
    Pb_average_logprob, Rb_average_logprob = get_PR_with_human_labels(average_logprob_scores,
                                                                      human_label_detect_True, pos_label=1,
                                                                      oneminus_pred=False)
    Pb_average_entropy, Rb_average_entropy = get_PR_with_human_labels(average_entropy_scores,
                                                                        human_label_detect_True, pos_label=1,
                                                                        oneminus_pred=True)
    Pb_lowest_logprob, Rb_lowest_logprob = get_PR_with_human_labels(lowest_logprob_scores, human_label_detect_True,
                                                                    pos_label=1, oneminus_pred=False)
    Pb_highest_entropy, Rb_highest_entropy = get_PR_with_human_labels(highest_entropy_scores,
                                                                        human_label_detect_True, pos_label=1,
                                                                        oneminus_pred=True)

    print("-----------------------")
    print("Baseline1: Avg(logP)")
    print_AUC(Pb_average_logprob, Rb_average_logprob)
    print("-----------------------")
    print("Baseline2: Avg(H)")
    print_AUC(Pb_average_entropy, Rb_average_entropy)
    print("-----------------------")
    print("Baseline3: Max(-logP)")
    print_AUC(Pb_lowest_logprob, Rb_lowest_logprob)
    print("-----------------------")
    print("Baseline4: Max(H)")
    print_AUC(Pb_highest_entropy, Rb_highest_entropy)

    arr = []
    for v in human_label_detect_True.values():
        arr.extend(v)
    random_baseline = np.mean(arr)

    # with human label, Detecting Non-factual*
    plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_average_logprob, Pb_average_logprob, '-', label='Avg(logP)')
    plt.plot(Rb_average_entropy, Pb_average_entropy, '-', label='Avg(H)')
    plt.plot(Rb_lowest_logprob, Pb_lowest_logprob, '-', label='Max(-logP)')
    plt.plot(Rb_highest_entropy, Pb_highest_entropy, '-', label='Max(H)')
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()