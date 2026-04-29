import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    acc2 = accuracy_score(binary_truth, binary_preds)

    print("MAE: ", mae)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc2)
    return acc2


def count_parameters(model):
    trainable_params = 0
    total_params = 0
    trainable_params_list = list()
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_list.append(name)
    print(f"Total params: {total_params}, Trainable params: {trainable_params}")
