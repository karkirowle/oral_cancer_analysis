
from tqdm import tqdm
import numpy as np
from tDCF_python_v1.eval_metrics import compute_eer

def calculate_acc_and_eer(test_cancer_acoustic, test_healthy_acoustic, model,no_pause,th=0.5):

    all_prediction = len(test_cancer_acoustic) + len(test_healthy_acoustic)
    right = 0
    cancer_scores = []
    healthy_scores = []
    for idx in tqdm(range(len(test_cancer_acoustic))):
        x = test_cancer_acoustic[idx].T
        if no_pause:
            x = x[:,:-1]
        score = np.mean(model.predict(x))
        cancer_scores.append(score)
        right += score >= th
    for idx in tqdm(range(len(test_healthy_acoustic))):
        x = test_healthy_acoustic[idx].T
        if no_pause:
            x = x[:,:-1]
        score = np.mean(model.predict(x))
        healthy_scores.append(score)
        right += score < th

    accuracy = np.round(right / all_prediction, decimals=4)
    print(accuracy, end="\t")

    eer, _ = np.round(compute_eer(np.array(cancer_scores), np.array(healthy_scores)),decimals=4)
    print(eer, end="")