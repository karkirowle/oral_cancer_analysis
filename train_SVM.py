

from preprocessing import NPYDataSource,NPYDataSource2, KaldiSource, LTASSource, combine_stack_and_label
from nnmnkwii.datasets import FileSourceDataset
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm
import joblib
import os

from tDCF_python_v1.eval_metrics import compute_eer
from sklearn.svm import SVC

np.random.seed(0)

def svm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,ltas,delta_delta):

    svm_dir = "svm_checkpoints/"
    num_samples = 30000
    if train:

        scp_file = train_scp_file

        if ltas:
            train_cancer_acoustic_source = LTASSource(scp_file, subset="cancer")
            train_healthy_acoustic_source = LTASSource(scp_file, subset="healthy")
        else:
            train_cancer_acoustic_source = KaldiSource(scp_file, subset="cancer",delta_delta=delta_delta)
            train_healthy_acoustic_source = KaldiSource(scp_file, subset="healthy",delta_delta=delta_delta)

        train_cancer_acoustic = FileSourceDataset(train_cancer_acoustic_source)
        train_healthy_acoustic = FileSourceDataset(train_healthy_acoustic_source)

        X, Y = combine_stack_and_label(train_cancer_acoustic, train_healthy_acoustic,num_samples)

        svm = SVC(gamma='auto')

        print(X.shape)
        print(Y.shape)
        svm.fit(X.T,Y)

        svmpath = os.path.join(svm_dir, experiment + ".pkl")
        joblib.dump(svm, svmpath)

    else:
        svmpath = os.path.join(svm_dir, experiment + ".pkl")
        svm = joblib.load(svmpath)

    # EVAL GMM
    scp_file = test_scp_file

    if ltas:
        test_cancer_acoustic_source = LTASSource(scp_file, subset="cancer")
        test_healthy_acoustic_source = LTASSource(scp_file, subset="healthy")
    else:
        test_cancer_acoustic_source = KaldiSource(scp_file, subset="cancer",delta_delta=delta_delta)
        test_healthy_acoustic_source = KaldiSource(scp_file, subset="healthy",delta_delta=delta_delta)

    test_cancer_acoustic = FileSourceDataset(test_cancer_acoustic_source)
    test_healthy_acoustic = FileSourceDataset(test_healthy_acoustic_source)

    #test_set, test_labels = combine_stack_and_label(test_cancer_acoustic,test_healthy_acoustic,num_samples)
    #accuracy = svm.score(test_set.T,test_labels)

    all_prediction = len(test_cancer_acoustic) + len(test_healthy_acoustic)
    right = 0
    cancer_scores = []
    healthy_scores = []
    for idx in tqdm(range(len(test_cancer_acoustic))):
        x = test_cancer_acoustic[idx].T
        score = np.mean(svm.predict(x))
        cancer_scores.append(score)
        right += score >= 0.5
    for idx in tqdm(range(len(test_healthy_acoustic))):
        x = test_healthy_acoustic[idx].T
        score = np.mean(svm.predict(x))
        healthy_scores.append(score)
        right += score < 0.5

    accuracy = right / all_prediction
    print(accuracy,end="\t")


    eer, _ = compute_eer(np.array(healthy_scores), np.array(cancer_scores))
    print(eer)

    import matplotlib.pyplot as plt

    # bins = 100
    # plt.subplot(1, 2, 1)
    # plt.hist(healthy_scores, bins)
    # plt.subplot(1, 2, 2)
    # plt.hist(cancer_scores, bins)
    #plt.show()


def svm_ppg_script(experiment,train,no_pause):
    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_ppg/"
    svm_dir = "/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/gmm/"

    num_samples = 20000
    if train:
        cancer_train_source = NPYDataSource2(DATA_ROOT, subset="cancer")
        cancer_train = FileSourceDataset(cancer_train_source)
        healthy_train_source = NPYDataSource2(DATA_ROOT, subset="healthy")
        healthy_train = FileSourceDataset(healthy_train_source)


        X, Y = combine_stack_and_label(cancer_train, healthy_train, num_samples)

        svm = SVC(gamma='auto')

        print(X.shape)
        print(Y.shape)

        if no_pause:
            X = X[:-1,:]

        svm.fit(X.T, Y)

        svmpath = os.path.join(svm_dir, experiment + ".pkl")
        joblib.dump(svm, svmpath)

    else:
        svmpath = os.path.join(svm_dir, experiment + ".pkl")
        svm = joblib.load(svmpath)


    # EVAL GMM
    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_ppg/"

    cancer_test_source = NPYDataSource2(DATA_ROOT, subset="cancer")
    cancer_test = FileSourceDataset(cancer_test_source)
    healthy_test_source = NPYDataSource2(DATA_ROOT, subset="healthy")
    healthy_test = FileSourceDataset(healthy_test_source)

    all_prediction = len(cancer_test) + len(healthy_test)
    right = 0

    cancer_scores = []
    healthy_scores = []

    for idx in tqdm(range(len(cancer_test))):
        x = cancer_test[idx].T
        score = np.mean(svm.predict(x))
        cancer_scores.append(score)
        right += score >= 0.5
    for idx in tqdm(range(len(healthy_test))):
        x = healthy_test[idx].T
        score = np.mean(svm.predict(x))
        healthy_scores.append(score)
        right += score < 0.5

    accuracy = right / all_prediction
    print(accuracy,end="\t")


    eer, _ = compute_eer(np.array(cancer_scores), np.array(healthy_scores))
    print(eer)


if __name__ == '__main__':

    import configargparse

    p = configargparse.ArgParser()

    # Configuration strings


    p.add('--train_scp_file', required=False, help='Training protocol file path')
    p.add('--test_scp_file', required=False, help='Dev protocol file path')
    p.add('--experiment', required=False, help='Evaluation protocol file path')

    p.add("--train", action="store_true", help='Train or just reproduce')
    p.add("--ltas", action="store_true", help='Prosodic features')
    p.add("--ppg", action="store_true", help='PPG-based features')
    p.add("--delta", action="store_true", help="Delta-delta features for MFCC and PLP")
    p.add("--no_pause", action="store_true", help="Omitting pause features from linguistic feat analysis")

    args = p.parse_args()


    train_scp_file = args.train_scp_file
    test_scp_file = args.test_scp_file
    train = args.train
    experiment = args.experiment
    ltas = args.ltas
    ppg = args.ppg
    delta_delta = args.delta
    no_pause = args.no_pause


    if ppg:
        svm_ppg_script(experiment, train, no_pause)
    else:
        svm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,ltas,delta_delta)
