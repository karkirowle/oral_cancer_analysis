

from preprocessing import NPYDataSource2, KaldiSource, LTASSource, combine_stack_and_label
from nnmnkwii.datasets import FileSourceDataset
import numpy as np
import joblib
import os

from accuracy import calculate_acc_and_eer

from sklearn.linear_model import Lasso
np.random.seed(0)

def lasso_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,ltas,delta_delta,alpha):

    lasso_dir = "svm_checkpoints/"
    num_samples = 2000000
    if train:

        scp_file = train_scp_file

        if ltas:
            train_cancer_acoustic_source = LTASSource(scp_file, subset="cancer")
            train_healthy_acoustic_source = LTASSource(scp_file, subset="healthy")
        else:
            train_cancer_acoustic_source = KaldiSource(scp_file, subset="cancer",delta_delta=delta_delta,normalise=True)
            train_healthy_acoustic_source = KaldiSource(scp_file, subset="healthy",delta_delta=delta_delta,normalise=True)

        train_cancer_acoustic = FileSourceDataset(train_cancer_acoustic_source)
        train_healthy_acoustic = FileSourceDataset(train_healthy_acoustic_source)

        X, Y = combine_stack_and_label(train_cancer_acoustic, train_healthy_acoustic,num_samples)

        lasso = Lasso(alpha)

        
        lasso.fit(X.T,Y)

        lassopath = os.path.join(lasso_dir, experiment + ".pkl")
        joblib.dump(lasso, lassopath)

    else:
        lassopath = os.path.join(lasso_dir, experiment + ".pkl")
        lasso = joblib.load(lassopath)

    # EVAL GMM
    scp_file = test_scp_file

    if ltas:
        test_cancer_acoustic_source = LTASSource(scp_file, subset="cancer")
        test_healthy_acoustic_source = LTASSource(scp_file, subset="healthy")
    else:
        test_cancer_acoustic_source = KaldiSource(scp_file, subset="cancer",delta_delta=delta_delta,normalise=True)
        test_healthy_acoustic_source = KaldiSource(scp_file, subset="healthy",delta_delta=delta_delta,normalise=True)

    test_cancer_acoustic = FileSourceDataset(test_cancer_acoustic_source)
    test_healthy_acoustic = FileSourceDataset(test_healthy_acoustic_source)

    calculate_acc_and_eer(train_cancer_acoustic, train_healthy_acoustic,lasso, False)
    print("",end="\t")
    calculate_acc_and_eer(test_cancer_acoustic, test_healthy_acoustic,lasso, False)
    print("")

def lasso_ppg_script(experiment: str,train: bool,no_pause: bool,alpha: np.float32):
    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_ppg_asr/"
    svm_dir = "svm_checkpoints/"
    num_samples = 2000000
    if train:
        cancer_train_source = NPYDataSource2(DATA_ROOT, subset="cancer",transpose=True)
        cancer_train = FileSourceDataset(cancer_train_source)
        healthy_train_source = NPYDataSource2(DATA_ROOT, subset="healthy",transpose=True)
        healthy_train = FileSourceDataset(healthy_train_source)


        X, Y = combine_stack_and_label(cancer_train, healthy_train, num_samples)
        lasso = Lasso(alpha)

        if no_pause:
            X = X[:-1,:]
        lasso.fit(X.T, Y)
        
        ardpath = os.path.join(svm_dir, experiment + ".pkl")
        joblib.dump(lasso, ardpath)

    else:
        ardpath = os.path.join(svm_dir, experiment + ".pkl")
        lasso = joblib.load(ardpath)


    # EVAL GMM
    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_ppg_asr/"

    cancer_test_source = NPYDataSource2(DATA_ROOT, subset="cancer",transpose=True)
    cancer_test = FileSourceDataset(cancer_test_source)
    healthy_test_source = NPYDataSource2(DATA_ROOT, subset="healthy",transpose=True)
    healthy_test = FileSourceDataset(healthy_test_source)

    calculate_acc_and_eer(cancer_train, healthy_train, lasso, no_pause)
    print("",end="\t")
    calculate_acc_and_eer(cancer_test, healthy_test, lasso, no_pause)
    print("")


if __name__ == '__main__':

    import configargparse

    p = configargparse.ArgParser()

    # Configuration strings


    p.add('--train_scp_file', required=False, help='Training protocol file path')
    p.add('--test_scp_file', required=False, help='Dev protocol file path')
    p.add('--experiment', required=False, help='Evaluation protocol file path')
    p.add("--alpha", help='Lasso Alpha')

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
    alpha = np.float32(args.alpha)
    if ppg:
        lasso_ppg_script(experiment, train, no_pause,alpha)
    else:
        lasso_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,ltas,delta_delta,alpha)
