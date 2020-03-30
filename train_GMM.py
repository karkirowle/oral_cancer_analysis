

from preprocessing import NPYDataSource,NPYDataSource2, KaldiSource, LTASSource
from nnmnkwii.datasets import FileSourceDataset
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm
import joblib
import os

from accuracy import calculate_acc_and_eer
from tDCF_python_v1.eval_metrics import compute_eer

np.random.seed(0)


class GMM_Wrapper:
    def __init__(self,gmm_healthy,gmm_cancer):
        self.gmm_healthy = gmm_healthy
        self.gmm_cancer = gmm_cancer

    def predict(self, x):
        return self.gmm_cancer.score(x) - self.gmm_healthy.score(x)

def gmm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,gmm_comps,ltas,delta_delta):

    gmm_dir = "gmm_checkpoints/"
    #gmm_dir = "/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/gmm/"

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

        num_sample = 200000


        x = train_cancer_acoustic[0]
        for idx in tqdm(range(1, len(train_cancer_acoustic))):
            x = np.hstack((x, train_cancer_acoustic[idx]))
        #print(x.shape[1])

        if (x.shape[1] > num_sample):
            idx = np.random.choice(x.shape[1],num_sample)
            x = x[:,idx]
        # TRAIN GMM
        #print(x.shape)
        gmm_cancer = GaussianMixture(n_components=gmm_comps, covariance_type="diag", verbose=0)
        gmm_cancer.fit(x.T)

        gmmpath = os.path.join(gmm_dir, experiment + "_cancer.pkl")
        joblib.dump(gmm_cancer, gmmpath)

        y = train_healthy_acoustic[0]
        for idx in tqdm(range(1, len(train_healthy_acoustic))):
            y = np.hstack((y, train_healthy_acoustic[idx]))

        if (y.shape[1] > num_sample):
            idx = np.random.choice(y.shape[1],num_sample)
            y = y[:,idx]


        gmm_healthy = GaussianMixture(n_components=gmm_comps, covariance_type="diag", verbose=0)
        gmm_healthy.fit(y.T)
        gmmpath = os.path.join(gmm_dir, experiment + "_healthy.pkl")
        joblib.dump(gmm_healthy, gmmpath)

    else:
        gmmpath = os.path.join(gmm_dir, experiment + "_healthy.pkl")
        gmm_healthy = joblib.load(gmmpath)
        gmmpath = os.path.join(gmm_dir, experiment + "_cancer.pkl")
        gmm_cancer = joblib.load(gmmpath)

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

    model = GMM_Wrapper(gmm_healthy, gmm_cancer)
    calculate_acc_and_eer(train_cancer_acoustic, train_healthy_acoustic, model, False, 0)
    print("",end="\t")
    calculate_acc_and_eer(test_cancer_acoustic, test_healthy_acoustic, model, False, 0)
    print("")


def gmm_ppg_script(experiment,gmm_comps,train,no_pause):

    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_ppg_asr"
    #DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_ppg/"
    gmm_dir = "gmm_checkpoints/"
    #experiment = "gmm_ppg_debug"
    #train = False

    if train:
        cancer_train_source = NPYDataSource2(DATA_ROOT, subset="cancer",transpose=True)
        cancer_train = FileSourceDataset(cancer_train_source)
        healthy_train_source = NPYDataSource2(DATA_ROOT, subset="healthy",transpose=True)
        healthy_train = FileSourceDataset(healthy_train_source)
        #print("Acoustic linguistic feature dim", cancer_train[0].shape[-1])
        #print(len(cancer_train))


        x = cancer_train[0]
        for idx in tqdm(range(1, len(cancer_train))):
            x = np.hstack((x, cancer_train[idx]))
        # TRAIN GMM
        #print(x.shape)
        num_sample = 200000
        if (x.shape[1] > num_sample):
            idx = np.random.choice(x.shape[1],num_sample)
            x = x[:,idx]

        if no_pause:
            x = x[:-1,:]
        gmm_cancer = GaussianMixture(n_components=gmm_comps, covariance_type="diag")
        gmm_cancer.fit(x.T)

        gmmpath = os.path.join(gmm_dir, experiment + "_cancer.pkl")
        joblib.dump(gmm_cancer, gmmpath)

        y = healthy_train[0]
        for idx in tqdm(range(1, len(healthy_train))):
            y = np.hstack((y, healthy_train[idx]))

        if (y.shape[1] > num_sample):
            idx = np.random.choice(y.shape[1], num_sample)
            y = y[:, idx]

        if no_pause:
            y = y[:-1, :]
        gmm_healthy = GaussianMixture(n_components=gmm_comps, covariance_type="diag")
        gmm_healthy.fit(y.T)
        gmmpath = os.path.join(gmm_dir, experiment + "_healthy.pkl")
        joblib.dump(gmm_healthy, gmmpath)
    else:
        gmmpath = os.path.join(gmm_dir, experiment + "_healthy.pkl")
        gmm_healthy = joblib.load(gmmpath)
        gmmpath = os.path.join(gmm_dir, experiment + "_cancer.pkl")
        gmm_cancer = joblib.load(gmmpath)

    # EVAL GMM
    #DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_ppg_2"

    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_ppg_asr/"

    cancer_test_source = NPYDataSource2(DATA_ROOT, subset="cancer",transpose=True)
    cancer_test = FileSourceDataset(cancer_test_source)
    healthy_test_source = NPYDataSource2(DATA_ROOT, subset="healthy",transpose=True)
    healthy_test = FileSourceDataset(healthy_test_source)

    model = GMM_Wrapper(gmm_healthy,gmm_cancer)
    calculate_acc_and_eer(cancer_train, healthy_train, model, no_pause, 0)
    print("",end="\t")
    calculate_acc_and_eer(cancer_test, healthy_test, model, no_pause, 0)
    print("")

def gmm_mfcc_script():
    DATA_ROOT = "/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/wav/"
    gmm_dir = "/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/gmm/"
    experiment = "gmm_debug"
    train = False

    if train:
        cancer_train_source = NPYDataSource(DATA_ROOT, subset="cancer",train=True)
        cancer_train = FileSourceDataset(cancer_train_source)
        healthy_train_source = NPYDataSource(DATA_ROOT,subset="healthy",train=True)
        healthy_train = FileSourceDataset(healthy_train_source)
        print("Acoustic linguistic feature dim", cancer_train[0].shape[-1])
        print(len(cancer_train))

        x = cancer_train[0]
        for idx in tqdm(range(1, len(cancer_train))):
            x = np.hstack((x, cancer_train[idx]))
        # TRAIN GMM
        print(x.shape)
        gmm_cancer = GaussianMixture(n_components=512,covariance_type="diag")
        gmm_cancer.fit(x.T)

        gmmpath = os.path.join(gmm_dir, experiment + "_cancer.pkl")
        joblib.dump(gmm_cancer, gmmpath)

        y = healthy_train[0]
        for idx in tqdm(range(1, len(healthy_train))):
            y = np.hstack((y, healthy_train[idx]))
        gmm_healthy = GaussianMixture(n_components=512,covariance_type="diag")
        gmm_healthy.fit(y.T)
        gmmpath = os.path.join(gmm_dir, experiment + "_healthy.pkl")
        joblib.dump(gmm_healthy, gmmpath)
    else:
        gmmpath = os.path.join(gmm_dir, experiment + "_healthy.pkl")
        gmm_healthy = joblib.load(gmmpath)
        gmmpath = os.path.join(gmm_dir, experiment + "_cancer.pkl")
        gmm_cancer = joblib.load(gmmpath)


    # EVAL GMM
    cancer_test_source = NPYDataSource(DATA_ROOT, subset="cancer",train=False)
    cancer_test = FileSourceDataset(cancer_test_source)
    healthy_test_source = NPYDataSource(DATA_ROOT,subset="healthy",train=False)
    healthy_test = FileSourceDataset(healthy_test_source)


    model = GMM_Wrapper(gmm_healthy, gmm_cancer)
    calculate_acc_and_eer(cancer_train, healthy_train, model, no_pause, 0)
    print("",end="\t")
    calculate_acc_and_eer(cancer_test, healthy_test, model, no_pause, 0)
    print("")

if __name__ == '__main__':

    import configargparse

    p = configargparse.ArgParser()

    # Configuration strings


    p.add('--train_scp_file', required=False, help='Training protocol file path')
    p.add('--test_scp_file', required=False, help='Dev protocol file path')
    p.add('--experiment', required=False, help='Evaluation protocol file path')
    p.add('--gmm_comps', type=int, default=16, help="Number of GMM comps")

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
    gmm_comps = args.gmm_comps
    ltas = args.ltas
    ppg = args.ppg
    delta_delta = args.delta
    no_pause = args.no_pause
    #train_scp_file = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp"
    #test_scp_file = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp"
    #train = False
    #gmm_comps = 16
    #gmm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,gmm_comps,ltas)

    if ppg:
        gmm_ppg_script(experiment,gmm_comps,train,no_pause)
    else:
        gmm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,gmm_comps,ltas,delta_delta)
