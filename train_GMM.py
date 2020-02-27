

from preprocessing import NPYDataSource,NPYDataSource2, KaldiSource, LTASSource
from nnmnkwii.datasets import FileSourceDataset
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm
import joblib
import os

from tDCF_python_v1.eval_metrics import compute_eer

np.random.seed(0)

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

    all_prediction = len(test_cancer_acoustic) + len(test_healthy_acoustic)
    right = 0

    cancer_scores = []
    healthy_scores = []
    for idx in tqdm(range(len(test_cancer_acoustic))):
        x = test_cancer_acoustic[idx].T
        score = np.mean(gmm_healthy.score(x) - gmm_cancer.score(x))
        cancer_scores.append(score)
        right += score < 0
    for idx in tqdm(range(len(test_healthy_acoustic))):
        x = test_healthy_acoustic[idx].T
        score = np.mean(gmm_healthy.score(x) - gmm_cancer.score(x))
        healthy_scores.append(score)
        right += score > 0

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


def gmm_ppg_script(experiment,gmm_comps,train):
    DATA_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_ppg/"
    gmm_dir = "/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/gmm/"
    #experiment = "gmm_ppg_debug"
    #train = False

    if train:
        cancer_train_source = NPYDataSource2(DATA_ROOT, subset="cancer")
        cancer_train = FileSourceDataset(cancer_train_source)
        healthy_train_source = NPYDataSource2(DATA_ROOT, subset="healthy")
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
        score = np.mean(gmm_healthy.score(x) - gmm_cancer.score(x))
        cancer_scores.append(score)
        #if score > 0:
            #print(cancer_test.collected_files[idx])
        right += score < 0
    for idx in tqdm(range(len(healthy_test))):
        x = healthy_test[idx].T
        score = np.mean(gmm_healthy.score(x) - gmm_cancer.score(x))
        healthy_scores.append(score)
        #if score < 0:
            #print(healthy_test.collected_files[idx])
        right += score > 0

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
    # plt.show()

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


    all_prediction = len(cancer_test) + len(healthy_test)
    right = 0

    cancer_scores = []
    healthy_scores = []
    for idx in tqdm(range(len(cancer_test))):
        x = cancer_test[idx].T
        score = np.mean(gmm_healthy.score(x) - gmm_cancer.score(x))
        cancer_scores.append(score)

        if score > 0:
            print(cancer_test.collected_files[idx])
        right += score < 0
    for idx in tqdm(range(len(healthy_test))):
        x = healthy_test[idx].T
        score = np.mean(gmm_healthy.score(x) - gmm_cancer.score(x))
        healthy_scores.append(score)

        if score < 0:
            print(healthy_test.collected_files[idx])
        right += score > 0

    accuracy = right / all_prediction
    print(accuracy)

    import matplotlib.pyplot as plt

    bins=100
    plt.subplot(1,2,1)
    plt.hist(healthy_scores,bins)
    plt.subplot(1,2,2)
    plt.hist(cancer_scores,bins)
    #plt.show()

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

    args = p.parse_args()


    train_scp_file = args.train_scp_file
    test_scp_file = args.test_scp_file
    train = args.train
    experiment = args.experiment
    gmm_comps = args.gmm_comps
    ltas = args.ltas
    ppg = args.ppg
    delta_delta = args.delta
    #train_scp_file = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp"
    #test_scp_file = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp"
    #train = False
    #gmm_comps = 16
    #gmm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,gmm_comps,ltas)

    if ppg:
        gmm_ppg_script(experiment,gmm_comps,train)
    else:
        gmm_kaldi_frontend(experiment,train,train_scp_file,test_scp_file,gmm_comps,ltas,delta_delta)
