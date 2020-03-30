


import matplotlib.pyplot as plt
import matplotlib
import joblib
import numpy as np
import pandas as pd


from preprocessing import FileSourceDataset, LibrosaSpectrogramSource, LogspecLoader, LabelDataSource, KaldiSource, KaldiLabelDataSource
from nnmnkwii.datasets import PaddedFileSourceDataset
from vis.visualization import visualize_cam
from vis.utils import utils
import tqdm
from keras import activations
import model.resnet
from keras import optimizers

import tensorflow as tf
import keras.backend as K

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def ltas_lasso_interpreter(model):


    lasso = joblib.load(model)
    M = len(lasso.coef_)
    idx = np.arange(len(lasso.coef_))
    print(idx[abs(lasso.coef_) > 0])

    duration = 1000
    heatmap = np.zeros((1000,M))

    mean_feat = lasso.coef_[:257]
    std_feat = lasso.coef_[257:]
    heatmap = np.tile(mean_feat,(1000,1)).T
    heatmap_2 = np.tile(std_feat,(1000,1)).T
    y = np.linspace(0, 8000, 257)

    font = {'size': 22}

    matplotlib.rc('font', **font)

    fig = plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(y,mean_feat,linewidth=4)
    plt.plot(y,std_feat,linewidth=4)
    plt.ylabel("coefficient for frequency bin",fontsize=32)
    plt.xlabel("frequency (Hz)",fontsize=32)
    plt.legend(["mean LTAS features","std LTAS features"],fontsize=32)
    #fancy_spectrogram(heatmap_2,"control",cmap="gray")
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig("figures/ltas_sparsity.pdf",padlen=0.005)

    #plt.show()
def confusion_dnn(calculate):
    if calculate:
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(graph=tf.get_default_graph(),
                          config=config)
        K.set_session(sess)
        learning_rate = 0.001
        pad_len = 4000
        resnet = model.resnet.ResNet_Logspec(pad_len, 257)

        optimiser = optimizers.Adam(lr=learning_rate)
        resnet.trainer.compile(optimiser, loss='categorical_crossentropy',
                               weighted_metrics=["accuracy"],
                               metrics=['accuracy'],
                               sample_weight_mode="None")

        experiment = "masked_100_frames_50_earlystopped_3"
        resnet.trainer.load_weights("checkpoints/" + experiment + ".hd5")

        test_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp"

        cancer_list =  ["id001", "id005", "id012","id013","id006","id008"]
        healthy_list = ["id010013","id10094","id10110","id10509","id10855","id11217"]

        print("Test set")
        for speaker in cancer_list:
            test_acoustic_source = KaldiSource(test_ROOT, subset=speaker, transpose=False)
            test_acoustic = PaddedFileSourceDataset(test_acoustic_source, pad_len)
            test_label = np.zeros((len(test_acoustic),2))
            test_label[:,1] = 1
            shuffle = False
            batch_size = 1
            val_gen = LogspecLoader(test_acoustic, test_label, shuffle, batch_size)

            results = resnet.trainer.evaluate_generator(val_gen)
            print(speaker,end="\t")
            print(results[1])
        for speaker in healthy_list:
            test_acoustic_source = KaldiSource(test_ROOT, subset=speaker, transpose=False)
            test_acoustic = PaddedFileSourceDataset(test_acoustic_source, pad_len)
            test_label = np.zeros((len(test_acoustic),2))
            test_label[:,0] = 1
            shuffle = False
            batch_size = 1
            val_gen = LogspecLoader(test_acoustic, test_label, shuffle, batch_size)

            results = resnet.trainer.evaluate_generator(val_gen)
            print(speaker, end="\t")
            print(results[1])

        train_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp"

        cancer_list = ["id002", "id003", "id004", "id007", "id011"]
        healthy_list = ["id10078","id100111", "id10242", "id10571", "id11250"]

        print("Train set")
        for speaker in cancer_list:
            test_acoustic_source = KaldiSource(train_ROOT, subset=speaker, transpose=False)
            test_acoustic = PaddedFileSourceDataset(test_acoustic_source, pad_len)
            test_label = np.zeros((len(test_acoustic),2))
            test_label[:,1] = 1
            shuffle = False
            batch_size = 1
            val_gen = LogspecLoader(test_acoustic, test_label, shuffle, batch_size)

            results = resnet.trainer.evaluate_generator(val_gen)
            print(speaker, end="\t")
            print(results[1])
        for speaker in healthy_list:
            test_acoustic_source = KaldiSource(train_ROOT, subset=speaker, transpose=False)
            test_acoustic = PaddedFileSourceDataset(test_acoustic_source, pad_len)
            test_label = np.zeros((len(test_acoustic),2))
            test_label[:,0] = 1
            shuffle = False
            batch_size = 1
            val_gen = LogspecLoader(test_acoustic, test_label, shuffle, batch_size)

            results = resnet.trainer.evaluate_generator(val_gen)
            print(speaker, end="\t")
            print(results[1])
def phonet_gmm_figure(model_cancer, model_control):
    font = {'size': 22}

    matplotlib.rc('font', **font)

    cancer_gmm = joblib.load(model_cancer)
    control_gmm = joblib.load(model_control)

    meandiff = np.mean(cancer_gmm.means_.T - control_gmm.means_.T,axis=1)
    proxy = np.zeros_like(meandiff)
    print(meandiff.shape)
    print(meandiff)

    fig = plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)

    cats = ["vocalic", "consonantal", "back", "anterior", "open", "close", "nasal", "stop", "continuant", "lateral",
     "flap", "trill", "voice", "strident", "labial", "dental", "velar"]

    meandiff, cats = (list(t) for t in zip(*sorted(zip(meandiff, cats))))
    meandiff = np.array(meandiff)
    color = np.array(list("r" * len(cats)))
    print(color.shape)
    color_binary = meandiff > 0
    color[color_binary] = "r"
    color[~color_binary] = "b"
    ax.grid(b=True,color="black",axis="x")
    ax.set_axisbelow(b=True)
    ax.barh(cats, width=meandiff, height=1, color=list(color),linewidth=1,edgecolor="black")
    ax.barh(cats,proxy)
    ax.text(0.002, -0.2, "/p/, /b/, /t/, /k/, /g/, /tS/, /d/", fontsize=22)
    ax.text(-0.018, 15.8, "/m/, /n/", fontsize=22)

    #plt.xlim([-1, 1])
    plt.xticks(rotation=50)
    plt.xlabel("mean difference of GMM bins")
    plt.legend(["more cancer like","more control like"])
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')
    #plt.show()
    plt.savefig("figures/ppg_gmm_barplot.png")

def asr_gmm_figure(model_cancer, model_control):
    font = {'size': 22}

    matplotlib.rc('font', **font)

    cancer_gmm = joblib.load(model_cancer)
    control_gmm = joblib.load(model_control)

    meandiff = np.mean(cancer_gmm.means_.T - control_gmm.means_.T,axis=1)
    #proxy = np.zeros_like(meandiff)
    print(meandiff.shape)
    print(meandiff)

    fig = plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)

    #cats = [str(i) for i in range(39)]

    cats = pd.read_csv("fac_via_ppg/test/data/phoneme_table", delimiter="\t", header=None, names=["a","b"])
    cats = cats["a"]
    print(cats)
    cats = cats[:-1]
    print(cats.shape)
    print(meandiff.shape)
    idx = np.abs(meandiff) > 0.005
    print(idx.shape)
    cats = cats[idx]
    meandiff = meandiff[idx]

    meandiff, cats = (list(t) for t in zip(*sorted(zip(meandiff, cats))))
    meandiff = np.array(meandiff)
    color = np.array(list("r" * len(cats)))
    print(color.shape)
    color_binary = meandiff > 0
    color[color_binary] = "r"
    color[~color_binary] = "b"
    ax.grid(b=True,color="black",axis="x")
    ax.set_axisbelow(b=True)
    ax.barh(cats, width=meandiff, height=1, color=list(color),linewidth=1,edgecolor="black")
    proxy = np.zeros_like(meandiff)
    ax.barh(cats,proxy)
    #ax.text(0.002, -0.2, "/p/, /b/, /t/, /k/, /g/, /tS/, /d/", fontsize=22)
    #ax.text(-0.018, 15.8, "/m/, /n/", fontsize=22)

    #plt.xlim([-1, 1])
    plt.xticks(rotation=50)
    plt.xlabel("mean difference of GMM bins")
    plt.legend(["more cancer like","more control like"])
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')
    #plt.show()
    plt.tight_layout()
    plt.savefig("figures/ppg_gmm_barplot_2.pdf",padlen=0.005)

def mean_gradcam(calculate=False):

    if calculate:
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(graph=tf.get_default_graph(),
                          config=config)
        K.set_session(sess)
        learning_rate = 0.001
        pad_len = 4000
        resnet = model.resnet.ResNet_Logspec(pad_len,257)

        optimiser = optimizers.Adam(lr=learning_rate)
        resnet.trainer.compile(optimiser, loss='categorical_crossentropy',
                               weighted_metrics=["accuracy"],
                               metrics=['accuracy'],
                               sample_weight_mode="None")

        experiment = "masked_100_frames_50_earlystopped_3"
        resnet.trainer.load_weights("checkpoints/" + experiment + ".hd5")

        test_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp"
        test_acoustic_source = KaldiSource(test_ROOT,subset="ignore",transpose=False)
        test_acoustic = PaddedFileSourceDataset(test_acoustic_source,pad_len)
        test_label = FileSourceDataset(KaldiLabelDataSource(test_ROOT))
        shuffle = False
        batch_size = 1
        val_gen = LogspecLoader(test_acoustic,test_label, shuffle, batch_size)

        dim_1 = val_gen.dim_1
        dim_2 = val_gen.dim_2
        print(dim_1)
        print(dim_2)
        # TODO: last dimension should be 3?
        healthy_grads = np.zeros((dim_1,dim_2,3))
        cancer_grads = np.zeros((dim_1,dim_2,3))

        samples = 863
        # TODO: where 3 is coming from
        stacked_grads = np.zeros((samples,dim_1*dim_2*1))
        stacked_labels = np.zeros((samples,2),dtype=int)

        num_healthy = 0
        num_cancer = 0

        # Utility to search for layer index by name.
        # Alternatively we can specify this as
        # -1 since it corresponds to the last layer.

        layer_idx = -1

        # Swap softmax with linear

        # print(model.layers[layer_idx])
        resnet.trainer.layers[layer_idx].activation = activations.linear
        resnet.trainer = utils.apply_modifications(resnet.trainer)

        for i in tqdm.tqdm(range(len(val_gen))):
            output,labels = val_gen[i]


            class_idx = labels[0]
            stacked_labels[i] = class_idx

            if class_idx[0] == 0:
                cam = visualize_cam(resnet.trainer, layer_idx, filter_indices=0,
                                  seed_input=output)

                healthy_grads = healthy_grads + cam
                #stacked_grads[i,:] = np.ravel(cam
                #plt.imshow(np.transpose(cam[:1000,:,:],[1,0,2]),aspect="auto")
                #plt.show()
                num_healthy = num_healthy + 1
            else:
                cam = visualize_cam(resnet.trainer, layer_idx, filter_indices=1,
                                  seed_input=output)

                cancer_grads = cancer_grads + cam
                #plt.imshow(cam)
                #plt.show()
                #stacked_grads[i,:] = np.ravel(cam)

                num_cancer = num_cancer + 1

        # Number of health/cancer patients
        print(num_cancer)
        print(num_healthy)


        # Calculation of mean activations on spectograms
        healthy_grads = healthy_grads / num_healthy
        cancer_grads = cancer_grads / num_cancer


        np.save("healthy_2.npy", healthy_grads)
        np.save("cancer_2.npy",cancer_grads)

    healthy_grads = np.load("healthy.npy")
    cancer_grads = np.load("cancer.npy")

    print(healthy_grads.shape)

    fig = plt.figure(num=None, figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')
    font = {'size': 26}
    matplotlib.rc('font', **font)


    plt.subplot(2,1,1)
    
    fancy_spectrogram(np.mean(healthy_grads[:1000,:,:],axis=2).T,"healthy speech")
    plt.subplot(2,1,2)
    fancy_spectrogram(np.mean(cancer_grads[:1000,:,:],axis=2).T,"oral cancer speech")
    #plt.margins(-0.4,-0.3)
    #plt.imshow(np.mean(cancer_grads[:1000,:,:],axis=2),cmap="jet")
    plt.tight_layout()
    plt.savefig("figures/mean_activation_maps.pdf",pad_len=0.005)
    #plt.show()

def fancy_spectrogram(spectrogram,title,cmap="jet"):

    # Flip upside down
    plt.imshow(np.flipud(spectrogram),cmap)

    # Labels
    plt.ylabel("Hz")
    plt.xlabel("frames")
    plt.title(title,fontsize=30 )

    y = np.linspace(8000, 0, 257)
    # the grid to which your data corresponds
    ny = y.shape[0]
    no_labels = 7  # how many labels to see on axis x
    step_y = int(ny / (no_labels - 1))  # step between consecutive labels
    y_positions = np.arange(0, ny, step_y)  # pixel count at label position
    y_labels = y[::step_y]  # labels you want to see
    plt.yticks(y_positions, y_labels)
    #plt.colorbar()

    # Calculate sum and normalise
    sum_val = np.sum(spectrogram, axis=1)
    sum_val = sum_val / np.max(sum_val) * 100
    #plt.plot(sum_val, np.arange(257)[::-1], "r--")



if __name__ == '__main__':
    #confusion_dnn(calculate=True)
    asr_gmm_figure("gmm_checkpoints/gmm_ppg_30sec_16_components_cancer.pkl",
                      "gmm_checkpoints/gmm_ppg_30sec_16_components_healthy.pkl")
    mean_gradcam(calculate=False)
    ltas_lasso_interpreter("svm_checkpoints/ard_ltas_0.01_components.pkl")
