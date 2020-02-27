
from preprocessing import FileSourceDataset, LibrosaSpectrogramSource, LogspecLoader, LabelDataSource, KaldiSource, KaldiLabelDataSource
from nnmnkwii.datasets import PaddedFileSourceDataset
import matplotlib.pyplot as plt
import model.resnet
from keras import optimizers
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend as K

import numpy as np
from tDCF_python_v1.eval_metrics import compute_eer

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def resnet_classifier():
    #print("run this")

    #np.random.seed(0)

    experiment = "masked_100_frames_50_earlystopped_3"
    audio_folder = "train_real"
    save_dir = "kaldi_spec_feature_extended"
    train = False
    learning_rate = 1e-3
    epochs = 50
    train_test_ratio = 0.8
    batch_size = 4

    masking = False
    
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(graph=tf.get_default_graph(),
                      config=config)
    K.set_session(sess)

    shuffle = True
    batch_size = 4
    pad_len = 4000

    TRAIN_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp"
    train_acoustic_source = KaldiSource(TRAIN_ROOT,subset="ignore",transpose=False)
    train_label = FileSourceDataset(KaldiLabelDataSource(TRAIN_ROOT))
    train_acoustic = PaddedFileSourceDataset(train_acoustic_source,pad_len)
    
    test_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp"
    test_acoustic_source = KaldiSource(test_ROOT,subset="ignore",transpose=False)
    test_acoustic = PaddedFileSourceDataset(test_acoustic_source,pad_len)
    test_label = FileSourceDataset(KaldiLabelDataSource(test_ROOT))

    train_loader = LogspecLoader(train_acoustic,train_label, shuffle,batch_size)
    val_loader = LogspecLoader(test_acoustic,test_label, shuffle, batch_size)


    #examples, label = train_loader[0]


    resnet = model.resnet.ResNet_Logspec(pad_len,257)
    #print(resnet.trainer.summary())

    optimiser = optimizers.Adam(lr=learning_rate)

    mcp_save = ModelCheckpoint('best_val_loss.hdf5', save_best_only=True, monitor='val_loss', mode='auto')
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=True)
    tb = TensorBoard(log_dir="logs/" + experiment)
    resnet.trainer.compile(optimiser, loss='categorical_crossentropy',
                          weighted_metrics=["accuracy"],
                          metrics=['accuracy'],
                          sample_weight_mode="None")


    if train:
        resnet.trainer.fit_generator(generator=train_loader,
                                     validation_data=val_loader,
                                    use_multiprocessing=False,
                                    workers=8,  # 8
                                    epochs=epochs,
                                    max_queue_size=10,
                                    callbacks=[tb,es,mcp_save],
                                     verbose=1)
        resnet.trainer.save_weights("checkpoints/" + experiment + ".hd5")
    else:
        resnet.trainer.load_weights("checkpoints/" + experiment + ".hd5")


    results = resnet.trainer.evaluate_generator(val_loader)
    print(results[1],end="\t")

    val_loader = LogspecLoader(test_acoustic,test_label, shuffle, batch_size=1)
    healthy_scores = []
    cancer_scores = []
    for i in range(len(val_loader)):
        x,label = val_loader[i]
        score = resnet.trainer.predict(x)
        if label[0,0] == 0:
            cancer_scores.append(score[0,0])
        else:
            healthy_scores.append(score[0,0])

    eer, _ = compute_eer(np.array(healthy_scores), np.array(cancer_scores))
    print(eer,end="")




if __name__ == '__main__':
    resnet_classifier()
