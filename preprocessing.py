"""
Prepare features for DNN-training
"""

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from htk_io import HTKFile
import librosa
from glob import glob
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
from os.path import join,splitext,basename,split
import os
import keras
#from phonet import Phonet as phon
import kaldi_io as ko
import adv_kaldi_io as ako
import librosa

import scipy

class KaldiSource(FileDataSource):
    """
    Generic nnnkwii source class for Kaldi frontend
    """
    def __init__(self,scp_file,subset,delta_delta=False,transpose=True,normalise=False,max_files=None):
        self.max_files = max_files
        self.scp_file = scp_file
        self.subset = subset
        self.transpose = transpose
        self.normalise = normalise
        self.key_list = ako.read_all_key(scp_file)
        self.delta_delta = delta_delta
        self.alpha = None

    def collect_files(self):

        global_speaker_list = ["id011","id002","id003","id004","id007","id012","id013","id001","id005","id006","id008",
                               "id10242", "id10571", "id10078", "id10111", "id11250", "id10094", "id11217", "id10509",
                               "id10110","id10013", "id10855"]
        if self.subset == "cancer":
            speaker_list = ["id011","id002","id003","id004","id007","id012","id013","id001","id005","id006","id008"]
        elif self.subset == "healthy":
            speaker_list = ["id10242","id10571","id10078","id10111","id11250","id10094","id11217","id10509","id10110",
            "id10013","id10855"]
        elif self.subset in global_speaker_list:
            speaker_list = [self.subset]
        else:
            return self.key_list

        utterance_list = [utt_id for utt_id in self.key_list if any(speaker in utt_id for speaker in speaker_list)]

        return utterance_list

    def collect_features(self, utt_id):

        # Load data and get label
        #print(utt_id)

        tensor = ako.read_mat_key(self.scp_file, utt_id)


        if self.transpose:
            X = tensor.T
        else:
            X = tensor

        if self.delta_delta:
            delta = librosa.feature.delta(X,order=1)
            delta_delta = librosa.feature.delta(X,order=2)

            X = np.vstack((X,delta,delta_delta))

        if self.normalise:
            mu = np.mean(X)
            std = np.std(X)
            X = (X-mu)/std
        #print(X.shape)
        return X.astype(np.float32)


class HTKSource(FileDataSource):
    """
    Generic HTK file front end for nnmnkwii
    """
    def __init__(self,htk_dir,subset,transpose=True,max_files=None):
        self.max_files = max_files
        self.htk_dir = htk_dir
        self.transpose = transpose
        self.subset = subset
        self.alpha = None
        self.htk_reader = HTKFile()


    def collect_files(self):

        file_list = glob(join(self.htk_dir,"*.htk"))
        if self.subset == "cancer":
            speaker_list = ["id011","id002","id003","id004","id007","id012","id013","id001","id005","id006","id008"]
        elif self.subset == "healthy":
            speaker_list = ["id10242","id10571","id10078","id10111","id11250","id10094","id11217","id10509","id10110",
            "id10013","id10855"]
        else:
            return file_list

        utterance_list = [utt_id for utt_id in file_list if any(speaker in utt_id for speaker in speaker_list)]

        return utterance_list

    def collect_features(self, utt_id):
        # Load data and get label
        self.htk_reader.load(utt_id)
        result = np.array(self.htk_reader.data)
        if self.transpose:
            result = result.T
        return result.astype(np.float32)


class LTASSource(KaldiSource):
    def __init__(self,scp_file,subset,max_files=None):
        super().__init__(scp_file,subset)

    def collect_files(self):
        return super().collect_files()

    def collect_features(self, utt_id):
        # Load data and get label
        tensor = ako.read_mat_key(self.scp_file, utt_id)
        X = tensor.T
        mean_ltas = np.mean(X,axis=1)
        std_ltas = np.std(X,axis=1)
        features = np.hstack((mean_ltas,std_ltas))
        features = features[None,:].T
        #y = self.utt2label[utt_id]
        return features.astype(np.float32)





class MFCCSource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):

        # TODO Change back if needed: (For MFCC Initial comomented works, but for Kaldi recipe it does not)
        #wav_paths = sorted(glob(join(self.data_root, "wav",  "*.wav")))
        #print(self.data_root)
        wav_paths = sorted(glob(join(self.data_root, "*","*.wav")))
        label_paths = wav_paths
        #label_paths = sorted(glob(join(self.data_root, label_dir_name, "*.lab")))
        if self.max_files is not None and self.max_files > 0:
            return wav_paths[:self.max_files], label_paths[:self.max_files]
        else:
            return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)
        mfcc = librosa.feature.mfcc(x)
        return mfcc.astype(np.float32)

# class PhonSource(MFCCSource):
#     def __init__(self,data_root,max_files=None):
#         #pass
#         self.phone_instance = phon(["vocalic", "consonantal", "back", "anterior", "open", "close", "nasal", "stop", "continuant", "lateral",
#                                "flap", "trill", "voice", "strident", "labial", "dental", "velar", "pause"])
#         super().__init__(data_root,max_files)
#
#     def collect_files(self):
#         #pass
#         return super().collect_files()
#
#     def collect_features(self, wav_path,label_path):
#         try:
#             phon2 = self.phone_instance.get_posteriorgram(wav_path)
#         except:
#             print("Failed feature extraction due to short utterance in file", print(wav_path))
#             phon2 = np.zeros((18,1))
#         return phon2

import sys
sys.path.insert(0,'/home/boomkin/PycharmProjects/oral_cancer_analysis/fac_via_ppg/src')
import fac_via_ppg.src.ppg as ppg
import fac_via_ppg.src.common.feat as feat

class PhonSource2(MFCCSource):
    def __init__(self,data_root,max_files=None):
        #pass
        super().__init__(data_root,max_files)

    def collect_files(self):
        #pass
        return super().collect_files()

    def collect_features(self, wav_path,label_path):
        deps = ppg.DependenciesPPG()
        wave_data = feat.read_wav_kaldi(wav_path)
        ppgs = ppg.compute_monophone_ppg(wave_data, deps.nnet, deps.lda,
                                         deps.monophone_trans)
        return ppgs

class LabelDataSource(MFCCSource):
    def __init__(self,data_root,max_files=None):
        super().__init__(data_root,max_files)

    def collect_files(self):
        return super().collect_files()

    def collect_features(self, wav_path, label_path):

        speaker_list_cancer = ["id011","id002","id003","id004","id007","id012","id013","id001","id005","id006","id008"]
        speaker_list_healthy = ["id10242","id10571","id10078","id10111","id11250","id10094","id11217","id10509","id10110",
            "id10013","id10855"]

        if any(speaker in wav_path for speaker in speaker_list_cancer):
            label = [0,1]
        elif any(speaker in wav_path for speaker in speaker_list_healthy):
            label = [1,0]
        else:
            #print(wav_path)
            raise NameError("File does not have any label in its pathname")

        return label


class KaldiLabelDataSource(KaldiSource):
    def __init__(self,data_root,max_files=None):
        super().__init__(data_root,max_files)

    def collect_files(self):
        return super().collect_files()

    def collect_features(self, wav_path):

        #print(wav_path)
        speaker_list_cancer = ["id011","id002","id003","id004","id007","id012","id013","id001","id005","id006","id008"]
        speaker_list_healthy = ["id10242","id10571","id10078","id10111","id11250","id10094","id11217","id10509","id10110",
            "id10013","id10855"]

        if any(speaker in wav_path for speaker in speaker_list_cancer):
            label = [0,1]
        elif any(speaker in wav_path for speaker in speaker_list_healthy):
            label = [1,0]
        else:
            print(wav_path)
            raise NameError("File does not have any label in its pathname")

        return label


class LibrosaSpectrogramSource(MFCCSource):
    def __init__(self,data_root,max_files=None):
        super().__init__(data_root,max_files)

    def collect_files(self):
        return super().collect_files()

    def collect_features(self, wav_path, label_path):
        """
        Creates the format for the framewise data loading of the dataset
        audio_path - folder containing audio to be processed
        save_dir - folder to save features, etc.
           """
        n_fft = 512
        window_length = 20

        sound, fs = librosa.core.load(wav_path, sr=16000)

        if fs != 16000:
            print(wav_path)

        # Preemphasis
        preemp_sound = np.append(sound[0], sound[1:] - 0.97 * sound[:-1])

        # STFT
        spect = librosa.core.stft(preemp_sound,
                                  n_fft=n_fft,
                                  win_length=window_length * int(fs / 1000),
                                  hop_length=window_length * int(fs / 2000),
                                  window=scipy.signal.hamming,
                                  center=True)

        spect = np.log10(np.transpose(abs(spect[:, 1:]) ** 2) + 1e-16)

        return spect


class NPYDataSource(FileDataSource):
    def __init__(self, data_root, subset, train, max_files=None):
        self.data_root = data_root
        self.subset = subset
        self.train = train
        self.max_files = max_files

    def collect_files(self):

        if self.train:
            npy_files = sorted(glob(join(self.data_root,"train","mfcc",self.subset + "*.npy")))
        else:
            npy_files = sorted(glob(join(self.data_root,"test","mfcc",self.subset + "*.npy")))

        return npy_files

    def collect_features(self, path):
        return np.load(path)

class NPYDataSource2(FileDataSource):
    def __init__(self, data_root, subset, max_files=None, transpose=False):
        self.data_root = data_root
        self.subset = subset
        self.max_files = max_files
        self.transpose = transpose
    def collect_files(self):

        self.file_list = glob(join(self.data_root,"*.npy"))
        #print(self.file_list)
        if self.subset == "cancer":
            speaker_list = ["id011","id002","id003","id004","id007","id012","id013","id001","id005","id006","id008"]
        elif self.subset == "healthy":
            speaker_list = ["id10242","id10571","id10078","id10111","id11250","id10094","id11217","id10509","id10110",
            "id10013","id10855"]
        else:
            print("baj")

        file_list = [file_name for file_name in self.file_list if any(speaker in file_name for speaker in speaker_list)]

        return file_list

        return npy_files

    def collect_features(self, path):

        feat = np.nan_to_num(np.load(path))
        if self.transpose:
            feat = np.transpose(feat)
        return feat

def combine_stack_and_label(filesource_dataset_1,filesource_dataset_2,num_sample):
    """
    Stacks two datasets (healthy and cancer) and creates the label for them to be suitable
    for 2-class classification problems

    :param filesource_dataset_1:
    :param filesource_dataset_2:
    :return:
    """

    x = filesource_dataset_1[0]
    x_utterances = len(filesource_dataset_1)
    for idx in tqdm(range(1, x_utterances)):
        x = np.hstack((x, filesource_dataset_1[idx]))
        #print(x.shape)
    y = filesource_dataset_2[0]
    y_utterances = len(filesource_dataset_2)
    for idx in tqdm(range(1, y_utterances)):
        y = np.hstack((y, filesource_dataset_2[idx]))
    X = np.hstack((x,y))
    Y = np.hstack((np.ones((x.shape[1])),np.zeros((y.shape[1]))))

    if (X.shape[1] > num_sample):
        idx = np.random.choice(X.shape[1], num_sample)
        X = X[:, idx]
        Y = Y[idx]
    return X, Y

class LogspecLoader(keras.utils.Sequence):
    def __init__(self, file_source,label_source,shuffle,batch_size):

        self.file_source = file_source
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_source))
        self.batch_size = batch_size
        self.dim_1 = file_source[0].shape[0]
        #print(self.dim_1)
        self.dim_2 = file_source[0].shape[1]
        #print(self.dim_2)
        self.label_source = label_source
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation(indexes)
        return X, y

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        """

        # Initialization
        x = np.zeros((self.batch_size, self.dim_1,self.dim_2,1))
        y = np.zeros((self.batch_size, 2))

        # Generate data

        for i, ID in enumerate(indexes):

            x[i,:,:,0] = self.file_source[ID]
            y[i,:] = self.label_source[ID]

        return x, y







if __name__ == '__main__':




    # root_dir = "/home/boomkin/repos/kaldi/egs/cancer_30/"
    # scp_file = "/home/boomkin/repos/kaldi/egs/cancer_30/audio/train"
    # train_acoustic_source = PhonSource(scp_file)
    # train_acoustic = FileSourceDataset(train_acoustic_source)
    #
    #
    # for idx in tqdm(range(len(train_acoustic))):
    #     x = train_acoustic[idx]
    #     name = splitext(basename(train_acoustic.collected_files[idx][0]))[0]
    #     speaker = split(split(train_acoustic.collected_files[idx][0])[0])[1]
    #     xpath = join(root_dir, "data", "train_ppg",speaker + "_" + name)
    #     np.save(xpath, x)

    # # Preparing training data
    #TRAIN_ROOT = "/home/boomkin/repos/kaldi/egs/cancer_30/audio/train"
    #train_acoustic_source = LibrosaSpectrogramSource(TRAIN_ROOT)
    #train_acoustic = FileSourceDataset(train_acoustic_source)
    # train_acoustic_dir = join(TRAIN_ROOT, "mfcc")
    # # Prepare testing data
    # TEST_ROOT = "/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/wav/test"
    # test_acoustic_source = MFCCSource(TEST_ROOT)
    # test_acoustic = FileSourceDataset(test_acoustic_source)
    # test_acoustic_dir = join(TEST_ROOT, "mfcc")
    #
    #
    # print("Acoustic linguistic feature dim", train_acoustic[0].shape[-1])
    # print(len(train_acoustic))
    #

    root_dir = "/home/boomkin/repos/kaldi/egs/cancer_30"
    ppg_dir = "/home/boomkin/repos/kaldi/egs/cancer_30/audio/train"
    train_acoustic_source = PhonSource2(ppg_dir)
    train_acoustic = FileSourceDataset(train_acoustic_source)
    import matplotlib.pyplot as plt
    for idx in tqdm(range(len(train_acoustic))):
          x = train_acoustic[idx]

          name = splitext(basename(train_acoustic.collected_files[idx][0]))[0]
          speaker = split(split(train_acoustic.collected_files[idx][0])[0])[1]
          xpath = join(root_dir, "data", "train_ppg_asr", speaker + "_" + name + ".npy")
          np.save(xpath, x)

    root_dir = "/home/boomkin/repos/kaldi/egs/cancer_30"
    ppg_dir = "/home/boomkin/repos/kaldi/egs/cancer_30/audio/test"
    train_acoustic_source = PhonSource2(ppg_dir)
    train_acoustic = FileSourceDataset(train_acoustic_source)
    import matplotlib.pyplot as plt
    for idx in tqdm(range(len(train_acoustic))):
          x = train_acoustic[idx]

          name = splitext(basename(train_acoustic.collected_files[idx][0]))[0]
          speaker = split(split(train_acoustic.collected_files[idx][0])[0])[1]
          xpath = join(root_dir, "data", "test_ppg_asr", speaker + "_" + name + ".npy")
          np.save(xpath, x)

    #      name = splitext(basename(test_acoustic.collected_files[idx][0]))[0]
    #      speaker = split(split(test_acoustic.collected_files[idx][0])[0])[1]
    #


    # root_dir = "/home/boomkin/repos/kaldi/egs/cancer_30/"
    # scp_file = "/home/boomkin/repos/kaldi/egs/cancer_30/audio/test"
    # test_acoustic_source = PhonSource(scp_file)
    # test_acoustic = FileSourceDataset(test_acoustic_source)
    #
    # for idx in tqdm(range(len(test_acoustic))):
    #     x = test_acoustic[idx]
    #     name = splitext(basename(test_acoustic.collected_files[idx][0]))[0]
    #     speaker = split(split(test_acoustic.collected_files[idx][0])[0])[1]
    #     xpath = join(root_dir, "data", "test_ppg", speaker + "_" + name)
    #     np.save(xpath, x)

         #name = splitext(basename(train_acoustic.collected_files[idx][0]))[0]
         #xpath = join(train_acoustic_dir, name)
    #     np.save(xpath, x)
    #
    # for idx in tqdm(range(1, len(test_acoustic))):
    #     x = test_acoustic[idx]
    #     name = splitext(basename(test_acoustic.collected_files[idx][0]))[0]
    #     xpath = join(test_acoustic_dir, name)
    #     np.save(xpath, x)
