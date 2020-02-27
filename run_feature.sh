. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank
plpdir=`pwd`/plp
specdir=`pwd`/logspec
vadir=`pwd`/vad
stage=0
num_components=128 # UBM
ivector_dim=200 # ivector

root_dir=/media/boomkin/HD-B2/datasets/oral_cancer_speaker_partitioned/wav/train

ls $root_dir

if [ $stage -eq 0 ]; then
    # cqcc_spectrogram (863) feature extraction
    # apply cmvn sliding window
      utils/fix_data_dir.sh data/train_cqcc_spectrogram
      utils/copy_data_dir.sh data/train_cqcc_spectrogram data/train_cqcc_spectrogram_cmvn
      feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_cqcc_spectrogram/feats.scp ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_cqcc_spectrogram_cmvn/feats.ark,`pwd`/data/${name}_cqcc_spectrogram_cmvn/feats.scp
fi
