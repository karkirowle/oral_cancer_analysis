

source venv/bin/activate
out_file=results_debug.csv
echo "Extract features"
/repos/kaldi/egs/cancer_30/run_feature.sh stage=-3

array=(4 8 10 12 16 )

printf "model" >> $out_file
printf "\t" >> $out_file
printf "accuracy" >> $out_file
printf "\t" >> $out_file
printf "EER" >> $out_file
printf "\n" >> $out_file


printf "ResNet" >> $out_file
printf "\t" >> $out_file
python train_DNN.py >> $out_file
printf "\n" >> $out_file


for num in ${array[*]}
do
printf "Pitch $num" >> $out_file
printf "\t" >> $out_file
	python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_pitch/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_pitch/feats.scp" --gmm_comps=$num --experiment="gmm_pitch_30sec_${num}_components" --train >> $out_file
done

for num in ${array[*]}
do
printf "PPG $num" >> $out_file
printf "\t" >> $out_file
	python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp" --train --gmm_comps=$num --experiment="gmm_ppg_30sec_${num}_components" --ppg >> $out_file
done
for num in ${array[*]}
do
printf "PLP $num" >> $out_file
printf "\t" >> $out_file
	python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_plp_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_plp_vad/feats.scp" --train --gmm_comps=$num --experiment="gmm_plp_30sec_${num}_components" >> $out_file
done

for num in ${array[*]}
do
printf "PLP DD $num" >> $out_file
printf "\t" >> $out_file
python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_plp_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_plp_vad/feats.scp" --train --gmm_comps=$num --experiment="gmm_plp_30sec_delta_delta_${num}_components" --delta >> $out_file
done



for num in ${array[*]}
do
printf "MFCC $num" >> $out_file
printf "\t" >> $out_file
	python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_mfcc_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_mfcc_vad/feats.scp" --train --gmm_comps=$num --experiment="gmm_mfcc_30sec_${num}_components" >> $out_file
done

for num in ${array[*]}
do
printf "MFCC DD $num" >> $out_file
printf "\t" >> $out_file
python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_mfcc_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_mfcc_vad/feats.scp" --train --gmm_comps=$num --experiment="gmm_mfcc_30sec_delta_delta_${num}_components" --delta >> $out_file
done



for num in ${array[*]}
do
printf "LTAS $num" >> $out_file
printf "\t" >> $out_file
	python3 train_GMM.py --train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp" --train --gmm_comps=$num --experiment="gmm_ltas_30sec_${num}_components" --ltas >> $out_file
done



#train_GMM.py--train_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/train_spec_vad/feats.scp" --test_scp_file="/home/boomkin/repos/kaldi/egs/cancer_30/data/test_spec_vad/feats.scp" --train --gmm_comps=16 --experiment="gmm_ltas_30sec"


