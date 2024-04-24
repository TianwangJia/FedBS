python3 -u ../src/train.py \
--model shallowconvnet \
--sample_rate 250 \
--TemporalKernel_Times 1 \
--class_num 4 \
--channels 22 \
--samples 1001 \
--dropout 0.5 \
--data_path '../data/BNCI2014001/' \
--sub_id '1,2,3,4,5,6,7,8,9' \
--output_path '../output' \
--ea True \
--global_epochs 200 \
--sample_num 4 \
--local_epochs 2 \
--batch_size 32 \
--lr 0.0001 \
--early False \
--fedbs True \
--rho 0.1 \