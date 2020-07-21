python train.py \
	--stride 40 \
	--model_config config/model-dev.json \
	--data_dir data/tokenized \
	--overwrite_output_dir \
	--do_train \
	--num_train_epochs 1 \
	--save_steps 1 \
	--output_dir data/model \
	--vocab_file data/vocab/vocab.txt
	#--pretrained_model_path model/model_epoch_1
