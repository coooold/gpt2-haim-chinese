python train.py \
	--stride 384 \
	--model_config config/model-dev.json \
	--data_dir data/tokenized \
	--overwrite_output_dir \
	--do_train \
	--num_train_epochs 1 \
	--save_steps 15 \
	--output_dir data/model \
	--save_total_limit 1 \
	--per_device_train_batch_size 4 \
	--learning_rate 4e-4 \
	--vocab_file data/vocab/clue_vocab.txt
	#--pretrained_model_path model/model_epoch_1
