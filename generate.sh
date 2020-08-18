python generate.py \
  --device 0 \
  --length 5 \
  --vocab_file data/vocab/clue_vocab.txt \
  --model_path data/model/checkpoint-15 \
  --topp 1 \
  --temperature 1.0 \
  --nsamples 5
