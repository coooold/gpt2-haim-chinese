python generate.py \
  --length 500 \
  --vocab_file data/vocab/clue_vocab.txt \
  --model_path data/model_gpu \
  --repetition_penalty 1.15 \
  --topp 0.9 --topk 50 \
  --temperature 1.0 \
  --nsamples 3
