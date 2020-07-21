python generate.py \
  --device 0 \
  --length 5 \
  --vocab_file data/vocab/vocab.txt \
  --model_path data/model/checkpoint-2 \
  --prefix "农民说：" \
  --suffix "然后回了家。" \
  --topp 1 \
  --temperature 1.0
