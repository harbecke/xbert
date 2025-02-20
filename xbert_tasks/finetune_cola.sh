export GLUE_DIR=../data/glue_data/
export TASK_NAME=CoLA

python run_glue.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 1e-5 \
  --warmup_steps 320 \
  --save_steps 1000 \
  --max_steps 5336 \
  --logging_steps 500 \
  --eval_all_checkpoints \
  --output_dir /raid/calt/roberta/$TASK_NAME/
