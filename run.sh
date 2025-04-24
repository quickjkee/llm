export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"

ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
MODEL_NAME_LLM="NousResearch/Meta-Llama-3-8B"
DATASET_PATH="configs/data/mj_sd3.5_cfg4.5_40_steps_preprocessed.yaml"


CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes=1 --mixed_precision fp16 --main_process_port $PORT main.py \
    --pretrained_model_name_or_path_dm=$MODEL_NAME \
    --pretrained_model_name_or_path_llm=$MODEL_NAME_LLM \
    --train_dataloader_config_path=$DATASET_PATH \
    --text_column="text" \
    --image_column="image" \
    --text_embedding_column="vit_l_14_text_embedding" \
    --text_embedding_2_column="vit_bigg_14_text_embedding" \
    --text_embedding_3_column="t5xxl_text_embedding" \
    --pooled_text_embedding_column="vit_l_14_pooled_text_embedding" \
    --pooled_text_embedding_2_column="vit_bigg_14_pooled_text_embedding" \
    --train_batch_size=3 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 \
    --learning_rate=2e-6 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=300 \
    --seed=42 \
    --output_dir="results" \
    --rank=64 \
    --apply_lora_to_attn_projections \
    --validation_steps=20 \
    --evaluation_steps=10 \
    --coco_ref_stats_path stats/fid_stats_mscoco256_val.npz \
    --inception_path stats/pt_inception-2015-12-05-6726825d.pth \
    --max_train_steps=1000 \
    --resume_from_checkpoint=latest \
    --max_eval_samples=100 \
    --pickscore_model_name_or_path yuvalkirstain/PickScore_v1 \
    --clip_model_name_or_path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \

