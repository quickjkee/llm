export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"

ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
DATASET_PATH="configs/data/mj_sd3.5_cfg4.5_40_steps_preprocessed.yaml"


CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 --multi_gpu --mixed_precision fp16 --main_process_port $PORT main.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_dataloader_config_path=$DATASET_PATH \
    --text_column="text" \
    --image_column="image" \
    --train_batch_size=2 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 \
    --learning_rate=2e-6 \
    --num_boundaries=6 \
    --scales="32,48,64,80,96,128" \
    --do_gan_loss \
    --do_pixels_downscale \
    --stochastic_case \
    --num_discriminator_upds=3 \
    --num_discriminator_layers=4 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=300 \
    --seed=42 \
    --output_dir="results" \
    --rank=64 \
    --cls_blocks=11 \
    --pdm_blocks=22 \
    --cfg_teacher=4.5 \
    --cfg_fake=4.5 \
    --apply_lora_to_attn_projections \
    --apply_lora_to_mlp_projections \
    --validation_steps=20 \
    --evaluation_steps=10 \
    --coco_ref_stats_path stats/fid_stats_mscoco256_val.npz \
    --inception_path stats/pt_inception-2015-12-05-6726825d.pth \
    --max_train_steps=1000 \
    --resume_from_checkpoint=latest \
    --max_eval_samples=100 \
    --pickscore_model_name_or_path yuvalkirstain/PickScore_v1 \
    --clip_model_name_or_path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
