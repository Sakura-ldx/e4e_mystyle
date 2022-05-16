CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py \
--exp_dir='./experiment/PMT_lr1e-4'  --dataset_type='PMT_encode' \
--workers=2  --batch_size=2  --test_batch_size=2 --test_workers=2 \
--learning_rate=1e-4  --train_decoder=True \
--stylegan_size=1024 --checkpoint_path='./pretrained_models/e4e_ffhq_encode.pt' \
--image_interval=1000  --val_interval=10000 --save_interval=150000 --max_steps=300000 \
--save_training_data  --resume_training_from_ckpt='./experiment/PMT_lr1e-4/checkpoints/best_model.pt'

