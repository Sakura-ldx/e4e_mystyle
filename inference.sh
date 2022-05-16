for i in {1..2}
do
echo "$i"
CUDA_VISIBLE_DEVICES=1 python ./scripts/inference.py \
--images_dir=./PMT_dataset/inversion_nonmakeup.txt  --edit_attribute="makeup$i"  --edit_degree=1.0 \
--save_dir=./result/PMT_align_lr1e-4/nonmakeup  --ckpt=./experiment/PMT_align_lr1e-4/checkpoints/best_model.pt
done

#CUDA_VISIBLE_DEVICES=1 python ./scripts/inference.py \
#--images_dir=./PMT_dataset/inversion_makeup.txt  --edit_attribute='inversion'  \
#--save_dir=./result/PMT_lr1e-4/makeup  --ckpt=./experiment/PMT_lr1e-4/checkpoints/best_model.pt

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='beard'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='eyes'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='smile' --edit_degree=1.0  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='age' --edit_degree=3  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 
