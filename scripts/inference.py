import argparse

import torch
import numpy as np
import sys
import os
import dlib
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("..")


from criteria import id_loss
from criteria.lpips.lpips import LPIPS
from utils import train_utils
from editings import latent_editor
from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'car' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)
    editor = latent_editor.LatentEditor(net.decoder, is_cars)

    # initial inversion
    latent_codes, names = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
    dic = {names[i]: latent_codes[i] for i in range(len(names))}

    # set the editing operation
    if args.edit_attribute == 'inversion':
        edit_directory_path = os.path.join(args.save_dir, args.edit_attribute)
        os.makedirs(edit_directory_path, exist_ok=True)

        f = open(os.path.join(edit_directory_path, 'timestamp.txt'), 'a')
        id_l = id_loss.IDLoss().to(device).eval()
        lpips_l = LPIPS(net_type='alex').to(device).eval()
        agg_loss_dict = []

        torch.save(dic, os.path.join(edit_directory_path, 'latent.pt'))
    elif args.edit_attribute == 'age' or args.edit_attribute == 'smile' or 'makeup' in args.edit_attribute:
        interfacegan_directions = {
            'age': './editings/interfacegan_directions/age.pt',
            'smile': './editings/interfacegan_directions/smile.pt',
            'makeup1': './editings/interfacegan_directions/makeup1.pt'
        }
        edit_directory_path = os.path.join(args.save_dir, f'{args.edit_attribute}_{args.edit_degree}')
        os.makedirs(edit_directory_path, exist_ok=True)
        edit_direction = torch.load(f'./editings/interfacegan_directions/align/{args.edit_attribute}.pt').to(device)
    else:
        ganspace_pca = torch.load('./editings/ganspace_pca/ffhq_pca.pt')
        ganspace_directions = {
            'eyes': (54, 7, 8, 20),
            'beard': (58, 7, 9, -20),
            'lip': (34, 10, 11, 20)}
        edit_direction = ganspace_directions[args.edit_attribute]

    # perform high-fidelity inversion or editing
    with torch.no_grad():
        for i, (batch, name) in enumerate(data_loader):
            if args.n_sample is not None and i > args.n_sample:
                print('inference finished!')
                break
            x = batch.to(device).float()

            # calculate the distortion map
            imgs, _ = generator([latent_codes[i].unsqueeze(0).to(device)], input_is_latent=True, randomize_noise=False, return_latents=True)

            # produce initial editing image
            # edit_latents = editor.apply_interfacegan(latent_codes[i].to(device), interfacegan_direction, factor_range=np.linspace(-3, 3, num=40))
            if args.edit_attribute == 'inversion':
                img_edit = imgs
            elif args.edit_attribute == 'age' or args.edit_attribute == 'smile' or 'makeup' in args.edit_attribute:
                img_edit = editor.apply_interfacegan(latent_codes[i].unsqueeze(0).to(device), edit_direction,
                                                                   factor=args.edit_degree)
            else:
                img_edit = editor.apply_ganspace(latent_codes[i].unsqueeze(0).to(device), ganspace_pca,
                                                               [edit_direction])

            imgs = img_edit
            if is_cars:
                imgs = imgs[:, :, 64:448, :]

            # save images
            imgs = torch.nn.functional.interpolate(imgs, size=(256, 256), mode='bilinear')
            if args.edit_attribute == 'inversion':
                loss_dict = cal_loss(id_l, lpips_l, x, x, imgs)
                f.write('name - {}\n{}\n'.format(name, loss_dict))
                agg_loss_dict.append(loss_dict)

            save_image(imgs[0], edit_directory_path, f"{name[0]}")
    if args.edit_attribute == 'inversion':
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        f.write('total\n{}\n'.format(loss_dict))
        f.close()


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents, names = [], []
    i = 0
    with torch.no_grad():
        for (x, name) in data_loader:
            if n_images is not None and i > n_images:
                break
            print(i, ' ', name[0])
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            names.append(name[0])
            i += len(latents)
    print('end')
    return torch.cat(all_latents), names


def save_image(img, save_dir, name):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, name)
    print(im_save_path)
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, names, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(args.n_sample):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, names[i])


def cal_loss(id_l, lpips_l, x, y, y_hat):
    loss_dict = {}

    loss_id, _, _ = id_l(y_hat, y, x)
    loss_dict['loss_id'] = float(loss_id)

    loss_l2 = F.mse_loss(y_hat, y)
    loss_dict['loss_l2'] = float(loss_l2)

    loss_lpips = lpips_l(y_hat, y)
    loss_dict['loss_lpips'] = float(loss_lpips)
    return loss_dict


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['my_shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default='../PMT_dataset/inversion.txt',
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default='./PMT_inversion_my',
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")
    parser.add_argument("--edit_attribute", default='inversion', type=str, help="path to generator checkpoint")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degree")

    args = parser.parse_args()
    main(args)
