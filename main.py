"""
Perturb and optimize
"""
import argparse
import os

import monai
import numpy as np
import torch
import torchio
import wandb

from VoxelMorph import VxmDense
from utils import set_seed, multi_resolution_NCC, TrainableTensor, save_data
import SimpleITK as sitk

DEFAULT_TRANSFORM_SETTINGS = {
    'biasfield': [0., 0.2, 0.5]}

MODEL_PATH = './vxm_dense_brain_T1_3D_mse_pytorch.pth'
fixed_image_path = './example_data/OASIS_0075_0000.nii.gz'
moving_image_path = './example_data/OASIS_0001_0000.nii.gz'

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-seed', '--random_seed', type=int, metavar='', default=1000, help='random seed')
    parser.add_argument('--perturb_type', type=str, metavar='', default='biasfield', help='noise, blur, biasfield')
    parser.add_argument('--perturb_magn', type=float, metavar='', default=None, nargs='+')

    # Optim
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='', default=0.01, help='learning rate')
    parser.add_argument('-its', '--max_iterations', type=int, metavar='', default=100, help='maximum iterations')
    parser.add_argument('--threshold', type=float, metavar='', default=1e-4, help='threshold')
    parser.add_argument('--patience', type=float, metavar='', default=20, help='threshold')
    parser.add_argument('-dev', '--device', type=str, metavar='', default='cpu', help='device / gpu used')
    args = parser.parse_args()
    args.imgsize = (160, 192, 160)

    if args.perturb_magn is None:
        args.perturb_magn = DEFAULT_TRANSFORM_SETTINGS[args.perturb_type]

    print(args.perturb_magn)
    return args

def optim(model,
          moving, fixed,
          args):

    # Make movign image a trainable tensor and put images on device
    moving_trainable = TrainableTensor(moving.clone())
    moving_untrainable = moving_trainable.tensor.detach()

    # init loss term
    loss_similarity = multi_resolution_NCC(win=7, scale=3)

    # Init optimizer
    optimizer = torch.optim.Adam(moving_trainable.parameters(), lr=args.learning_rate)
    prev_loss = float('inf')

    # Perform optimization
    wait_count = 0
    for step in range(args.max_iterations):
        _, dvf_pred = model(moving_trainable.tensor, fixed)
        moving_warped = model.transformer(moving_untrainable, dvf_pred)

        # Compute loss and backpropagate
        loss = loss_similarity(moving_warped, fixed)

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        moving_untrainable = moving_trainable.tensor.detach()
        # wandb.log({"loss": loss.item()})

        # Check if the change in loss is less than the threshold
        loss_change = prev_loss - loss.item()
        if loss_change >= args.threshold:
            wait_count = 0
        else:
            wait_count += 1
            if wait_count >= args.patience:
                print(f"Training stopped at step {step} because loss change is below {args.threshold}.")
                break

        # Update previous loss for the next iteration
        prev_loss = loss
    return moving_trainable.tensor, fixed, moving_warped, dvf_pred


if __name__ == "__main__":
    args = parse_arguments()
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="",
    #     mode='online',
    #     config=dict(**vars(args)))

    device = torch.device(args.device)
    save_path_images = './output/images_optim'
    if not os.path.exists(save_path_images):
        os.makedirs(save_path_images)

    # load pretrained model and freeze trainable parameters
    model = VxmDense((160, 192, 160),
                     unet_half_res=True,
                     int_downsize=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.freeze()

    # Load example data
    fixed_image = torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path)).astype('float32')).unsqueeze(0).permute(0, 3, 2, 1)
    moving_image = torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)).astype('float32')).unsqueeze(0).permute(0, 3, 2,
                                                                                                          1)
    # iterate over transforms T
    for value in args.perturb_magn:
        set_seed(args.random_seed)

        # Perturb images with perturbation
        if value > 0.0:
            if args.perturb_type == 'biasfield':
                transform = torchio.transforms.RandomBiasField(coefficients=(-value, value))
            if args.perturb_type == 'blur':
                transform = torchio.transforms.Blur(std=(value,) * 3)
            if args.perturb_type == 'noise':
                transform = monai.transforms.RandRicianNoise(std=value, prob=1.0, channel_wise=True, relative=True, sample_std=False)
            fixed = torch.clamp(transform(fixed_image), 0., 1.).to(device).unsqueeze(0)
            moving = torch.clamp(transform(moving_image), 0., 1.).to(device).unsqueeze(0)
        else:
            fixed = fixed_image.clone().to(device).unsqueeze(0)
            moving = moving_image.clone().to(device).unsqueeze(0)

        # Infer model without input optimization
        moving_warped, dvf_pred = model(moving, fixed)

        # Saving perturbed input space ...
        save_data(save_path_images, 0,
                  moving, fixed,
                  moving_warped, dvf_pred,
                  category='perturbed_{}'.format(value))

        # Infer model with input optimization
        moving_proj, fixed_proj, moving_warped_proj, dvf_pred_proj = optim(model, moving, fixed, args=args)

        # Saving projected input space  ...
        save_data(save_path_images, 0,
                  moving_proj, fixed_proj,
                  moving_warped_proj, dvf_pred_proj,
                  category='projected_{}'.format(value))