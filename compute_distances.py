import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

from main import DEFAULT_TRANSFORM_SETTINGS
from utils import read_data

def mse(gt, pred):
    """Compute Mean Squared Error (MSE)"""
    return torch.mean((gt - pred) ** 2)

def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict. Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            dict_obj[key] = [dict_obj[key]]  # If type is not list then make it list
        dict_obj[key].append(value)
    else:
        # As key is not in dict, add key-value pair
        dict_obj[key] = value

if __name__ == "__main__":
    perturb_type = 'biasfield'

    save_path_images = './output/images_optim'
    perturb_magn = DEFAULT_TRANSFORM_SETTINGS[perturb_type]

    dataframe_list = []

    csv_path = './output/distances__{}_{}.csv' .format(perturb_type, perturb_magn)
    print('Perturbation type: {} - Values: {}'.format(perturb_type, perturb_magn))

    # Init distance metrics
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # ssim = StructuralSimilarityIndexMeasure(data_range=1.)
        metrics = dict()

        # Load perturbed and projected images
        for value in tqdm(perturb_magn, file=sys.stdout):
            for batch_idx in [0]:
                # Compute metric for each perturbed case in the dataset
                for key, val in zip(
                        ['perturb_type', 'perturb_magn', 'batch_idx'],
                        [perturb_type, value, batch_idx]):
                    append_value(metrics, key, val)

                moving, fixed, moving_warped, dvf_pred = read_data(save_path_images, batch_idx, category='perturbed_{}'.format(0.0))
                moving_pert, fixed_pert, moving_warped_pert, dvf_pred_pert = read_data(save_path_images, batch_idx, category='perturbed_{}'.format(value))
                moving_proj, fixed_proj, moving_warped_proj, dvf_pred_proj = read_data(save_path_images, batch_idx, category='projected_{}'.format(value))
                images = [moving, moving_warped, dvf_pred.squeeze()]
                images_perturbed = [moving_pert, moving_warped_pert, dvf_pred_pert.squeeze()]
                images_projected = [moving_proj, moving_warped_proj, dvf_pred_proj.squeeze()]

                for which, image_list_1, image_list_2 in zip(['pert', 'proj'],
                                                             [images , images_perturbed],
                                                             [images_perturbed, images_projected]):
                    for i, input in enumerate(['x', 'y', 'dvf']):
                        append_value(metrics,
                                     f'mse_{input}_{which}',
                                     mse(image_list_1[i], image_list_2[i]).item())
        df = pd.DataFrame(metrics)
        df.to_csv(csv_path)

