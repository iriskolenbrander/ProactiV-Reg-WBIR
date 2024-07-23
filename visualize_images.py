import numpy as np
from matplotlib import pyplot as plt

from main import DEFAULT_TRANSFORM_SETTINGS
from utils import read_data, acs_slice, to_numpy


def visualize(images_perturbed_list,
              images_projected_list,
              perturb_magn,
              file, ax_cor_sag, slice):
    Max = 0.25
    Max_dvf = 10
    fig, axs = plt.subplots(len(images_perturbed_list) + 1, 8, figsize=(15, 10))
    for (row, images_perturbed), images_projected in zip(enumerate(images_perturbed_list), images_projected_list):
        [moving_pert, fixed_pert, moving_warped_pert, dvf_pred_pert] = images_perturbed
        [moving_proj, fixed_proj, moving_warped_proj, dvf_pred_proj] = images_projected

        axs[row, 0].imshow(np.flip(acs_slice(to_numpy(fixed_pert), slice, ax_cor_sag).transpose()), cmap='gray',
                           vmin=-0., vmax=1.)
        axs[row, 1].imshow(np.flip(acs_slice(to_numpy(moving_pert), slice, ax_cor_sag).transpose()), cmap='gray',
                           vmin=-0., vmax=1.)
        axs[row, 2].imshow(np.flip(acs_slice(to_numpy(moving_warped_pert), slice, ax_cor_sag).transpose()), cmap='gray',
                           vmin=-0., vmax=1.)
        axs[row, 3].imshow(np.flip(acs_slice(to_numpy(moving_proj), slice, ax_cor_sag).transpose()), cmap='gray',
                           vmin=-0., vmax=1.)
        axs[row, 4].imshow(np.flip(acs_slice(to_numpy(moving_warped_proj), slice, ax_cor_sag).transpose()), cmap='gray',
                           vmin=-0., vmax=1.)
        axs[row, 5].imshow(np.flip(acs_slice(to_numpy(moving_proj - moving_pert), slice, ax_cor_sag).transpose()),
                           cmap='bwr', vmin=-Max, vmax=Max)
        axs[row, 6].imshow(np.flip(acs_slice(to_numpy(dvf_pred_pert)[0, ...], slice, ax_cor_sag).transpose()),
                           cmap='coolwarm',
                           vmin=-Max_dvf, vmax=Max_dvf)
        axs[row, 7].imshow(np.flip(acs_slice(to_numpy(dvf_pred_proj)[0, ...], slice, ax_cor_sag).transpose()),
                           cmap='coolwarm',
                           vmin=-Max_dvf, vmax=Max_dvf)
        axs[row, 0].set_ylabel(perturb_magn[row])

    axs[0, 0].set_title('$F^T$')
    axs[0, 1].set_title('$M^T$')
    axs[0, 2].set_title('$\phi(M^T)$')
    axs[0, 3].set_title('$\hat{M}^T$')
    axs[0, 4].set_title('$\phi(\hat{M}^T)$')
    axs[0, 5].set_title('$\hat{M}^T - M^T$')
    axs[0, 6].set_title('$\phi^T$')
    axs[0, 7].set_title('$\hat{\phi^T}$')

    for axis in axs.flat:
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(file, dpi=220)
    fig.show()


if __name__ == "__main__":
    perturb_type = 'biasfield'
    save_path_images = './output/images_optim'
    perturb_magn = DEFAULT_TRANSFORM_SETTINGS[perturb_type]

    # Load perturbed and projected images
    batch_idx = 0
    images_perturbed_list = []
    images_projected_list = []
    for value in perturb_magn:
        moving_pert, fixed_pert, moving_warped_pert, dvf_pred_pert = read_data(save_path_images, batch_idx,
                                                                               category='perturbed_{}'.format(
                                                                                   value))
        moving_proj, fixed_proj, moving_warped_proj, dvf_pred_proj = read_data(save_path_images, batch_idx,
                                                                               category='projected_{}'.format(
                                                                                   value))
        print('min value={:.02f} - max value=={:.02f}'.format(moving_pert.min(), moving_pert.max()))
        slice = 192 // 2
        ax_cor_sag = 'ax'
        images_perturbed = [moving_pert, fixed_pert, moving_warped_pert, dvf_pred_pert]
        images_projected = [moving_proj, fixed_proj, moving_warped_proj, dvf_pred_proj]
        images_perturbed_list.append(images_perturbed)
        images_projected_list.append(images_projected)

        visualize(images_perturbed_list, images_projected_list,
                  perturb_magn=perturb_magn,
                  file=f"{perturb_type}__batch_{batch_idx}_{ax_cor_sag}.png",
                  ax_cor_sag='ax',
                  slice=slice)