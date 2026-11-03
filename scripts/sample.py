"""
A script for sampling from a diffusion model for paired image-to-image translation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.skullbreakerloader import SkullBreakDataset
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))

    diffusion.mode = 'inpainting'
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices

    if args.dataset == 'skullbreaker':
        ds = SkullBreakDataset(args.data_dir, mode='test')
    else:
        print("This dataset can't be synthesized.")

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        if args.dataset == 'skullbreaker':
            batch['image'] = batch['image'].to(dist_util.dev())
            batch['missing'] = batch['missing'].to(dist_util.dev())

            subj = batch['subj'][0]
            cond_type = batch['missing_type'][0]
            print(subj)
            print(cond_type)

            target = batch['image']  # target
            cond_1 = batch['missing']  # condition

            # Conditioning vector for SkullBreaker
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
            cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        else:
            print('There is no defined dataset.')

        noise_name = 'Gaussian'
        if noise_name == 'Gaussian':
            noise = th.randn_like(cond).to(dist_util.dev())
        elif noise_name == 'Bernoulli':
            noise = Bernoulli(probs=0.5).sample(cond.shape).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample[sample < 0.5] = 0
        sample[sample >= 0.5] = 1

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        if len(target.shape) == 5:
            target = target.squeeze(dim=1)


        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(args.output_dir, subj)).mkdir(parents=True, exist_ok=True)

        for i in range(sample.shape[0]):
            print(f'Sample shape: {sample.shape[0]}')
            output_name = os.path.join(args.output_dir, subj, f'{cond_type}_sample.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(args.output_dir, subj, f'{cond_type}_target.nii.gz')
            img = nib.Nifti1Image(target.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=512,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        prod_ct=True,
        contr="",
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

















