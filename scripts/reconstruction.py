import sys
sys.path.append("/mnt/lustre/fnzhan/projects/icml2022/taming-transformers/")
import argparse, os, sys, glob, math, time
import torch
import math
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
from streamlit import caching
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from taming.models.cond_transformer import Net2NetTransformer
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

rescale = lambda x: (x + 1.) / 2.


def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def save_img(xstart, fname):
    I = (xstart.clip(0,1)[0]*255).astype(np.uint8)
    Image.fromarray(I).save(fname)



def get_interactive_image(resize=False):
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"])
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        print("upload image shape: {}".format(image.shape))
        img = Image.fromarray(image)
        if resize:
            img = img.resize((256, 256))
        image = np.array(img)
        return image


def single_image_to_torch(x, permute=True):
    assert x is not None, "Please provide an image through the upload function"
    x = np.array(x)
    x = torch.FloatTensor(x/255.*2. - 1.)[None,...]
    if permute:
        x = x.permute(0, 3, 1, 2)
    return x


def pad_to_M(x, M):
    hp = math.ceil(x.shape[2]/M)*M-x.shape[2]
    wp = math.ceil(x.shape[3]/M)*M-x.shape[3]
    x = torch.nn.functional.pad(x, (0,wp,0,hp,0,0,0,0))
    return x

@torch.no_grad()
def run_conditional(model, dsets):
    # if len(dsets.datasets) > 1:
    #     split = st.sidebar.radio("Split", sorted(dsets.datasets.keys()))
    #     dset = dsets.datasets[split]
    # else:
    #     dset = next(iter(dsets.datasets.values())). 10 10 3, 24 /  23.  20 / 10   23  10   0, 1, 2

    model.eval()

    dset = dsets.datasets['validation']
    num_im = 2993
    batch_size = 10
    num_batch = math.ceil(float(num_im) / float(batch_size))
    # print(num_batch)
    # 1/0  300 0, 299.  299  end 2993-1  2990, 2991.  200.

    ind_list = []

    for i in range(0, num_batch):

        # print('****', i, num_batch-1)

        start_idx = batch_size * i
        if i == (num_batch - 1):
            end_idx = num_im
        else:
            end_idx = batch_size * (i + 1)

        indices = list(range(start_idx, end_idx))
        example = default_collate([dset[i] for i in indices])
        # print(len(dset)). range(1990, 1999) 1990,  (1, )
        # print(dset)
        # print('********', example).

        x = model.get_input(example, "image").to(model.device)

        scale_factor = st.sidebar.slider("Scale Factor", min_value=0.5, max_value=4.0, step=0.25, value=1.00)
        if scale_factor != 1.0:
            x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="bicubic")
            # c = torch.nn.functional.interpolate(c, scale_factor=scale_factor, mode="bicubic")

        # quant_z, z_indices = model.encode_to_z(x)
        quant, _, ind = model.encode(x)



        dec = model.decode(quant)
        ims = dec

        for j in range(len(indices)):
            im = ims[j].permute(1,2,0).detach().cpu().numpy()
            im = ((im + 1.0) * 127.5).clip(0, 255)
            im = Image.fromarray(im.astype('uint8'))
            idx = i * batch_size + j
            im.save('results/faces_mixvq_recon/pre/{}.png'.format(idx))

            im = x[j].permute(1, 2, 0).detach().cpu().numpy()
            im = ((im + 1.0) * 127.5).clip(0, 255)
            im = Image.fromarray(im.astype('uint8'))
            idx = i * batch_size + j
            im.save('results/faces_mixvq_recon/gt/{}.png'.format(idx))

            print(idx)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    return parser


def load_model_from_config(config, gpu=True, eval_mode=True):

    # model = Net2NetTransformer(**config.params)
    model = instantiate_from_config(config)

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model_and_dset(config, gpu, eval_mode):

    dsets = get_data(config)   # calls data.config ..

    model = load_model_from_config(config.model, gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    config_path = opt.resume
    config = OmegaConf.load(config_path)
    gpu = True
    #eval_mode = st.sidebar.checkbox("Eval Mode", value=True)
    eval_mode = True

    dsets, model = load_model_and_dset(config, gpu, eval_mode)
    # gs.text(f"Global step: {global_step}")
    run_conditional(model, dsets)
