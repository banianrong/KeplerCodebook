import sys
sys.path.append("/mnt/lustre/fnzhan/projects/icml2022/taming-transformers/")
import argparse, os, sys, glob, math, time
import torch
import torchvision
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

    idx = torch.range(0, 1023)

    quant = model.codebook_visual(provided_idx=idx)
    dec = model.decode(quant)
    ims = dec  # 1024, 3, 16, 16

    im_batch = torchvision.utils.make_grid(ims, nrow=32, padding=2)
    im = im_batch.permute(1, 2, 0).detach().cpu().numpy()
    im = ((im + 1.0) * 127.5).clip(0, 255)
    im = Image.fromarray(im.astype('uint8'))
    im.save('results/ade20k_mixvq2/codebook.png')




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
