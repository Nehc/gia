#@title VQ-GAN procedures { vertical-output: true }
Model = "f16_1024"
import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x, roll=True):
  x = 2.*x - 1.
  if roll:
    x = np.rollaxis(x,3,1)
  x = torch.Tensor(x)
  return x

def preprocess(x, permt=True):
  if permt:
    x = x.permute(0,2,3,1).numpy()
  x = np.clip(x, -1., 1.)
  x = (x + 1.)/2.
  return x

def custom_to_pil(x):
  x = np.clip(x, -1., 1.)
  x = (x + 1.)/2.
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

vq_conf = load_config(f"chk_points/vqgan_imagenet_{Model}/configs/model.yaml", display=False)
vq_model = load_vqgan(vq_conf, ckpt_path=f"chk_points/vqgan_imagenet_{Model}/ckpts/last.ckpt")

