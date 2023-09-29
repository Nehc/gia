import yaml, torch, os
from urllib.request import urlretrieve
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def download_model(model_name="f16_1024", save_dir="chk_points"):
  if model_name=="f16_1024":
    uid = '8088892a516d4e3baf92'
  elif model_name=="f16_16384":
    uid = 'a7530b09fed84f80a887'
  else: 
    raise NameError("Only f16_1024 and f16_16384 allowed")
  m_path = os.path.join(save_dir, f'vqgan_imagenet_{model_name}')
  if not os.path.exists(m_path):
    os.makedirs(m_path, exist_ok=True) 
  for k,v in {'configs':'model.yaml','ckpts':'last.ckpt'}.items():
    path = os.path.join(m_path,k)
    if not os.path.exists(path):
      os.mkdir(path)
    url = ('https://heibox.uni-heidelberg.de/d/'
           f'{uid}/files/?p=%2F{k}%2F{v}&dl=1')
    f_name = os.path.join(path, v)
    print(url, f_name, sep=" -> ")
    urlretrieve(url, f_name)
     
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

Model = "f16_1024"
download_model(Model)
vq_conf = load_config(f"chk_points/vqgan_imagenet_{Model}/configs/model.yaml", display=False)
vq_model = load_vqgan(vq_conf, ckpt_path=f"chk_points/vqgan_imagenet_{Model}/ckpts/last.ckpt")

