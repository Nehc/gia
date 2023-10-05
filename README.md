# GIA
Project of goal-agent in simulated 3D environment 

## VQ-GAN (for tokenize visual perception)
https://github.com/CompVis/taming-transformers
<details>
  <summary>Details</summary>  
  
```python
import torch
from PIL import Image
from gia.vqgan import VQGAN, preprocess_vqgan
import numpy as np

vq_gan = VQGAN()
img = Image.open('photo.jpg').convert("RGB")
x = preprocess_vqgan(np.expand_dims(np.array(img)/255,0))
with torch.no_grad():
  z, _, [_, _, ind] = vq_gan.encode(x)
  b,c,h,w = z.shape # 1, 256, 32, 32
  ind.squeeze_()
```
source image is 512x512x3. **ind** is 1024 (32x32 of 16x16 tiles)

```python
from gia.vqgan import custom_to_pil

with torch.no_grad():
  nz = vq_gan.quantize.get_codebook_entry(ind, (b,h,w,c))
  rec = vq_gan.decode(nz).detach().cpu()
  rec.squeeze_()

np_img = np.rollaxis(rec.numpy(),0,3)
img = custom_to_pil(np_img)
```
source and reconstructed image:

![image](https://github.com/Nehc/gia/assets/8426195/07d596ca-02c7-4f4a-a99c-e86fa8302bdb)

</details>

## Thinker (goal-agent model)
![image](https://github.com/Nehc/gia/assets/8426195/a92f3088-0f7e-41ee-859c-0dd0e375b7d7)

#### 51 token on **Frame** of observation:
- **R** - reference int index from refs_list, 
- **Visual** - 49 visual vq_gan tokens (i am use 112x122 input, that give 7x7 tiles), 
- **A** - action tokens from acts_list, 
- **G** - goal, tkn.GOAL_IDX.

```colab
!git clone https://github.com/Nehc/gia.git
!pip install -q -r gia/requirements.txt
```
### Environment init
```Python
from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name=None, # None if local UnityEnvironment avalible  
                       #seed=SEED,     # if use SEED
                       side_channels=[])
env.reset()
env_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[env_name]
print(env_name, spec)
```
### Inference
```python
import tqdm, torch, numpy as np

from gia.tokenizer import Thinkenizer
from gia.config import Thinker_Conf
from gia.model import Thinker
from gia.solver import Solver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tkn = Thinkenizer(refs_list = ['-','Barrel','Picture','Boxes','Vine box','Market','Gate','Door'],
                  acts_list = ['No','Fwd','Bck','Rgt','Lft','Rsf','Lsf','Goal'])

cf = Thinker_Conf(GOAL_IDX=tkn.GOAL_IDX)

th = (Thinker.load_from_checkpoint("last.ckpt",
              cf, tkn=tkn) if path.exists("last.ckpt") else
      Thinker(cf, tkn=tkn)).to(device)

s = Solver(th)

obs = None; datas=[]

for i in tqdm.trange(1000):
  decision_steps, _ = env.get_steps(env_name)
  act, obs = s.Action_on_Decision(decision_steps,obs[-100:,:])
  datas.append(torch.unsqueeze(obs, 1))
  env.set_actions(env_name,act)
  env.step()

datas = torch.cat(datas,dim=1)
torch.save(datas, 'dataset.pt')
```

### Train!: 
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from gia.config import Thinker_Conf
from gia.dataset import ThinkDataset
from gia.model import Thinker

datas = torch.load('dataset.pt')
tkn = Thinkenizer(refs_list = ['-','Barrel','Picture','Boxes','Vine box','Market','Gate','Door'],
                  acts_list = ['No','Fwd','Bck','Rgt','Lft','Rsf','Lsf','Goal'])
cf = Thinker_Conf(GOAL_IDX=tkn.GOAL_IDX)
ds = ThinkDataset(cf, datas, True, use_mask=True, mask_probability=0.9)
th = (Thinker.load_from_checkpoint("last.ckpt",
              cf, ds, tkn) if path.exists("last.ckpt") else
      Thinker(cf, ds, tkn)).to(device)

loader = DataLoader(ds, batch_size=10, shuffle=True)
trainer = pl.Trainer(gpus=1, max_epochs=5, precision=16)
trainer.fit(th, loader)

trainer.save_checkpoint("last.ckpt")
```
