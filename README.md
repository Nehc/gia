# GIA
Project of goal-agent in simalted 3D environment 

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
  b,c,h,w = z.shape
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

```colab
!git clone https://github.com/Nehc/gia.git
!pip install -q -r gia/requirements.txt
```

```python
import torch
from gia.tokenizer import Tokenizer
from gia.config import Thinker_Conf
from gia.dataset import ThinkDataset
from gia.model import Thinker

tkn = Tokenizer(refs_list = ['-','Barrel','Picture','Boxes','Vine box','Market','Gate','Door'],
                acts_list = ['No','Fwd','Bck','Rgt','Lft','Rsf','Lsf','Goal'])

cf = Thinker_Conf(GOAL_IDX=tkn.GOAL_IDX)

all = torch.randint(0, 1024, size=(10, 1000, 51)) # fake data (real comming soon)
ds = ThinkDataset(cf, all, True, use_mask=True, mask_probability=0.9)

th = Thinker(cf,ds,tkn) # or just th = Thinker(cf), if not test predict needed
```
### Since this is a pl-model - train is simlpe: 
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=10, shuffle=True)
trainer = pl.Trainer(gpus=1, max_epochs=5, precision=16)
trainer.fit(model, loader)
```
