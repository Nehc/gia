# gia
Project of goal-agent in simalted 3D environment 

![image](https://github.com/Nehc/gia/assets/8426195/a92f3088-0f7e-41ee-859c-0dd0e375b7d7)

```colab
!git clone https://github.com/Nehc/gia.git
!pip install -q -r gia/requirements.txt
%cd gia
```

```python
import torch
from tokenizer import Tokenizer
from config import Thinker_Conf
from dataset import ThinkDataset
from model import Thinker

tkn = Tokenizer(refs_list = ['-','Barrel','Picture','Boxes','Vine box','Market','Gate','Door'],
                acts_list = ['No','Fwd','Bck','Rgt','Lft','Rsf','Lsf','Goal'])
cf = Thinker_Conf(GOAL_IDX=tkn.GOAL_IDX)
all = torch.linspace(0, 1052, steps=10*1000*51).reshape(10, 1000, 51) # fake data, real comming soon
ds = ThinkDataset(cf, all, True, use_mask=True, mask_probability=0.9)

th =  Thinker(cf,ds,tkn) 
```
