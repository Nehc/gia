#@title Процедура Action_on_Decision для сбора датасета
import torch, numpy as np
from mlagents_envs.environment import ActionTuple
from torch import Tensor, argmax

from .vqgan import VQGAN, preprocess_vqgan
from .model import Thinker

class Solver:
  def __init__(self, thinker:Thinker):
    self.thinker = thinker
    self.tkn = thinker.tkn
    self.vq_gan = VQGAN().to(thinker.device)
    self.PAD_IDX = self.tkn.PAD_IDX

  def Action_on_Decision(self, DS, 
                         olds=None,
                         goal=None):
    '''
    TODO:
    при goal=None можно/нужно формировать цель случайно на основании ref_list
    '''
    if not goal:
      ...
    act = ActionTuple()
    tg = argmax(Tensor(DS.obs[1][:,:-2]), dim=1)
    vis = Tensor(np.rollaxis(DS.obs[2], 3, 1))
    x = preprocess_vqgan(vis,False)
    with torch.no_grad():
      z, _, [_, _, ind] = self.vq_gan.encode(x)
      #b,c,h,w = z.shape # 1, 256, 32, 32
      #ind.squeeze_()
      if olds:
        input = torch.cat([olds,self.tkn.encode(tg,ind,self.PAD_IDX)])
      else:
        input = self.tkn.encode(tg,ind,self.PAD_IDX)
      pr = self.thinker.predict(goal, input.to(device))
      res = pr[0].reshape(-1,51).cpu()
      _, a = tkn.decode_one(res[:,-1].squeeze(),False,True)
    acts = np.array(a,np.int8)
    acts = np.expand_dims(acts,axis=1)
    act.add_discrete(acts)
    return act, torch.cat([input[:,:-1],res[:,-1]])
