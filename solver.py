#@title Процедура Action_on_Decision для сбора датасета
import torch, numpy as np
from mlagents_envs.environment import ActionTuple
from torch import LongTensor, argmax

from gia.vqgan import VQGAN, preprocess_vqgan
from gia.model import Thinker

class Solver:
  def __init__(self, thinker:Thinker):
    self.device = thinker.device
    self.thinker = thinker
    self.fr_size = thinker.config.frame_size
    self.max_len = thinker.config.max_percept_len
    self.tkn = thinker.tkn
    self.PAD_IDX = self.tkn.PAD_IDX
    self.vq_gan = VQGAN().to(self.device)
    self.history = None

  def Action_on_Decision(self, DS,
                         goal=None):
    act = ActionTuple()
    tg = argmax(LongTensor(DS.obs[1][:,:-2]), dim=1)
    count = tg.shape[0]
    tg = tg.unsqueeze(1).to(self.device)
    vis = LongTensor(np.rollaxis(DS.obs[2], 3, 1))
    x = preprocess_vqgan(vis,False)
    with torch.no_grad():
      _, _, [_, _, ind] = self.vq_gan.encode(x.to(self.device))
      ind = ind.squeeze().reshape(count,-1)
      pd = LongTensor([self.PAD_IDX]).repeat(count,1).to(self.device)
      if type(self.history) == torch.Tensor:
        input = torch.cat([self.history[:,self.max_len:,:],
                           self.tkn.encode(tg,ind,pd).unsqueeze(1)],dim=1)
      else:
        input = self.tkn.encode(tg,ind, pd).unsqueeze(1)
      if goal==None:
        masked = torch.torch.ones_like(input[:,-1,1:-1]) * self.tkn.MASK_IDX
        randgoal = torch.randint(self.tkn.ref_tokens,
                                 self.tkn.act_tokens,
                                (count,1))
        G = torch.ones(count,1)*self.tkn.GOAL_IDX
        goal = torch.cat([randgoal,masked,G])
      pr = self.thinker(goal.to(self.device),
                        input.reshape(count,-1).to(self.device))
      res = pr[:,-1:,self.tkn.act_tokens:].argmax(-1)
    rand_acts = torch.randint_like(res, 0, pr.shape[1])
    res[res==0] = rand_acts[res==0]  
    acts = np.array(res[:,-1].cpu(),np.int8)
    acts = np.expand_dims(acts,axis=1)
    act.add_discrete(acts)
    history = torch.cat([input.reshape(count,-1)[:,:-1],res+self.tkn.act_tokens],dim=-1)
    self.history = history.reshape(count,-1,self.fr_size)  
    return act, self.history[:,-1:,:]
