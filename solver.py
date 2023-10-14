#@title Процедура Action_on_Decision для сбора датасета
import torch, numpy as np
from mlagents_envs.environment import ActionTuple
from torch import LongTensor, argmax

from .vqgan import VQGAN, preprocess_vqgan
from .model import Thinker

class Solver:
  def __init__(self, thinker:Thinker):
    self.device = thinker.device
    self.thinker = thinker   # думалка
    self.fr_size = thinker.config.frame_size
    self.max_len = thinker.config.max_percept_len
    self.tkn = thinker.tkn   # токенайзер
    self.PAD_IDX = self.tkn.PAD_IDX
    self.vq_gan = VQGAN().to(self.device) # гляделка
    self.history, self.goal, self.last_acts = None, None, None

  def Action_on_Decision(self, DS, goal=None):
    act = ActionTuple()
    if goal is not None: self.goal = goal 
    tg = argmax(LongTensor(DS.obs[1][:,:-2]), dim=1) # Спорно... вообще это метка класса (reference), 
                                                     # если он в поле зрения. Не должны ли мы его
                                                     # маскировать на инфренсе? Хотя бы через раз
    count = tg.shape[0] # определяем число агентов
    tg = tg.unsqueeze(1).to(self.device)
    vis = LongTensor(np.rollaxis(DS.obs[2], 3, 1))   # Получаем зрение...
    x = preprocess_vqgan(vis,False)                  # и готовим для гляделки 
    with torch.no_grad():
      _, _, [_, _, ind] = self.vq_gan.encode(x.to(self.device)) # Гляделка!
      ind = ind.squeeze().reshape(count,-1)
      pd = LongTensor([self.PAD_IDX]).repeat(count,1).to(self.device) # Cпорно! типа паддим action
                                                     # Вообще, я думаю надо изменить сдвиг и вообще все так, что бы 
                                                     # Action шел прям первым в кадре... Может просто местами поменять?
                                                     # Хотя... Учитывая, что кадр мы предсказываем целиком - без разницы!     
      if type(self.history) == torch.Tensor: # Если история есть 
        input = torch.cat([self.history[:,self.max_len:,:], # Ее тоже на вход
                           self.tkn.encode(tg,ind,pd).unsqueeze(1)],dim=1)
      else: # А если нету - просто собираем из reference, vis и pad вместо aсtion 
        input = self.tkn.encode(tg,ind, pd).unsqueeze(1)
      if (self.goal is None # Вот оно! :) Тут собираем рандомную цель с маскированым vis
          or np.any(self.last_acts == # Если цель не задана, или задана и досигнута 
                    self.tkn.GOAL_IDX-self.tkn.act_tokens)): # хотя бы одним из агентов
        masked = torch.torch.ones_like(input[:,-1,1:-1], device='cpu') * self.tkn.MASK_IDX
        randgoal = torch.randint(self.tkn.ref_tokens, self.tkn.act_tokens, (count,1))
        G = torch.ones(count,1)*self.tkn.GOAL_IDX
        self.goal = torch.cat([randgoal,masked,G],dim=1).to(torch.int)
      pr = self.thinker(self.goal.to(self.device),       # И наконец засылаем в "думалку"
                        input.reshape(count,-1).to(self.device))
      res = pr[:,-1:,self.tkn.act_tokens:].argmax(-1) # Можно как-нить и похитрее семплить! 
    rand_acts = torch.randint_like(res, 0, self.tkn.act_vocab_size-1) # Простые рандомные acts... Хотя... Не такие и простые!
                                                                 # Так получилось, что среди Acts есть GOAL, и вот его мы
                                                                 # не должны получать рандомно! Поэтому -1, ибо GOAL - крайний!
    res[res==0] = rand_acts[res==0]  # Если пока тупенький и act=0, делаем random
    acts = np.array(res[:,-1].cpu(),np.int8)
    acts = np.expand_dims(acts,axis=1)
    self.last_acts = acts 
    act.add_discrete(acts)
    history = torch.cat([input.reshape(count,-1)[:,:-1],res+self.tkn.act_tokens],dim=-1)
    self.history = history.reshape(count,-1,self.fr_size)  
    return act, self.history[:,-1:,:]
