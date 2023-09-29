import torch
from enum import Enum
import numpy as np

ttypes = Enum('ttypes',['ACT','VIS','REF','SPC'])

class Tokenizer():
  def __init__(self,
               vis_vocab_size=1024,
               ref_vocab_size=16,
               act_vocab_size=8,
               refs_list = [],
               acts_list = [],
               pad_id=0, mask_id=1, 
               goal_id = 1050):

    self.size = 4+vis_vocab_size+ref_vocab_size+act_vocab_size

    self.spc_names = np.array(['PAD','MASK','GOAL'])

    self.PAD_IDX  = pad_id
    self.MASK_IDX  = mask_id
    self.GOAL_IDX = goal_id

    self.vis_vocab_size = vis_vocab_size
    self.ref_vocab_size = ref_vocab_size
    self.act_vocab_size = act_vocab_size

    self.ref_names = np.array(refs_list+['-']*ref_vocab_size)
    self.act_names = np.array(acts_list+['-']*act_vocab_size)

    bias = 3
    self.vis_tokens = bias
    bias += vis_vocab_size
    self.ref_tokens = bias
    bias += ref_vocab_size
    self.act_tokens = bias
    self.UNKN = self.size-1

  
  def encode_one(self, indx, tp:ttypes=None)->int:
    if   tp == ttypes.VIS and torch.all(indx < self.vis_vocab_size):
      return indx+self.vis_tokens
    elif tp == ttypes.REF and torch.all(indx < self.ref_vocab_size):
      return indx+self.ref_tokens
    elif tp == ttypes.ACT and torch.all(indx < self.act_vocab_size):
      return indx+self.act_tokens
    elif (tp == ttypes.SPC or tp == None) and torch.all(indx < 3):
      return indx
    else:
      return self.UNKN

  
  def encode(self, ref, obs, acts):
    ref = self.encode_one(ref,ttypes.REF)
    obs = self.encode_one(obs,ttypes.VIS)
    acts = self.encode_one(acts,ttypes.ACT)

    return torch.cat((ref,obs,acts),dim=-1)

  
  def decode_one(self,indx,r_type=False,r_str=False):
    print(torch.all(indx) >= self.size or torch.all(indx)<0,
          torch.all(indx) >= self.act_tokens,
          torch.all(indx) >= self.ref_tokens,
          torch.all(indx) >= self.vis_tokens)
    if torch.all(indx >= self.size) or torch.all(indx<0):
      r = -100, # Вообще не может быть
      if r_type: r = *r, ttypes.SPC
      if r_str: r = *r, 'UNKN'
      if not r_type and not r_str: r, = r
      return r
    elif torch.all(indx >= self.act_tokens):
      r = indx - self.act_tokens,
      if r_type: r = *r, ttypes.ACT
      if r_str: r = *r, self.act_names[indx - self.act_tokens]
      if not r_type and not r_str: r, = r
      return r
    elif torch.all(indx >= self.ref_tokens):
      r = indx - self.ref_tokens,
      if r_type: r = *r, ttypes.REF
      if r_str: r = *r, self.ref_names[indx - self.ref_tokens]
      if not r_type and not r_str: r, = r
      return r
    elif torch.all(indx >= self.vis_tokens):
      r = indx - self.vis_tokens,
      if r_type: r = *r, ttypes.VIS
      if r_str: r = *r, str(indx - self.vis_tokens)
      if not r_type and not r_str: r, = r
      return r
    else:
      r = indx,
      if r_type: r = *r, ttypes.SPC
      if r_str: r = *r, self.spc_names[indx]
      if not r_type and not r_str: r, = r
      return r

# usage Example:
# tkn = Tokenizer(refs_list = ['-','Barrel','Picture','Boxes','Vine box','Market','Gate','Door'],
#                 acts_list = ['No','Fwd','Bck','Rgt','Lft','Rsf','Lsf','Goal'])
