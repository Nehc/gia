import torch
import torch.nn.functional as F

def discount_rewards(p_len:int,gamma=0.998):
  discounted_r = torch.ones(p_len)
  running_add = 1.
  for t in reversed(range(0, p_len-1)):
    running_add = running_add * gamma
    discounted_r[t] = running_add
  return discounted_r

def oneHotProb(targets, probs, num_classes=-1):
  one_hot_tg = F.one_hot(targets, num_classes)
  oneHotProb = one_hot_tg * probs.unsqueeze(-1)
  return oneHotProb
