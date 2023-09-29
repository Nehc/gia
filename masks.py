#@title Маски
import torch 
from torch import Tensor

'''
TODO:
Хорошо бы разобраться, почему не работает percept2goal_mask:
созданная маска вроде бы соответствует, но при ее применении
выход превращается в nan
'''
def square_subsequent_mask(percept_len, frame_size, boolean = True):
    mask = (torch.triu(torch.ones((percept_len//frame_size,
                                percept_len//frame_size))) == 1).transpose(0, 1)
    mask = mask.repeat_interleave(frame_size,1).repeat_interleave(frame_size,0)
    if boolean:
      mask = torch.logical_not(mask)
    else:
      mask = mask.float().masked_fill(mask == 0,
                                    float('-inf')).masked_fill(mask == 1,
                                                               float(0.0))
    return mask

def percept2goal_mask(goal_len, percept_len, frame_size, boolean = True):
    mask = torch.zeros(frame_size,goal_len)
    mask[-1,:] = mask[-1,:]+1
    #if not boolean: mask[-1,0] = mask[-1,0]+1
    mask = mask.repeat(percept_len//frame_size,1)
    if boolean:
      mask = torch.logical_not(mask)
    else:
      mask = mask.float().masked_fill(mask == 0,
                                    float('-inf')).masked_fill(mask == 1,
                                          #float(0.0)).masked_fill(mask == 2,
                                                                    float(0.0))
    return mask

def create_mask(goal:Tensor,
                perception:Tensor,
                PAD_IDX:int,
                frame_len=1,
                add_ref=False):
    device = goal.device
    goal_len = goal.shape[1]
    percept_len = perception.shape[1]
    goal_mask = torch.zeros((goal_len, goal_len),device=device).type(torch.bool)
    percept_mask = square_subsequent_mask(percept_len,frame_len).to(device)
    action_mask = percept2goal_mask(goal_len, percept_len, frame_len, not add_ref).to(device)
    goal_padding_mask = (goal == PAD_IDX).to(device) #.transpose(0, 1) # batch_first = True
    percept_padding_mask = (perception == PAD_IDX).to(device) #.transpose(0, 1)
    return goal_mask, percept_mask, action_mask, goal_padding_mask, percept_padding_mask
