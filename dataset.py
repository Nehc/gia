#@title Класс Датасета
import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import Tensor
from random import random

class ThinkDataset(Dataset):
  def __init__(s, th_conf, perception:Tensor,
               with_ds=False,max_tg_len=100,
               GOAL_IDX=None, MASK_IDX=0,
               use_mask=False, mask_probability=0.5):
    '''
    на входе у нас тензор наблюдений агентов shape: num_agents, steps, frame_len
    где первое - число независимых агентов, steps - число шагов в каждом наблюдении,
    frame_len - размера одного кадра наблюдения, включая токен дествия агента

    Датасет берет все возможные фреймы из последовательностей, объявляет их
    целью (goal), и строит к ним переходы из других фреймов, удаленных от цели
    до max_tg_len включительно. Т.к. чаще всего max_tg_len заведомо больше
    максимальной длины контекста max_percept_len, то применяется дисконтирование,
    задающее степень удаленности от цели. Этот трюк в теории так же должен
    привести к семплингу более отптимальных траекторий.
    '''
    super(ThinkDataset, s).__init__()

    s.with_ds = with_ds
    s.MASK_IDX = MASK_IDX
    s.use_mask = use_mask
    s.mask_probability = mask_probability

    bsize = perception.shape[0]
    discont = discount_rewards(max_tg_len)
    datas, goals, disconts = [],[],[]

    if th_conf.GOAL_IDX and not GOAL_IDX:
      GOAL_IDX = th_conf.GOAL_IDX

    if len(perception.shape)<3:
      perception.unsqueeze(0)

    # percaption shape = batch (10), len (1000), frame_len (51)

    for i in tqdm.trange(2, perception.shape[-2]+1):
      # проходим всю входную последовательность окном max_tg_len
      sub_percept = perception[:,:i][:,-max_tg_len:]
      if GOAL_IDX: # Если нужно заменяем токен цели
        sub_percept = sub_percept.clone()
        sub_percept[:,-1,-1] = GOAL_IDX
      sp_len = sub_percept.shape[-2]
      # но с учетом того, что в самом начале кусочки короче
      if sp_len <= th_conf.max_percept_len:
        goals.append(sub_percept[:,-1])
        if sp_len < th_conf.max_percept_len:
          sub_percept = F.pad(sub_percept, # их нужно паддингом добить
                              (0,0,0,th_conf.max_percept_len-sp_len,0,0),
                              value=th_conf.PAD_IDX)
        datas.append(sub_percept)
        disconts.append(F.pad(discont[-sp_len:],
                              (0,th_conf.max_percept_len-sp_len),
                              value=th_conf.PAD_IDX).repeat(bsize,1))
      else: # а внутри окна max_tg_len - бежим окном max_percept_len
        for j in range(sp_len-th_conf.max_percept_len+1):
          goals.append(sub_percept[:,-1]) #цель в конце окна max_tg_len
          if j==0: # берем последовательость и дисконт сначала с хвоста
            datas.append(sub_percept[:,-th_conf.max_percept_len:])
            disconts.append(discont[-th_conf.max_percept_len:].repeat(bsize,1))
          else: # и далее, пока все не пройдем
            datas.append(sub_percept[:,-(j+th_conf.max_percept_len):-j])
            disconts.append(discont[-(j+th_conf.max_percept_len):-j].repeat(bsize,1))
    # в конце стакаем все в один большой тензор, вернее три тензора...
    s.goals = torch.cat(goals)
    s.datas = torch.cat(datas)
    s.disconts = torch.cat(disconts)

  def mask_tokens(s, tensor, kip_first = True, random_mask=True):
    """
    Заменяет случайное количество токенов на маску в тензоре tensor.
    :param tensor: 3D-тензор PyTorch
    :param MASK_IDX: берем из параметров датасета
    :param mask_probability: доля маскируемых токенов (float, где 0 - вообще без маски,
                                                                  1 - все токены заменяются на MASK)
    :return: тензор того же размера, в котором случайное число токенов заменено на маску
    """
    # (не)проверяем корректность входных данных
    #assert tensor.dim() == 3, "The input tensor should be 3D"
    #assert 0.0 <= mask_probability <= 1.0, "mask_probability should be in the range [0, 1]"
    # вычисляем число маскируемых токенов
    probability = s.mask_probability * random() if random_mask else s.mask_probability
    num_masked_tokens = int(round(tensor.numel() * probability))
    # создаем маску для замены токенов на маску
    mask = torch.zeros_like(tensor).byte()
    mask_flat = mask.view(-1)
    mask_flat[:num_masked_tokens] = 1
    # перемешиваем маску, чтобы случайным образом выбирать токены для маскирования
    mask_flat=mask_flat[torch.randperm(len(mask_flat))]
    if kip_first:
      mask_flat = mask_flat.reshape(-1, tensor.shape[-1])
      mask_flat[:,0]=0
    mask_flat[:,-1]=0
    mask = mask_flat.reshape(tensor.shape)
     # mask[mask.shape[:-1],1] = 0
    # создаем новый тензор, заменяя токены на маску с помощью маски
    masked_tensor = tensor.clone()
    masked_tensor[mask.bool()] = s.MASK_IDX
    return masked_tensor


  def __len__(s):
    return len(s.datas)

  def __getitem__(s, idx):
    sz = s.datas[idx].size()
    goals = s.mask_tokens(s.goals[idx]) if s.use_mask else s.goals[idx]
    if s.with_ds:
      return goals, s.disconts[idx], s.datas[idx].reshape(-1)
    else:
      return goals, s.datas[idx].reshape(-1)

# usage Example:
# all.shape = torch.Size([10, 1000, 51]) : num_agents, steps, frame_len
# ds = ThinkDataset(cf, all, True, use_mask=True, mask_probability=0.9)
# loader = DataLoader(ds, batch_size=10, shuffle=True)
