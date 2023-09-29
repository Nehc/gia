#@title Lightning-модель
import torch
from torch import nn, Tensor
from numpy.core.numeric import zeros_like
import pytorch_lightning as pl
from typing import Optional

from .config import Thinker_Conf
from .tokenizer import Tokenizer
from .dataset import ThinkDataset
from .masks import create_mask, square_subsequent_mask
from .rewards import oneHotProb 

class Thinker(pl.LightningModule):
  def __init__(self,
               conf: Thinker_Conf,
                 ds: Optional[ThinkDataset] = None,
                tkn: Optional[Tokenizer] = None):
    super(Thinker, self).__init__()
    self.config, self.ds, self.tkn = conf, ds, tkn
    self.token_emb = nn.Embedding(conf.vocab_size, conf.emb_size)
    self.frame_pos = nn.Embedding(conf.frame_size, conf.emb_size)
    self.time_embd = nn.Embedding(conf.max_percept_len, conf.emb_size)
    self.transformer = nn.Transformer(
        d_model = conf.emb_size,
        nhead = conf.nhead,
        num_encoder_layers = conf.num_encoder_layers,
        num_decoder_layers = conf.num_decoder_layers,
        dim_feedforward = conf.dim_feedforward,
        dropout=conf.dropout,
        batch_first = True,
        )
    #self.sq_seq_mask = square_subsequent_mask(conf.percept_len,conf.frame_size)
    self.head = nn.Linear(conf.emb_size, conf.vocab_size, bias=False)
    self.token_emb.weight = self.head.weight
    self.base_loss = nn.CrossEntropyLoss(reduction='sum',ignore_index=conf.PAD_IDX)
    self.act_loss = nn.CrossEntropyLoss(reduction='sum')

    # init all weights
    self.apply(self._init_weights)

    # report number of parameters
    n_params = sum(p.numel() for p in self.parameters())
    print("number of parameters: %.2fM" % (n_params/1e6,))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
      torch.nn.init.ones_(module.weight)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)

  
  def frame_time_pos_emb(self, p:Tensor):
    '''
    К каждому токену последовательности мы применяем три эмбеддинга:
    собственный эмбеддинг токена по позиции в словаре (разные для референса,
    визуала и собственных действий, что обеспечивается при кодировании
    токенайзером), позиционный эмбеддинг от позиции в кадре (можно считать
    номером канала), и позиционный эмбеддинг номера кадра, просто по порядку
    '''
    embd = self.token_emb(p) # собственный эмбеддинг токена
    p_len = p.shape[-1]
    frm_len = self.config.frame_size
    # frame_pos - каждому токену ставится в соответствие позиция в кадре
    frame_pos = torch.arange(0, frm_len,
                             dtype=torch.long,
                             device=p.device,
                             #requires_grad=True,
                             ).unsqueeze(0)
    frame_pos = frame_pos.repeat(p_len//frm_len,1).view(-1,p_len)
    embd += self.frame_pos(frame_pos) # добавляем эмбединг по позиции
    if p_len//frm_len>1: # если кадров больше одного... А надо ли? (!!?!!)
      # time_pos - номер кадра, опять же для каждого токена
      time_pos = torch.arange(0, p_len//frm_len,
                              dtype=torch.long,
                              device=p.device,
                              #requires_grad=True,
                              ).unsqueeze(0)
      time_pos = time_pos.repeat_interleave(frm_len,1)
      embd += self.time_embd(time_pos) # добавляем номера кадров
    return embd

  
  def forward(self,
              goal: Tensor,
              percept: Tensor,
              ):
    goal_mask, percept_mask, action_mask, \
    goal_padding_mask, perception_padding_mask = create_mask(goal, percept,
                                                  self.config.PAD_IDX,
                                                  self.config.frame_size,
                                                  self.config.additive_ref)
    goal_e = self.frame_time_pos_emb(goal)
    percept_e = self.frame_time_pos_emb(percept)
    outs = self.transformer(goal_e, percept_e,
                            goal_mask, percept_mask, None, #action_mask,
                            goal_padding_mask, perception_padding_mask,
                            goal_padding_mask
                            )
    s = self.head(outs)
    return s

  
  def encode(self, goal: Tensor, goal_mask: Tensor):
    return self.transformer.encoder(self.frame_time_pos_emb(goal), goal_mask)

  
  def decode(self, percept: Tensor, memory: Tensor,
             percept_mask: Tensor, action_mask: Tensor = None):
    return self.transformer.decoder(self.frame_time_pos_emb(percept),
                                    memory, percept_mask, action_mask)

  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, #0.0001, 3e-4,
                                 betas=(0.9, 0.98), eps=1e-9)
    #optimizer = torch.optim.Adam(self.parameters(), lr=1e-9)
    return optimizer

  
  # function to generate output sequence using greedy algorithm
  def greedy_decode(self, goal, goal_mask, max_gen, percept):
    finded, log = False,[]
    with torch.no_grad():
      memory = self.encode(goal, goal_mask)
      ys = percept
      max_gen -= percept.shape[-1]//self.config.frame_size
      max_len = self.config.max_percept_len*self.config.frame_size
      for i in range(max_gen):
        tgt_mask = (square_subsequent_mask(ys[:,-max_len:]
                    .size(-1),self.config.frame_size)
                    .type(torch.bool)).to(percept.device)
        out = self.decode(ys[:,-max_len:], memory, tgt_mask)
        prob_s = self.head(out[:, -self.config.frame_size:])
        _, next_frame = torch.max(prob_s, dim=-1)
        ys = torch.cat([ys, next_frame], dim=-1)
        if next_frame[:,-1] == self.config.GOAL_IDX:
          finded = True
          break
    return ys, finded

  
  def predict(self, goal, percept, max_len=100):
    self.eval()
    if len(goal.shape)<2:
      goal = goal.unsqueeze(0)
    if len(percept.shape)<2:
      percept = percept.unsqueeze(0)
    num_tokens = goal.shape[-1]
    mask = torch.zeros(num_tokens, num_tokens, device=goal.device).type(torch.bool)
    percept_tokens, f = self.greedy_decode(goal, mask, max_len, percept) #.flatten()
    return percept_tokens, f

  
  def step(self, batch):
    f_size = self.config.frame_size
    with_ds = len(batch)>2
    if len(batch) == 3:
      goals, disconts, percepts = batch
    elif len(batch) == 2:
      goals, percepts = batch
    elif len(batch) == 1:
      goals, percepts = batch[:,-self.config.frame_size:],batch
    else:
      raise ValueError('Batch shape not good!')

    in_p  = percepts[:,:-f_size] # Все, кроме последнего фрема
    out_p = percepts[:, f_size:] # Все, начиная со второго фрейма
    #---------------------------------------------------------------------------
    pr_p = self(goals,in_p) # shape: batch, tokens len, logits(vocab_size)
    #---------------------------------------------------------------------------
    loss = self.base_loss(pr_p.reshape(-1, pr_p.shape[-1]), out_p.reshape(-1))
    #---------------------------------------------------------------------------
    if with_ds:
      #           - - - - - - - - - - -
      a_bias = self.config.act_space_bias # индексы токенов действия отсюда
      a_len = self.config.act_space_len # и до сюда
      #           - - - - - - - - - - -
      pr_a = pr_p.reshape(-1, pr_p.shape[-1]) # shape: All, logit
      # Берем только последний токен фрейма (где действие),
      # И только часть логитов, соответствующие кодировке действий.
      pr_a = pr_a[f_size-1::f_size,a_bias:a_bias+a_len]
      # Для ground truth берем так же только хвосты фреймов,
      # И делам поправку на значения индексов (что бы классов поменьше)
      out_a = torch.clamp(out_p[:,f_size-1::f_size] - a_bias,0)
      out_a = oneHotProb(out_a, disconts[:,1:], num_classes=a_len)
      # pr_a У нас уже flatten, а out_a прям при передаче в лосс
      loss += self.act_loss(pr_a, out_a.reshape(-1, out_a.shape[-1]))
    #---------------------------------------------------------------------------
    return loss
  

  def training_step(self, batch, batch_idx: int):
    loss = self.step(batch)
    self.log('train_loss', loss)
    return loss
  

  def validation_step(self, batch, batch_idx: int):
    loss = self.step(batch, batch_idx)
    self.log('val_loss', loss)
  
  
  def training_epoch_end(self, training_step_outputs):
    if self.ds and self.tkn:
      g,_,d = self.ds[0]
      with torch.no_grad():
        pr = self.predict(g.to(self.device), d[:self.config.frame_size].to(self.device))
      res = pr[0].reshape(-1,51).cpu()
      _, r = self.tkn.decode_one(res[:,0].squeeze(),False,True)
      fr = self.tkn.decode_one(res[:,1:-1].squeeze()).reshape(-1,49)
      _, a = self.tkn.decode_one(res[:,-1].squeeze(),False,True)
      print(r,a,fr, sep='\n')
