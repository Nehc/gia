#@title Config
from dataclasses import dataclass

@dataclass
class Thinker_Conf:
      frame_size: int = 51
      max_goal_len: int = 1
      max_percept_len: int = 20
      emb_size: int = 512
      nhead: int = 16
      num_encoder_layers: int = 6
      num_decoder_layers: int = 12
      dim_feedforward: int = 2048
      vocab_size: int = 1052
      act_space_bias:int = 1043
      act_space_len:int = 8
      PAD_IDX: int = 0
      GOAL_IDX: int = None
      dropout: float = 0.1
      additive_ref = True

# usage Example:
# cf = Thinker_Conf(GOAL_IDX=tkn.GOAL_IDX)
