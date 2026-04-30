import torch
from torch.utils.data import RandomSampler, SequentialSampler
import random

from torch.utils.data.distributed import DistributedSampler
import math

class SeqBatchSampler(object):
  '''
  In case that `shuffle=False, batch_size=4`, 
  for common BatchSampler, it generates: 
  Batch 0: [0, 1, 2, 3]
  Batch 1: [4, 5, 6, 7]
  ...

  For TrunBatchSampler, it generates:
  Batch 0: [(0,a), (1,a), (2,a), (3,a)]
  Batch 1: [(4,b), (5,b), (6,b), (7,b)]
  ...
  where a,b... are random values in trun_range.

  The generated sequences are fed into function `__getitem__` of class Dataset.
  It can be used for dynamic LSTM sequences or dynamic nframe-spectrogram CNN inputs.
  ''' 
  def __init__(self, dataset, trun_range, step=1, shuffle=False, batch_size=1, drop_last=False):
    assert isinstance(trun_range, list) and len(trun_range) == 2
    self.trun_lens = list(range(trun_range[0], trun_range[1]+1, step))
    self.step = step
    self.batch_size = batch_size
    self.drop_last = drop_last    
    if shuffle:
      self.sampler = RandomSampler(dataset)
    else:
      # print('distributed shuffle', shuffle)
      # self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=16, shuffle=shuffle)
      self.sampler = SequentialSampler(dataset)

  def __iter__(self):
    batch = []
    # n = random.choice(self.trun_lens)
    for idx in self.sampler:
      batch.append((idx, self.trun_lens[-1]))
      if len(batch) == self.batch_size:
        yield batch
        batch = []
        # n = random.choice(self.trun_lens)
    if len(batch) > 0 and not self.drop_last:
      yield batch

  def __len__(self):
    if self.drop_last:
      return len(self.sampler) // self.batch_size
    else:
      return (len(self.sampler) + self.batch_size - 1) // self.batch_size
