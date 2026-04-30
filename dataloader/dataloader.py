import torch
from torch.utils.data import DataLoader
from dataloader.sampler import TrunBatchSampler, SeqBatchSampler

class InfDataLoader(DataLoader):
    def __init__(self, dataset, trun_range=[64000, 64000], step=1,
                 shuffle=False, batch_size=1, num_workers=1, drop_last=False):
        self.dataset = dataset
        self.batch_sampler = SeqBatchSampler(self.dataset,
                                              trun_range=trun_range,
                                              step=step,
                                              shuffle=shuffle,
                                              batch_size=batch_size,
                                              drop_last=drop_last)
        super().__init__(self.dataset,
                         collate_fn=self.collate_fn,
                         batch_sampler=self.batch_sampler,
                         num_workers=num_workers,
                         pin_memory=True)


    def collate_fn(self, batch):
        batch = list(zip(*batch))
        mix_utt, mix_wav, ilens, lip_emb, expr_emb = batch
        mix_wav = torch.stack(mix_wav, dim=0)
        ilens = torch.stack(ilens, dim=0)
        lip_emb = torch.stack(lip_emb, dim=0)
        expr_emb = torch.stack(expr_emb, dim=0)
        return mix_utt, mix_wav,  ilens, lip_emb, expr_emb