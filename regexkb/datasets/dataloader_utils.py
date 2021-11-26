import numpy as np
import torch
from torch.utils.data import Dataset


class KBDataset(Dataset):

    def __init__(self, facts, facts_filter, rel_path_ids, query_type, mode, num_entity, neg_sample_count):

        self.query_type = query_type
        self.facts = facts
        self.facts_filter = facts_filter
        self.rel_path_ids = rel_path_ids
        self.num_entity = num_entity
        self.neg_sample_count = neg_sample_count
        self.mode = mode
        self.len = len(self.facts)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = idx % len(self.facts)
        if self.mode == 'train':
            negative_sample = torch.randint(
                self.num_entity, size=(self.neg_sample_count,))
            return (torch.LongTensor(self.facts[idx]),
                    None,
                    negative_sample,
                    torch.tensor(self.rel_path_ids[idx]),
                    self.query_type)
        else:
            return (torch.LongTensor(self.facts[idx]),
                    self.facts_filter[idx],
                    None,
                    torch.tensor(self.rel_path_ids[idx]),
                    self.query_type)


class RandomSampler():
    def __init__(self, max):
        self.max = max

    def __iter__(self):
        return self

    def __next__(self):
        return np.random.randint(0, self.max)


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.iterators = [RandomSampler(len(cur_dataset)).__iter__()
                          for cur_dataset in dataset.datasets]
        self.offset = [0] + self.dataset.cumulative_sizes[:-1]
        self.current_dataset_idx = 0
        self.count = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if (self.count != 0) and (self.count % self.batch_size == 0):
            self.current_dataset_idx = (
                self.current_dataset_idx + 1) % self.number_of_datasets
        self.count += 1
        return self.offset[self.current_dataset_idx] + \
            self.iterators[self.current_dataset_idx].__next__()


def custom_collate_fn(samples):

    facts = torch.stack([_[0] for _ in samples], dim=0)

    if samples[0][1] is not None:
        facts_filter = [_[1] for _ in samples]
        lens = [len(_) for _ in facts_filter]
        max_lens = max(lens)
        facts_filter = [np.pad(x, (0, max_lens - len(x)), 'edge')
                        for x in facts_filter]
        facts_filter = torch.from_numpy(np.array(facts_filter))
    else:
        facts_filter = None

    if samples[0][2] is not None:
        negative_sample = torch.stack([_[2] for _ in samples], dim=0)
    else:
        negative_sample = None

    rel_path_ids = torch.stack([_[3] for _ in samples], dim=0)
    query_type = [_[4] for _ in samples]
    assert all(_ == query_type[0] for _ in query_type),\
        "Batch contains data from different query types"

    if facts_filter is None:
        return facts, negative_sample, rel_path_ids, query_type[0]
    elif negative_sample is None:
        return facts, facts_filter, rel_path_ids, query_type[0]
    else:
        assert False, "Either of facts_filter and negative sample should be None"
