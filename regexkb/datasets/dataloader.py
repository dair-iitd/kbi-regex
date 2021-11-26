import pytorch_lightning as pl
import os.path as path
import os
import zipfile
import pickle

from .data_utils import get_maps, get_graph, process_kbc, process_regex
from .dataloader_utils import KBDataset, BatchSchedulerSampler, custom_collate_fn
from torch.utils.data import DataLoader, ConcatDataset, dataset


class DataModule(pl.LightningDataModule):

    def __init__(self, cachedir, args):
        super().__init__()
        self.cachedir = cachedir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.eval_batch_size = args.eval_batch_size
        assert args.dataset in [
            'fb15k', 'wiki', 'wiki_v2'], "Dataset should be either fb15k or wiki"
        self.dataset = args.dataset
        self.data = None

    def prepare_data(self):

        assert path.exists(
            self.cachedir + '/raw-datasets/' + self.dataset + '_regex_data.zip')

        if not path.exists(self.cachedir + '/unpacked-datasets/' + self.dataset):
            os.makedirs(self.cachedir +
                        '/unpacked-datasets/' + self.dataset, exist_ok=True)
            with zipfile.ZipFile(self.cachedir + '/raw-datasets/' + self.dataset + '_regex_data.zip', 'r') as zip_ref:
                zip_ref.extractall(
                    self.cachedir + '/unpacked-datasets/' + self.dataset)

        if not path.exists(self.cachedir + '/encoded-datasets/' + self.dataset + '_regex.pkl'):
            os.makedirs(self.cachedir + '/encoded-datasets', exist_ok=True)

            entity_map, relation_map = get_maps(
                self.cachedir + '/unpacked-datasets/' + self.dataset)
            knowledge_graph = get_graph(
                self.cachedir + '/unpacked-datasets/' + self.dataset, entity_map, relation_map)

            data = {}

            data['entity_map'] = entity_map
            data['relation_map'] = relation_map

            for query_type in range(0, 22):
                data[query_type] = {}

                for mode in ['train', 'valid', 'test']:
                    data[query_type][mode] = {}

                    if query_type == 0:
                        facts, facts_filter, rel_path_ids = process_kbc(
                            self.cachedir + '/unpacked-datasets/' + self.dataset, entity_map, relation_map, knowledge_graph, mode)
                    else:
                        facts, facts_filter, rel_path_ids = process_regex(
                            self.cachedir + '/unpacked-datasets/' + self.dataset, entity_map, relation_map, query_type, mode)

                    data[query_type][mode]["facts"] = facts
                    data[query_type][mode]["facts_filter"] = facts_filter
                    data[query_type][mode]["rel_path_ids"] = rel_path_ids

            with open(self.cachedir + '/encoded-datasets/' + self.dataset + '_regex.pkl', 'wb') as fp:
                pickle.dump(data, fp)

    def setup(self, stage, args):
        if not self.data:
            with open(self.cachedir + '/encoded-datasets/' + self.dataset + '_regex.pkl', 'rb') as fp:
                self.data = pickle.load(fp)

        train = 0
        for query_type in args.query_types:
            train += len(self.data[query_type]['train']['facts'])
        print(train)

        valid = 0
        for query_type in args.query_types:
            valid += len(self.data[query_type]['valid']['facts'])
        print(valid)

        test = 0
        for query_type in args.query_types:
            test += len(self.data[query_type]['test']['facts'])
        print(test)

        # for query_type in args.query_types:
        #     print(f"Query type: {query_type}")
        #     print(f"Train - {len(self.data[query_type]['train']['facts'])}")
        #     print(f"Valid - {len(self.data[query_type]['valid']['facts'])}")
        #     print(f"Test - {len(self.data[query_type]['test']['facts'])}")
        #     print()

        self.num_entity = len(self.data['entity_map'])
        self.num_relation = len(self.data['relation_map'])

        train_datasets = []
        valid_datasets = []
        test_datasets = []

        if stage == "fit":
            for query_type in args.query_types:
                train_datasets.append(KBDataset(self.data[query_type]['train']['facts'],
                                                None,
                                                self.data[query_type]['train']['rel_path_ids'],
                                                query_type, 'train', self.num_entity,
                                                args.negative_sample_count))

                valid_datasets.append(KBDataset(self.data[query_type]['valid']['facts'],
                                                self.data[query_type]['valid']['facts_filter'],
                                                self.data[query_type]['valid']['rel_path_ids'],
                                                query_type, 'valid', self.num_entity,
                                                -1))
            self.train = ConcatDataset(train_datasets)
            self.valid = valid_datasets

        elif stage == "test":
            for query_type in args.query_types:
                test_datasets.append(KBDataset(self.data[query_type]['test']['facts'],
                                               self.data[query_type]['test']['facts_filter'],
                                               self.data[query_type]['test']['rel_path_ids'],
                                               query_type, 'test', self.num_entity,
                                               -1))
            self.test = test_datasets

        else:
            raise ValueError(f'unexpected setup stage: {stage}')

    def train_dataloader(self):
        return DataLoader(self.train,
                          sampler=BatchSchedulerSampler(
                              self.train, self.batch_size),
                          collate_fn=custom_collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(dataset,
                           batch_size=self.eval_batch_size,
                           num_workers=self.num_workers,
                           collate_fn=custom_collate_fn,
                           pin_memory=True) for dataset in self.valid]

    def test_dataloader(self):
        return [DataLoader(dataset,
                           batch_size=self.eval_batch_size,
                           num_workers=self.num_workers,
                           collate_fn=custom_collate_fn,
                           pin_memory=True) for dataset in self.test]
