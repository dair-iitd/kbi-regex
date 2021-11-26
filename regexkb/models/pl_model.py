from abc import ABC

import torch

import pytorch_lightning as pl
from metrics import hits_at_x, mean_rank, mean_rank_reciprocal

import wandb

METRIC_PROG_BAR = {
    'mr': False,
    'mrr': True,
    'hits_1': False,
    'hits_3': False,
    'hits_5': False,
    'hits_10': True
}


class Model(pl.LightningModule, ABC):

    def __init__(self):
        super().__init__()
        self.warmup_epochs = 0

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(),
            lr=self.hparams.learning_rate)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        if self.hparams.lr_schedule and self.warmup_epochs == 0:
            if self.hparams.lr_schedule == 'half':
                self.warmup_epochs = self.hparams.max_epochs // 2

        if self.hparams.lr_schedule:
            if self.hparams.lr_schedule == 'half':
                if epoch == self.warmup_epochs:
                    print()
                    print("Reducing LR by a factor of 5...")
                    print()
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['lr'] / 5
                    # self.hparams.lr_schedule = None
                    self.warmup_epochs = round(self.warmup_epochs * 1.5)
            # else:
            #     assert False, "LR Scheduler not properly defined. \
            #         Choose between - half, ..."

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):

        facts, negative_sample, rel_path_ids, query_type = batch
        positive_score = self(
            facts[:, 0],
            facts[:, 1:-1],
            facts[:, -1],
            rel_path_ids,
            query_type, 'train')

        negative_score = self(
            facts[:, 0],
            facts[:, 1:-1],
            negative_sample,
            rel_path_ids,
            query_type, 'train')

        return {'pos': positive_score, 'neg': negative_score}

    def training_step_end(self, outputs):

        loss = self.loss(outputs['pos'], outputs['neg'])
        self.log('train_loss', loss)
        wandb.log({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        facts, facts_filter, rel_path_ids, query_type = batch
        scores = self(
            facts[:, 0],
            facts[:, 1:-1],
            None,
            rel_path_ids,
            query_type, 'test')

        score_of_expected = scores.gather(1, facts[:, -1].unsqueeze(1).data)
        scores.scatter_(1, facts_filter, self.minimum_value)
        greater = scores.ge(score_of_expected).float()
        equal = scores.eq(score_of_expected).float()
        rank = greater.sum(dim=1) + 1 + equal.sum(dim=1) / 2.0
        # should store score too for future analysis
        return {'rank': rank}

    def validation_epoch_end(self, outputs):

        if len(self.hparams.query_types) == 1:
            outputs = [outputs]

        evaluate_kbc = self.hparams.query_types == [0]

        results = {}
        total_length = 0

        for idx, query_type in enumerate(self.hparams.query_types):
            ranks = torch.cat([_['rank'] for _ in outputs[idx]])

            metrics = {'mr': mean_rank(ranks),
                       'mrr': mean_rank_reciprocal(ranks),
                       'hits_1': hits_at_x(ranks, 1),
                       'hits_3': hits_at_x(ranks, 3),
                       'hits_5': hits_at_x(ranks, 5),
                       'hits_10': hits_at_x(ranks, 10)}
            self.logger.experiment.add_scalars(
                'metrics_query_' + str(query_type), metrics)
            if self.hparams.do_test:
                print("Query type is - " + str(query_type))
                print("number of queries is - " + str(ranks.shape[0]))
                print("MRR is " + str(metrics["mrr"].item()))
                print("HITS@10 is " + str(metrics["hits_10"].item()))
                print()
            if not evaluate_kbc and query_type == 0:
                pass
            else:
                total_length += ranks.shape[0]
                for k, v in metrics.items():
                    if k not in results:
                        results[k] = metrics[k] \
                            * ranks.shape[0]
                    else:
                        results[k] += metrics[k] \
                            * ranks.shape[0]

        # if len(self.hparams.query_types) > 1 and 0 in self.hparams.query_types:
        #     eval_queries = len(self.hparams.query_types) - 1
        # else:
        #     eval_queries = len(self.hparams.query_types)

        for k, v in results.items():
            # self.log(k, v/eval_queries,
            #          prog_bar=METRIC_PROG_BAR[k])
            self.log(k, v/total_length, prog_bar=METRIC_PROG_BAR[k])
            wandb.log({"metric_" + k: v/total_length})

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
