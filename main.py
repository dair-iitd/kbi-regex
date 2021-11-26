import argparse
import datetime
import os
import shutil
import json

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader
import torch

import params
import regexkb.models as models
from callbacks import ValEveryNSteps
from regexkb.datasets import DataModule

import wandb
wandb.init(project="kbi-regex")

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def main(args):

    dataset = DataModule('cache', args)
    dataset.prepare_data()
    dataset.setup('fit', args)

    args.num_entity = dataset.num_entity
    args.num_relation = dataset.num_relation

    # If disjunction op not mentioned, it is same as Kleene plus op
    if args.disjunction_op is None:
        args.disjunction_op = args.kleene_plus_op

    args.beta_mode = eval_tuple(args.beta_mode)

    wandb.config.update(args)

    model = getattr(models, args.model)(args)
    wandb.watch(model, log_freq=100)

    # If starting regex training from pre-trained KBC weights
    if args.resume_from_checkpoint and args.do_train:
        model.load_from_checkpoint(args.resume_from_checkpoint)
        args.resume_from_checkpoint = None

    logger = TensorBoardLogger(
        f'{args.save_dir}/tensorboard')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.save_dir}/checkpoints',
        filename='checkpoint-{step}-{mrr:.2f}',
        verbose=True,
        monitor='mrr',
        mode='max',
        save_top_k=1,
        period=1)

    trainer = Trainer(
        logger=logger,
        # log_every_n_steps=200,
        # flush_logs_every_n_steps=1000,
        # num_sanity_val_steps=0,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=args.resume_from_checkpoint,
        check_val_every_n_epoch=float('inf'),
        distributed_backend='dp',
        replace_sampler_ddp=False,
        callbacks=[ValEveryNSteps(args.evaluate_after_n_steps)])

    if args.do_train:
        trainer.logger.log_hyperparams(args)
        trainer.logger.save()
        trainer.fit(model, dataset)

        shutil.copyfile(
            checkpoint_callback.best_model_path,
            f'{args.save_dir}/checkpoints/checkpoint.ckpt')
        print('best checkpoint:', checkpoint_callback.best_model_path)

        dataset.setup('test', args)
        results = trainer.test(
            datamodule=dataset,
            verbose=False,
            ckpt_path=checkpoint_callback.best_model_path,
        )[0]
        print(results)

        os.makedirs(f'{args.save_dir}/results', exist_ok=True)
        with open(f'{args.save_dir}/results/experiment.json', "w") as f:
            json.dump(
                {**vars(args), 'results': results}, f)

    if args.do_test:
        dataset.setup('test', args)
        results = trainer.test(
            model=model,
            datamodule=dataset,
            verbose=False,
            ckpt_path=args.resume_from_checkpoint,
        )[0]
        print(results)
        if args.dataset=="fb15k":
            for key in results.keys():
                results[key] = results[key] * 0.948
        else:
            for key in results.keys():
                results[key] = results[key] * 0.756
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = params.add_args(parser)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.num_workers))
    main(args)
