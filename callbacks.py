import pytorch_lightning as pl


class ValEveryNSteps(pl.Callback):

    def __init__(self, evaluate_after_n_steps):
        self.freq = evaluate_after_n_steps

    def on_batch_start(self, trainer, pl_module):
        if (trainer.global_step % self.freq == 0 and trainer.global_step != 0):
                # (trainer.global_step == 0 and pl_module.hparams.resume_from_checkpoint):
            trainer.run_evaluation(test_mode=False)
