import time

from pytorch_lightning.callbacks import Callback
from termcolor import cprint
import numpy as np


class TimerCallback(Callback):

    time_init_start = None
    time_fit_start = None

    time_train_epoch_start = None
    interval_train_epochs = []
    num_batches_train_epochs = []

    time_valid_epoch_start = None
    interval_valid_epochs = []
    num_batches_valid_epochs = []

    total_epoch_count = 0
    train_batch_count = 0
    valid_batch_count = 0

    def __init__(self, stop_epochs=5):
        super(TimerCallback, self).__init__()
        self.stop_epochs = stop_epochs

    def on_init_start(self, trainer):
        self.time_init_start = time.time()
        cprint("\nInit start", "green")

    def on_fit_start(self, trainer, pl_module):
        self.time_fit_start = time.time()
        cprint("\nFit start", "green")

    def on_train_epoch_start(self, trainer, pl_module):
        self.total_epoch_count += 1
        self.train_batch_count = 0
        self.time_train_epoch_start = time.time()
        cprint(f"\nTraining epoch start {self.total_epoch_count}", "green")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        self.train_batch_count += 1
        cprint(f"\nTraining batch incremented: {self.train_batch_count}", "green")

    def on_train_epoch_end(self, trainer, pl_module):
        self.interval_train_epochs.append(time.time() - self.time_train_epoch_start)
        self.num_batches_train_epochs.append(self.train_batch_count)
        cprint("\nTraining epoch end", "green")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.valid_batch_count = 0
        self.time_valid_epoch_start = time.time()
        cprint("\nValidation epoch start", "green")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.valid_batch_count += 1
        cprint(f"\nValidation batch incremented: {self.valid_batch_count}", "green")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.interval_valid_epochs.append(time.time() - self.time_valid_epoch_start)
        self.num_batches_valid_epochs.append(self.valid_batch_count)

        if self.total_epoch_count == self.stop_epochs:

            total_end = time.time()
            dt_init_start = total_end - self.time_init_start
            dt_fit_start = total_end - self.time_fit_start
            cprint("\n------------------", "green")

            cprint(f"- init_start ~ : {dt_init_start}", "green")
            cprint(f"- fit_start ~ : {dt_fit_start}", "green")

            cprint("------------------", "yellow")

            cprint(f"- total_epoch: {self.total_epoch_count}", "yellow")

            cprint(f"- total_train_time: {sum(self.interval_train_epochs)}", "yellow")
            m = np.mean(self.interval_train_epochs)
            cprint(f"- time / train_epoch: {m}", "yellow")
            m = sum(self.interval_train_epochs) / sum(self.num_batches_train_epochs)
            cprint(f"- time / train_batch: {m}", "yellow")

            cprint(f"- total_valid_time: {sum(self.interval_valid_epochs)}", "yellow")
            m = np.mean(self.interval_valid_epochs)
            cprint(f"- time / valid_epoch: {m}", "yellow")
            m = sum(self.interval_valid_epochs) / sum(self.num_batches_valid_epochs)
            cprint(f"- time / valid_batch: {m}", "yellow")

            cprint("END------------------", "yellow")
            exit()
