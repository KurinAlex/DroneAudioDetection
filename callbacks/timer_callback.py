import time

from lightning import Callback


class TimerCallback(Callback):
    def __init__(self):
        super().__init__()

        self._start_time = 0.0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        duration = time.perf_counter() - self._start_time
        pl_module.log("train/time", duration, on_step=False, on_epoch=True)
