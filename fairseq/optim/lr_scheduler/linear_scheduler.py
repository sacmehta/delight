from . import FairseqLRScheduler, register_lr_scheduler
import numpy as np


@register_lr_scheduler('linear')
class LinearSchedule(FairseqLRScheduler):
    """Decay the LR linearly based on the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with linear.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        total_updates = args.max_update
        if total_updates <= 0:
            raise ValueError(
                'Cannot use linear scheduler with --max-updates <=0'
                'Consider passing a value for --max-updates argument.'
                'e.g., --max-updates=300000'
            )

        min_lr = args.warmup_init_lr
        if min_lr < 0:
            raise ValueError(
                'Cannot use linear scheduler with --warmup-init-lr <0'
                'Consider using a value >=0.'
                'A good value is 1e-7'
            )

        max_lr = args.lr[0]
        if max_lr <= 0:
            raise ValueError(
                'Cannot use linear scheduler with --lr <= 0'
                'Consider using a positive value.'
                'A good value is 0.0014'
            )

        lr_steps = []
        warm_up_steps = args.warmup_updates
        if warm_up_steps > 0:
            lr_steps = lr_steps + np.linspace(min_lr, max_lr, warm_up_steps).tolist()
        lr_steps = lr_steps + np.linspace(max_lr, min_lr, total_updates - warm_up_steps + 1).tolist()

        self.lr_steps = lr_steps

        self.max_lr_updates = len(self.lr_steps)
        self.warmup_steps = warm_up_steps
        self.reamining_steps = total_updates - args.warmup_updates + 1

        self.max_lr = max_lr
        self.min_lr = min_lr

        # initial learning rate
        self.lr = min_lr
        self.optimizer.set_lr(min_lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=8000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=1e-7, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.max_lr_updates:
            self.lr = self.lr_steps[num_updates]
        else:
            # go to default value
            self.lr = self.min_lr
        self.optimizer.set_lr(self.lr)
        return self.lr

    def __repr__(self):
        class_name = self.__class__.__name__
        s = '{}'.format(class_name)
        if self.warmup_steps > 0:
            s += '\n \t LR changes from {} to {} in {} steps'.format(self.min_lr, self.max_lr, self.warmup_steps)
        s += '\n \t LR changes from {} to {} in {} steps'.format(self.max_lr, self.min_lr, self.reamining_steps)
        return s
