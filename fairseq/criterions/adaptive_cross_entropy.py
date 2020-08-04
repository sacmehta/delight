import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F
import torch


@register_criterion('adaptive_cross_entropy')
class AdaptiveCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = 1e-7
        self.smooth = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0.2, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def get_class_weight(self, target, n_classes):
        with torch.no_grad():
            dev = target.device

            class_wise_dist = torch.histc(target.float(), bins=n_classes, min=0, max=n_classes-1)
            class_wise_dist = class_wise_dist.float().to(device=dev)

            class_wise_dist = (class_wise_dist / (torch.sum(class_wise_dist) + self.eps))

            ids_to_discard = (class_wise_dist == 0)
            class_wise_dist = 1.0 + self.smooth - class_wise_dist
            class_wise_dist[ids_to_discard] = 0.0

            return class_wise_dist.to(dev)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        n_classes = lprobs.size(-1)
        weights = self.get_class_weight(target=target, n_classes=n_classes)

        if self.training:
            # use weighted loss during training
            loss = F.nll_loss(
                lprobs,
                target,
                weight=weights,
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )
        else:
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )

        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True