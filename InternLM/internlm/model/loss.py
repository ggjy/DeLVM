#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn.functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


class FlashGPTLMLoss(nn.Module):
    """
    Loss function for flash GPT Language Model.
    """

    def __init__(self, parallel_output=True, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None:
            if label_smoothing != 0:
                if gpc.is_rank_for_log():
                    print(f"use label_smoothing: {label_smoothing}")
        else:
            label_smoothing = 0
        self.label_smoothing = label_smoothing

        if parallel_output:
            self.loss_fn = FlashCrossEntropyLoss(
                reduction="mean",
                inplace_backward=True,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                label_smoothing=label_smoothing,
            )  # The loss in this place is bound to the gather_output initialized by VocabParallelClassifier1D
        else:
            # Here, the output will gather output is set in the model, so use ordinary loss
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(self, *args):
        if len(args) == 3:
            # residual is to match prenorm
            logits, _, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(
            shift_logits, shift_labels
        )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
        # calculated through the calculation range, and -100 must be outside this range, so there is no problem

        return loss


class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = gpc.config.kd_config.get('temperature', 1)
        self.inverse = gpc.config.kd_config.get('inverse', False)

    def forward(self, *args):
        if len(args) == 3:
            if self.inverse:
                logits_teacher, logits_student, _ = args
            else:
                logits_student, logits_teacher, _ = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")

        logits_teacher = logits_teacher.contiguous().view(-1, logits_teacher.size(-1))
        logits_student = logits_student.contiguous().view(-1, logits_student.size(-1))

        log_pred_student = F.log_softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
        loss_kd *= self.temperature ** 2

        return loss_kd
