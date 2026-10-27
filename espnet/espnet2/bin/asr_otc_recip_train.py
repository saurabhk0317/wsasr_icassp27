#!/usr/bin/env python3
#
# Part of ESPnet. Licensed under the Apache License, Version 2.0
# (see the LICENSE file in the ESPnet source root).
from espnet2.tasks.asr_otc_recip import ASRTask


def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training with OTC (asr_otc_recip task).

    Example:

        % python asr_otc_recip_train.py --print_config --optim adam \
                > conf/train_asr_otc_recip.yaml
        % python asr_otc_recip_train.py --config conf/train_asr_otc_recip.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
