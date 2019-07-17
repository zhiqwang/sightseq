# Copyright (c) 2019-present, Zhiqiang Wang.

from fairseq.tasks import FairseqTask


class SightseqTask(FairseqTask):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def build_transforms(cls, args):
        return None

    @classmethod
    def build_transform(cls, args):
        return None

    @classmethod
    def build_target_transform(cls, args):
        return None
