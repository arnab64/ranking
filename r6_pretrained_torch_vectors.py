#!/usr/bin/python
import torch
import numpy as np
import xgboost as xgb
import pandas as pd
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import collate_tensors
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.nn import LockedDropout

loaded_data = ["now this ain't funny", "so don't you dare laugh"]
encoder = WhitespaceEncoder(loaded_data)
encoded_data = [encoder.encode(example) for example in loaded_data]

print("encoded_data",encoded_data)

encoded_data = [torch.randn(2), torch.randn(3), torch.randn(4), torch.randn(5)]

train_sampler = torch.utils.data.sampler.SequentialSampler(encoded_data)
train_batch_sampler = BucketBatchSampler(
            train_sampler, batch_size=2, drop_last=False, sort_key=lambda i: encoded_data[i].shape[0])

batches = [[encoded_data[i] for i in batch] for batch in train_batch_sampler]
batches = [collate_tensors(batch, stack_tensors=stack_and_pad_tensors) for batch in batches]

print("batches=",batches)
