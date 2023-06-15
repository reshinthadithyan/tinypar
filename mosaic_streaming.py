from torch.utils.data import DataLoader
from streaming import StreamingDataset
from tqdm import tqdm
import os
import streaming 
from torch.utils.data import DataLoader
from transformers import GPTNeoXTokenizerFast
import torch
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_dloader_from_path_config(data_path_config:dict):
    """
    data_path_config : 
    {
        "dataset_1_path" : num_epochs ,
        "dataset_2_path" : num_epochs ,
        "dataset_3_path" : num_epochs ,
    }
    """

    stream_list = []
    for path, num_epochs in data_path_config.items():
        stream_list.append(
            streaming.Stream(remote=path,repeat=num_epochs,local=f"/fsx/home-reshinth/.cache/mstreaming/ds{time.time()}")
        )
    dataset = StreamingDataset(
    streams = stream_list,
    shuffle=True,
    )
    return dataset


def chunk_list_np(lst, chunk_size):
    arr = np.array(lst)
    reshaped = arr.reshape(-1, chunk_size)
    return reshaped


def chunk_generator(dataset, batch_size, seqlen,tokenizer):
    buffer = []
    for sample in dataset:
        sample = sample["text"]
        encoded = tokenizer(sample,
                            truncation=False,
                            padding=False)
        ids = encoded['input_ids']  + [tokenizer.eos_token_id]
        buffer.extend(ids)
        #Ch
        while len(buffer) >= seqlen*batch_size:
            chunk, buffer = buffer[:seqlen*batch_size], buffer[seqlen*batch_size:]
            chunk = chunk_list_np(chunk, seqlen)
            yield chunk
    if len(buffer) > 0:
        yield buffer


class CustomStreamDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, seqlen, enc, *args, **kwargs):
        sampler = None
        super().__init__(dataset, batch_size=batch_size, *args, **kwargs)
        self.generator = chunk_generator(dataset, batch_size, seqlen, enc)

    def __iter__(self):
        return self.generator

def create_streaming_dataloader(data_path_config,batch_size:int,seq_len:int,tokenizer,num_workers,prefetch_factor:int):

    dataset = create_dloader_from_path_config(data_path_config)
    dataloader = CustomStreamDataLoader(dataset,batch_size=32,seqlen=seq_len,enc=tokenizer,num_workers=os.cpu_count(),prefetch_factor=prefetch_factor)
    return dataloader


if __name__ == "__main__":
    data_path_config = {
        "s3://pile-everything/redpajama_mds/arxiv" : 1,
        "s3://pile-everything/redpajama_mds/stackexchange" : 1
    }
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    logger.info(f"Loaded Tokenizer")
    dataloader = create_streaming_dataloader(data_path_config,batch_size=32,seq_len=2048,tokenizer=tokenizer,num_workers=os.cpu_count(),prefetch_factor=2)
    for batch in tqdm(dataloader):
        print(batch)
        print(batch.shape)



