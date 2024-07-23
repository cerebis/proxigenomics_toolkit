import warnings
import logging
import Bio.SeqIO
import tqdm
import torch
import pickle
import transformers
import argparse
import torch.utils.data as util_data
import torch.nn as nn
import random
import numpy as np
import os

from collections import OrderedDict


def calculate_llm_embedding(dna_sequences, model, tokenizer, batch_size, n_gpu, max_length):

    train_loader = util_data.DataLoader(dna_sequences,
                                        batch_size=batch_size*n_gpu,
                                        shuffle=False,
                                        num_workers=2*n_gpu)

    for i, batch in enumerate(train_loader):

        with torch.no_grad():

            token_feat = tokenizer.batch_encode_plus(
                    batch,
                    max_length=max_length,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True
                )

            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()

            model_output = model.forward(input_ids=input_ids,
                                         attention_mask=attention_mask)[0].detach().cpu()

            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

            if i == 0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    return np.array(embeddings.detach().cpu())


if __name__ == '__main__':

    # avoid warning about tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings('ignore', message='Unable to import Triton', category=Warning, append=False)
    warnings.filterwarnings('ignore', message='Increasing alibi size', category=Warning, append=False)

    logging.basicConfig(filename='embed.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Calculate embeddings for DNA sequences')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='Random seed [default: not set]')
    parser.add_argument('-n', '--chunk-size', default=5000,
                        type=int, help='Chunk-size for larger sequences [default: 5000]')
    parser.add_argument('-d', '--device-id', default=0, type=int,
                        help='Specify GPU device ID [default: 0]')
    parser.add_argument('--n-gpu', default=1,
                        type=int, help='Number of GPUs to use [default: 1]')
    parser.add_argument('-b', '--batch-size', default=50,
                        type=int, help='Batch-size for processing sequences [default: 50]')
    parser.add_argument('fasta', help='input fasta file')
    parser.add_argument('pickle', help='output pickled embed')
    args = parser.parse_args()

    if args.seed is not None:
        # UNSOLVED ISSUE
        # despite this effort, success runs produce different embeddings
        # chunking of input data IS deterministic.
        torch.manual_seed(args.seed)
        transformers.set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not torch.cuda.is_available():
        device = "cpu"
    else:
        gpu_ok = False
        print(f'Using: cuda:{args.device_id}')
        torch.cuda.device(args.device_id)
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
        if not gpu_ok:
            warnings.warn(
                "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
                "than expected."
            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            'zhihan1996/DNABERT-S',
            cache_dir=None,
            model_max_length=args.chunk_size + 500,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            revision='main',
        )

    model = transformers.AutoModel.from_pretrained(
            'zhihan1996/DNABERT-S',
            trust_remote_code=True,
            revision='main',
        )

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1 and args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to('cuda')

    # scan the file for the total number of sequences for the progress bar (lame)
    total_sequences = sum(1 for _ in Bio.SeqIO.parse(args.fasta, 'fasta'))

    embeds = OrderedDict()
    for seq in tqdm.tqdm(Bio.SeqIO.parse(args.fasta, 'fasta'), total=total_sequences):

        seq_str = str(seq.seq)

        # For long sequences, calculate embeddings for pieces
        # This is not necessary. DNABERT-S becomes chatty when handed sequences
        # exceed about 5kbp and processing appears to slow down. Instead, we
        # will calculate an average. Perhaps we should explore other sampling
        # strategies such as overlapping chunks or cover using a random sample
        # to avoid bias.
        if len(seq_str) <= args.chunk_size:
            chunks = [seq_str]
        else:
            n = len(seq_str) // args.chunk_size
            step = int(args.chunk_size + (len(seq_str) % args.chunk_size) / n + 1)
            chunks = [seq_str[i:i+step] for i in range(0, len(seq_str), step)]

        with open('chunks.txt', 'w') as out_h:
            for i, chk in enumerate(chunks):
                out_h.write(f'{i}\t{chk}\n')

        embeddings = calculate_llm_embedding(chunks,
                                             model,
                                             tokenizer,
                                             batch_size=args.batch_size,
                                             n_gpu=args.n_gpu,
                                             max_length=args.chunk_size+500)

        embeds[seq.id] = embeddings

    with open(args.pickle, 'wb') as out_h:
        pickle.dump(embeds, out_h)
