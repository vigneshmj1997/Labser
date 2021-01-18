#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code

import os
import sys
import faiss
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from psutil import virtual_memory

# get environment
assert os.environ.get("LASER"), "Please set the enviornment variable LASER"
LASER = os.environ["LASER"]

sys.path.append(LASER + "/source")
sys.path.append(LASER + "/source/tools")
from embed import SentenceEncoder, EncodeLoad, EncodeFile, EmbedLoad
from text_processing import Token, BPEfastApply


###############################################################################
#
# Load texts and remove duplicates
#
###############################################################################


def TextLoadUnify(fname, seek, args, start, stop):
    if args.verbose:
        print(" - loading texts {:s}: ".format(fname), end="")
    fin = open(fname, encoding=args.encoding, errors="surrogateescape")
    fin.seek(seek)
    inds = []
    sents = []
    sent2ind = {}
    n = 0
    nu = 0
    count = 0
    pos = seek
    for line in fin:
        pos = +len(line)
        new_ind = len(sent2ind)
        inds.append(sent2ind.setdefault(line, new_ind))
        if args.unify:
            if inds[-1] == new_ind:
                sents.append(line[:-1])
                nu += 1
        else:
            sents.append(line[:-1])
            nu += 1
        n += 1
        if count == stop - start - 1:
            break
        else:
            count += 1
    if args.verbose:
        print("{:d} lines, {:d} unique".format(n, nu))
    del sent2ind
    return inds, sents, pos


###############################################################################
#
# Wrapper for knn on CPU/GPU
#
###############################################################################

def make_vres_vdev(i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = 4
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(faiss.StandardGpuResources())
    return vres, vdev

def knn(x, y, k, use_gpu):
    return knnGPU(x, y, k) if use_gpu else knnCPU(x, y, k)

    ###############################################################################
    #
    # Perform knn on GPU
    #
    ###############################################################################

    # def knnGPU(x, y, k, mem=5 * 1024 * 1024 * 1024):
    dim = x.shape[1]
    batch_size = mem // (dim * 4)
    x = x.astype("float32")
    y = y.astype("float32")

    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            # print('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[yfrom:yto])
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto - yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i - xfrom, aux[i - xfrom, j]]
                ind[i, j] = binds[i - xfrom, aux[i - xfrom, j]]
    return sim, ind


def knnGPU(x, y, k, mem=5 * 1024 * 1024 * 1024):
    x = x.astype("float32")
    y = y.astype("float32")
    # print(dir(faiss))
    # exit()
    dim = x.shape[1]
    # IndexScalarQuantizer
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat32 = True
    cfg.device = 0
    vres, vdev = make_vres_vdev()
    idx = faiss.IndexFlatL2( dim)
    # index = faiss.IndexIVFPQ (coarse_quantizer, d,
    #                      ncentroids, code_size, 8)
    # quantizer, d, nlists, M, nbits
    idx = faiss.IndexIVFPQ(idx, dim, 100, 32, 8)
    idx = faiss.index_cpu_to_all_gpus( idx)
    
    #idx = faiss.GpuIndexIVFPQ(faiss.StandardGpuResources(), idx)
    idx.train(y)
    idx.add(y)
    idx.nprobe = 10
    sim, ind = idx.search(x, k)
    return sim, ind


###############################################################################
#
# Perform knn on CPU
#
###############################################################################


def knnCPU(x, y, k):
    dim = x.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(y)
    sim, ind = idx.search(x, k)
    return sim, ind


###############################################################################
#
# Scoring
#
###############################################################################


def score(x, y, fwd_mean, bwd_mean, margin):
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False):
    if verbose:
        print(" - scoring {:d} candidates".format(x.shape[0]))
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


###############################################################################
#
# Main
#
###############################################################################
def mine_text(
    src_start, src_stop, trg_start, trg_stop, args, src_file_pos, trg_file_pos, count
):

    # Load text

    src_inds, src_sents, src_file_pos = TextLoadUnify(
        args.src, src_file_pos, args, src_start, src_stop
    )
    trg_inds, trg_sents, trg_file_pos = TextLoadUnify(
        args.trg, trg_file_pos, args, trg_start, trg_stop
    )
    print(len(src_inds), "len of src sentence")
    print(len(trg_inds), "len of src sentence")

    def unique_embeddings(emb, ind, verbose=False):
        aux = {j: i for i, j in enumerate(ind)}
        if verbose:
            print(" - unify embeddings: {:d} -> {:d}".format(len(emb), len(aux)))
        return emb[[aux[i] for i in range(len(aux))]]

    # load Embeddings
    x = EmbedLoad(
        args.src_embeddings, src_start, src_stop, args.dim, verbose=args.verbose
    )
    if args.unify:
        x = unique_embeddings(x, src_inds, args.verbose)
    if args.encoder == "laser":
        faiss.normalize_L2(x)
    y = EmbedLoad(
        args.trg_embeddings, trg_start, trg_stop, args.dim, verbose=args.verbose
    )
    if args.unify:
        y = unique_embeddings(y, trg_inds, args.verbose)
    if args.encoder == "laser":
        faiss.normalize_L2(y)

    print(x.shape, "x_shape")
    print(y.shape, "y_shape")
    if args.retrieval is not "bwd":
        if args.verbose:
            print(" - perform {:d}-nn source against target".format(args.neighborhood))
        x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], args.neighborhood), args.gpu)
        x2y_mean = x2y_sim.mean(axis=1)

    if args.retrieval is not "fwd":
        if args.verbose:
            print(" - perform {:d}-nn target against source".format(args.neighborhood))
        y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], args.neighborhood), args.gpu)
        y2x_mean = y2x_sim.mean(axis=1)

    if args.margin == "absolute":
        margin = lambda a, b: a
    elif args.margin == "distance":
        margin = lambda a, b: a - b
    else:  # args.margin == 'ratio':
        margin = lambda a, b: a / b
    name = (
        str(args.output)
        + str(src_start)
        + "_"
        + str(src_stop)
        + "_"
        + str(trg_start)
        + "_"
        + str(trg_stop)
    )
    fout = open(name, mode="w", encoding=args.encoding, errors="surrogateescape")
    if args.mode == "mine":
        print("came here")
        if args.verbose:
            print(" - mining for parallel data")
        fwd_scores = None
        fwd_scores = None
        if args.encoder == "laser":
            fwd_scores = score_candidates(
                x, y, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose
            )
            bwd_scores = score_candidates(
                y, x, y2x_ind, y2x_mean, x2y_mean, margin, args.verbose
            )
        elif args.encoder == "labse":
            fwd_scores = x2y_sim
            bwd_scores = y2x_sim
        fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmin(axis=1)]
        bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmin(axis=1)]
        if args.verbose:
            print(" - writing alignments to {:s}".format(args.output))
            if args.threshold > 0:
                print(" - with threshold of {:f}".format(args.threshold))
        if args.retrieval == "fwd":
            for i, j in enumerate(fwd_best):
                print(
                    fwd_scores[i].max(), src_sents[i], trg_sents[j], sep="\t", file=fout
                )
        if args.retrieval == "bwd":
            for j, i in enumerate(bwd_best):
                print(
                    bwd_scores[j].max(), src_sents[i], trg_sents[j], sep="\t", file=fout
                )
        if args.retrieval == "intersect":
            for i, j in enumerate(fwd_best):
                if bwd_best[j] == i:
                    print(
                        fwd_scores[i].max(),
                        src_sents[i],
                        trg_sents[j],
                        sep="\t",
                        file=fout,
                    )
        if args.retrieval == "max":
            indices = np.stack(
                (
                    np.concatenate((np.arange(x.shape[0]), bwd_best)),
                    np.concatenate((fwd_best, np.arange(y.shape[0]))),
                ),
                axis=1,
            )
            scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
            seen_src, seen_trg = set(), set()

            for i in np.argsort(-scores):
                src_ind, trg_ind = indices[i]
                if not src_ind in seen_src and not trg_ind in seen_trg:
                    seen_src.add(src_ind)
                    seen_trg.add(trg_ind)
                    if scores[i] > args.threshold:
                        print(
                            scores[i],
                            src_sents[src_ind],
                            src_ind,
                            trg_sents[trg_ind],
                            trg_ind,
                            sep="\t",
                            file=fout,
                        )
                    if src_ind == trg_ind:
                        count += 1

    fout.close()
    print(count)
    return src_file_pos, trg_file_pos, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Mine bitext")
    parser.add_argument("src", help="Source language corpus")
    parser.add_argument("trg", help="Target language corpus")
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        choices=["laser", "labse"],
        help="encoder to be used",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="Character encoding for input/output"
    )
    parser.add_argument("--src-lang", required=True, help="Source language id")
    parser.add_argument("--trg-lang", required=True, help="Target language id")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument(
        "--threshold", type=float, default=0, help="Threshold on extracted bitexts"
    )

    # mining params
    parser.add_argument(
        "--mode",
        choices=["search", "score", "mine"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument(
        "-k", "--neighborhood", type=int, default=4, help="Neighborhood size"
    )
    parser.add_argument(
        "--margin",
        choices=["absolute", "distance", "ratio"],
        default="ratio",
        help="Margin function",
    )
    parser.add_argument(
        "--retrieval",
        choices=["fwd", "bwd", "max", "intersect"],
        default="max",
        help="Retrieval strategy",
    )
    parser.add_argument("--unify", action="store_true", help="Unify texts")
    parser.add_argument(
        "--gpu", action="store_true", help="Run knn on all available GPUs"
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    # embeddings
    parser.add_argument(
        "--src-embeddings", required=True, help="Precomputed source sentence embeddings"
    )
    parser.add_argument(
        "--trg-embeddings", required=True, help="Precomputed target sentence embeddings"
    )
    parser.add_argument(
        "--dim", type=int, default=1024, help="Embedding dimensionality"
    )
    parser.add_argument(
        "--num-lines", type=int, required=True, help="Number of lines in file"
    )
    args = parser.parse_args()

    print("LASER: tool to search, score or mine bitexts")
    if args.gpu:
        print(" - knn will run on all available GPUs (recommended)")
    else:
        print(" - knn will run on CPU (slow)")
    mem = virtual_memory()
    mem_total = mem.total

    batch_size_file = 3000
    if args.encoder == "labse":
        batch_size_file = int(mem_total * 6000000 // 64)
    if args.encoder == "laser":
        batch_size_file = int(mem_total * 3000000 // 64)
    src_pos = 0
    trg_pos = 0
    tot = 0
    for i in tqdm(range(0, args.num_lines, batch_size_file)):
        for j in range(0, args.num_lines, batch_size_file):
            src_stop = min(i + batch_size_file, args.num_lines)
            trg_stop = min(j + batch_size_file, args.num_lines)
            src_pos, trg_pos, count = mine_text(
                i, src_stop, j, trg_stop, args, src_pos, trg_pos, tot
            )
            tot += count
    print("result",args.num_lines / count)
# def mine_text(src_start,src_stop,trt_start,trg_stop,args,src_pos,trg_pos):
