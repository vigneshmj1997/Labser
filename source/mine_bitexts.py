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
import time


from sklearn.decomposition import PCA

# get environment
assert os.environ.get("LASER"), "Please set the enviornment variable LASER"
LASER = os.environ["LASER"]

sys.path.append(LASER + "/source")
sys.path.append(LASER + "/source/tools")
from embed import SentenceEncoder, EncodeLoad, EncodeFile, EmbedLoad
from text_processing import Token, BPEfastApply
import torch

###############################################################################
#
# Load texts and remove duplicates
#
###############################################################################


def TextLoadUnify(fname, seek, args, start, stop):
    print(start, ",", stop)
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

gpu_resources = []
ngpu = torch.cuda.device_count()
for i in range(torch.cuda.device_count()):
    res = faiss.StandardGpuResources()
    res.setTempMemory(8 * 1024 * 1024 * 1024)
    gpu_resources.append(res)


def make_vres_vdev(i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


def create_index(
    numpy_filename,
    encoder_name,
    dim=768,
    dtype="fp16",
    batch_size=5000000,
    out_file="indexfile",
):
    if dtype == "fp16":
        denomenator = 2
    elif dtype == "fp32":
        denomenator = 4

    file_size = int(os.stat(numpy_filename).st_size / (args.dim * denomenator))
    coarse_quantizer = faiss.IndexFlatIP(dim)
    indexall = faiss.IndexIVFFlat(coarse_quantizer, dim, 32768, faiss.METRIC_L2)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = True
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.reserveVecs = file_size
    co.shard = True
    assert co.shard_type in (0, 1, 2)
    vres, vdev = make_vres_vdev()
    gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, indexall, co)
    max_add = file_size
    print("add...")
    print(gpu_index)

    t0 = time.time()
    batch_size = min(batch_size, file_size)
    print(batch_size, "batch_size")
    for i0 in tqdm(range(0, file_size, batch_size)):
        i1 = min(i0 + batch_size, file_size)
        print(i1, "i1")

        xs = EmbedLoad(
            numpy_filename,
            i0,
            i1,
            dim=1024 if encoder_name == "laser" else 768,
            dtype=dtype,
        )
        print(gpu_index)
        xs = xs.astype(np.float32)
        # gpu_index.train(xs)
        # gpu_index.add(xs)
        print(xs.shape)
        print(np.arange(i0, i1).shape)
        print(gpu_index.ntotal, "gpuindex")
        print(max_add, "max_add")
        indexall.train(xs)
        print(gpu_index.is_trained, "gpu_index is trained")
        gpu_index.add_with_ids(xs, np.arange(i0, i1))
        if max_add > 0 and gpu_index.ntotal > max_add:
            print("Flush indexes to CPU")
            for i in range(ngpu):
                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                print("  index %d size %d" % (i, index_src.ntotal))
                index_src.copy_subset_to(indexall, 0, 0, dim)
                index_src_gpu.reset()
                index_src_gpu.reserveMemory(max_add)
            gpu_index.sync_with_shard_indexes()

        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    print("Aggregate indexes to CPU")
    t0 = time.time()

    if hasattr(gpu_index, "at"):
        # it is a sharded index
        for i in range(ngpu):
            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
            print("  index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(indexall, 0, 0, dim)
    else:
        # simple index
        index_src = faiss.index_gpu_to_cpu(gpu_index)
        index_src.copy_subset_to(indexall, 0, 0, dim)

    print("  done in %.3f s" % (time.time() - t0))

    if max_add > 0:
        # it does not contain all the vectors
        gpu_index = None
    file = open(out_file, "wb")
    print(gpu_index, "gpu__index")
    print(indexall, "indexall")
    print(indexall.is_trained, "is trained")
    print(indexall.ntotal, "ntotal indexall")

    faiss.write_index(indexall, out_file)
    return gpu_index, indexall


def knn(x, y, k, use_gpu, dim, index):
    return knnGPU(x, y, k, dim, index) if use_gpu else knnCPU(x, y, k)

    ###############################################################################
    #
    # Perform knn on GPU
    #
    ###############################################################################


def knnGPU(x, y, k, dim, index):
    x = x.astype("float32")
    d = x.shape[1]
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = False
    co.indicesOptions = 0
    co.verbose = True
    co.shard = True  # the replicas will be made "manually"
    t0 = time.time()
    vres, vdev = make_vres_vdev()
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
    ps = faiss.GpuParameterSpace()
    ps.initialize(index)
    ps.set_index_parameter(index, "nprobe", k)
    index.add(y)
    sim, ind = index.search(x, k)
    del index
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
    src_start,
    src_stop,
    trg_start,
    trg_stop,
    args,
    src_file_pos,
    trg_file_pos,
    count,
    src_index,
    trg_index,
):

    # Load text

    src_inds, src_sents, src_file_pos = TextLoadUnify(
        args.src, src_file_pos, args, src_start, src_stop
    )
    trg_inds, trg_sents, trg_file_pos = TextLoadUnify(
        args.trg, trg_file_pos, args, trg_start, trg_stop
    )
    # print(len(src_inds), "len of src sentence")
    # print(len(trg_inds), "len of trg sentence")

    def unique_embeddings(emb, ind, verbose=False):
        aux = {j: i for i, j in enumerate(ind)}
        if verbose:
            print(" - unify embeddings: {:d} -> {:d}".format(len(emb), len(aux)))
        return emb[[aux[i] for i in range(len(aux))]]

    # load Embeddings
    # load x
    pca_value = PCA(n_components=args.pca_dim)
    x = EmbedLoad(
        args.src_embeddings,
        src_start,
        src_stop,
        args.dim,
        verbose=args.verbose,
        dtype=args.dtype,
    )
    if args.unify:
        x = unique_embeddings(x, src_inds, args.verbose)

    # faiss.normalize_L2(x)

    # load y
    print(x.shape)
    y = EmbedLoad(
        args.trg_embeddings,
        trg_start,
        trg_stop,
        args.dim,
        verbose=args.verbose,
        dtype=args.dtype,
    )
    print(y.shape)
    # pca_value.fit(np.concatenate((x, y), axis=0))
    # x = pca_value.transform(x)
    # y = pca_value.transform(y)
    if args.unify:
        y = unique_embeddings(y, trg_inds, args.verbose)

    # faiss.normalize_L2(y)

    x2y_sim, x2y_ind = knn(
        x, y, min(y.shape[0], args.neighborhood), args.gpu, args.dim, trg_index
    )
    x2y_mean = x2y_sim.mean(axis=1)

    y2x_sim, y2x_ind = knn(
        y, x, min(x.shape[0], args.neighborhood), args.gpu, args.dim, src_index
    )
    y2x_mean = y2x_sim.mean(axis=1)

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
    if args.verbose:
        print(" - mining for parallel data")
    fwd_scores = score_candidates(
        x, y, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose
    )
    bwd_scores = score_candidates(
        y, x, y2x_ind, y2x_mean, x2y_mean, margin, args.verbose
    )
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

    if args.verbose:
        print(" - writing alignments to {:s}".format(args.output))
        if args.threshold > 0:
            print(" - with threshold of {:f}".format(args.threshold))
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
            if scores[i] >= args.threshold:
                print(
                    scores[i],
                    src_sents[src_ind],
                    trg_sents[trg_ind],
                    sep="\t",
                    file=fout,
                )
            if src_ind == trg_ind:
                count += 1

    fout.close()
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
        "--dtype",
        type=str,
        required=True,
        choices=["fp16", "fp32"],
        help="datatype of numpy array",
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
        "-k", "--neighborhood", type=int, default=4, help="Neighborhood size"
    )
    parser.add_argument(
        "--index-factory",
        type=str,
        default="IVF6384_HNSW32",
        help="Faiss Index factory string",
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
    parser.add_argument("--pca-dim", type=int, help="PCA value")
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
        "--chunk-size",
        type=float,
        required=True,
        help="size of data to be loaded at a time",
    )
    parser.add_argument(
        "--src_lines", nargs="+", type=int, help="start,stop lines for source file"
    )
    parser.add_argument(
        "--trg_lines", nargs="+", type=int, help="start,stop lines for target file"
    )

    args = parser.parse_args()

    print("LASER: tool to search, score or mine bitexts")
    if args.gpu:
        print(" - knn will run on all available GPUs (recommended)")
    else:
        print(" - knn will run on CPU (slow)")

    if args.dtype == "fp16":
        denomenator = 2
    elif args.dtype == "fp32":
        denomenator = 4
    src_start = 0
    trg_start = 0
    src_stop = int(os.stat(args.src_embeddings).st_size / (args.dim * denomenator))
    trg_stop = int(os.stat(args.trg_embeddings).st_size / (args.dim * denomenator))

    size_src = int(os.stat(args.src_embeddings).st_size / (args.dim * denomenator))
    size_trg = int(os.stat(args.trg_embeddings).st_size / (args.dim * denomenator))
    print(size_src)
    print(size_trg)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    if args.src_lines != None:
        src_start = args.src_lines[0]
        trg_start = args.trg_lines[0]
        src_stop = args.src_lines[1]
        trg_stop = args.trg_lines[1]
    chunk_lines = min(int(size_src / args.dim), int(size_trg / args.dim))
    if args.encoder == "labse":
        chunk_lines = int(args.chunk_size * 6000000 // 16)
    if args.encoder == "laser":
        chunk_lines = int(args.chunk_size * 3000000 // 24)
    src_pos = 0
    trg_pos = 0
    # According to the US Census Bureau, by 2010, the District's population declined to 20,876 people.
    tot = 0
    print(chunk_lines, "chunk_lines")
    if args.verbose:
        print("Creating index for src")
    src_gpu_index, src_index = create_index(
        args.src_embeddings, "labse", out_file="index_src"
    )
    if args.verbose:
        print("Creating index for trg")
    trg_gpu_index, trg_index = create_index(
        args.trg_embeddings, "labse", out_file="index_trg"
    )
    for i in tqdm(range(src_start, src_stop, chunk_lines)):
        for j in range(trg_start, trg_stop, chunk_lines):

            src_stop = min(i + chunk_lines, size_src)
            trg_stop = min(j + chunk_lines, size_trg)
            start = time.time()
            src_pos, trg_pos, count = mine_text(
                i,
                src_stop,
                j,
                trg_stop,
                args,
                src_pos,
                trg_pos,
                tot,
                src_index,
                trg_index,
            )
            stop = time.time()
            print("Total time is per itertation is ", start - stop)
            tot += count
    print(count)
