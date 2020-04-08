#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import argparse
import json
import logging
import numpy as np
import torch
import random
from kgembedUtils.kgutils import build_graph_from_triples

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def graph_construction(args, triples, num_entities, num_relations):
    with_cuda = args.cuda
    graph, _ = build_graph_from_triples(num_nodes=num_entities, num_relations=num_relations,
                                                 triples=np.array(triples, dtype=np.int64).transpose())
    if with_cuda:
        for key, value in graph.ndata.items():
            graph.ndata[key] = value.cuda()
        for key, value in graph.edata.items():
            graph.edata[key] = value.cuda()
    return graph


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    log_file = os.path.join(args.save_path, 'test_results.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def relation_divide(triples, relation_dict):
    nrelation = len(relation_dict)
    relation_trips_array = [[] for _ in range(nrelation)]
    for trip in triples:
        relation_trips_array[trip[1]].append(trip)
    divide_size = [len(x) for x in relation_trips_array]

    id2relation = {value: key for key, value in relation_dict.items()}
    for idx in range(nrelation):
        print(id2relation[idx], divide_size[idx])
    return relation_trips_array

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Testing Knowledge Graph Embedding Models')
    parser.add_argument('--cuda', default=False, action='store_true', help='use GPU')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--data_path', type=str, default='../data/wn18rr')
    parser.add_argument('-save', '--save_path', default='../models/wn18rr', type=str)
    parser.add_argument('--model', default='DistMult', type=str)
    parser.add_argument('-cpu', '--cpu_num', default=16, type=int)
    parser.add_argument('-g', '--gamma', default=0.15, help='label smoothing factor', type=float)
    parser.add_argument('--seed', default=2019, type=int, help='RANDOM SEED')
    return parser.parse_args(args)

def main(args):
    print('here')
    set_logger(args)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    relation_divide(valid_triples, relation2id)


if __name__ == '__main__':
    main(parse_args())