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

def relation_divide(triples, valid_triples, relationdict):
    true_head, true_tail = {}, {}
    n_relations = len(relationdict)
    head_counts = [[] for _ in range(n_relations)]
    tail_counts = [[] for _ in range(n_relations)]
    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    n2one_relations, one2n_realtions, n2n_relations, one2one_relations = [], [], [], []
    for trip in triples:
        h, r, t = trip
        head_counts[r].append(len((true_head[(r,t)])))
        tail_counts[r].append(len((true_tail[(h,r)])))

    for i in range(n_relations):
        avg_head = sum(head_counts[i])/len(head_counts[i])
        avg_tail = sum(tail_counts[i])/len(tail_counts[i])
        if avg_head < 1.5 and avg_tail < 1.5:
            one2one_relations.append(i)
        if avg_head < 1.5 and avg_tail >= 1.5:
            one2n_realtions.append(i)
        if avg_head >= 1.5 and avg_tail < 1.5:
            n2one_relations.append(i)
        if avg_head >= 1.5 and avg_tail >= 1.5:
            n2n_relations.append(i)

    print('1 to 1: {}'.format(len(one2one_relations)))
    print('1 to N: {}'.format(len(one2n_realtions)))
    print('N to 1: {}'.format(len(n2one_relations)))
    print('N to N: {}'.format(len(n2n_relations)))

    one2one_triples, one2n_triples, n2one_triples, n2n_triples = [], [], [], []
    for trip in valid_triples:
        head, relation, tail = trip
        if relation in one2n_realtions:
            one2n_triples.append(trip)
        if relation in one2one_relations:
            one2one_triples.append(trip)
        if relation in n2n_relations:
            n2n_triples.append(trip)
        if relation in n2one_relations:
            n2one_triples.append(trip)
    print('1 to 1: {}'.format(len(one2one_triples)))
    print('1 to N: {}'.format(len(one2n_triples,)))
    print('N to 1: {}'.format(len(n2one_triples)))
    print('N to N: {}'.format(len(n2n_triples)))
    return one2one_triples, one2n_triples, n2one_triples, n2n_triples

