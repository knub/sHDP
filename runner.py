#!/usr/bin/env python2.7

import argparse
import csv
from datetime import datetime
import operator
import os
import pickle as pk
import sys
from argparse import RawTextHelpFormatter
from collections import Counter
from multiprocessing import Pool
from functools import partial
from time import time

import numpy as np
from sklearn.preprocessing import normalize

from HDP import models
from HDP.util.general import sgd_passes
from HDP.util.text import progprint
from core.distributions import vonMisesFisherLogNormal

import mkl

mkl.set_num_threads(4)

project_path = ''
results_path = project_path + 'results/'

def build_embedding_corpus(corpus, embeddings):
    corpus_embeddings = []
    words = []
    for doc in corpus:
        embeddings_doc = []
        words_doc = []
        for word, count in doc:
            if word in embeddings:
                pass
            elif word.capitalize() in embeddings:
                word = word.capitalize()
            elif word.upper() in embeddings:
                word = word.upper()

            try:
                embeddings_doc.append((np.array(embeddings[word]).astype(float), count))
            except:
                pass
            words_doc.append(word)
        corpus_embeddings.append(np.array(embeddings_doc))
        words.append(np.array(words_doc))
    return corpus_embeddings, words


def read_corpus(args):
    with open(args.vocabulary, "r") as f:
        vocab = [l.rstrip() for l in f.readlines()]

    def normalize_to_unit_length(a):
        a = np.array(a)
        a = a.reshape(1, -1)
        a = normalize(a)
        a = a.reshape(-1)
        return a

    with open(args.embedding_model, "r") as f:
        embeddings = [l.rstrip() for l in f.readlines()]
        embeddings = [l.split(" ") for l in embeddings]
        embeddings = [[float(s) for s in split] for split in embeddings]
        embeddings = [normalize_to_unit_length(e) for e in embeddings]

    assert len(embeddings) == len(vocab), "sizes should match"
    embeddings = dict(zip(vocab, embeddings))

    corpus = []
    current_doc = []
    line_nr = 0
    with open(args.corpus, "r") as f:
        for line in f:
            line = line.rstrip()
            if line == "##":
                if len(current_doc) > 0:
                    corpus.append(current_doc)
                    current_doc = []
                else:
                    print "empty document at line " + line_nr
            else:
                word_id = int(line[:6])
                # topic_id = int(line[7:])
                current_doc.append(word_id)
                # topics.add(topicId)
            line_nr += 1

        corpus = [[vocab[i] for i in doc] for doc in corpus]
        corpus = [Counter(doc) for doc in corpus]
        corpus = [list(doc.iteritems()) for doc in corpus]

    return corpus, embeddings


def read_other_corpus(dataset="nips"):
    temp_file = open('data/' + dataset + '/texts.pk', 'rb')
    corpus = pk.load(temp_file)
    temp_file.close()

    csv.field_size_limit(sys.maxsize)
    vectors_file = open('data/' + dataset + '/wordvec.pk', 'rb')
    embeddings = pk.load(vectors_file)

    return corpus, embeddings


def HDPRunner(args):
    start_time = time()
    np.random.seed(args.seed)
    K = args.K
    alpha = args.alpha
    gamma = args.gamma
    tau = args.tau
    kappa_sgd = args.kappa_sgd
    multibatch_size = args.multibatch_size

    corpus_x, embeddings = read_corpus(args)
    corpus, embeddings_x = read_other_corpus()
    num_dim = len(embeddings["word"])

    if "nips" in args.corpus:
        corpus_name = "nips"
    elif "20news" in args.corpus:
        corpus_name = "20news"
    else:
        raise Exception("Corpus not known: " + args.corpus)

    results_folder = "results/%s/embeddings-ours.dim-%d.seed-%d.topics-%d.alpha-%s.gamma-%s.kappa-%s.tau-%s.batch-%d" % (
        corpus_name,
        num_dim,
        args.seed,
        K,
        str(float(alpha)).replace(".", "-"),
        str(float(gamma)).replace(".", "-"),
        str(float(kappa_sgd)).replace(".", "-"),
        str(float(tau)).replace(".", "-"),
        multibatch_size
    )
    try:
        os.mkdir(results_folder)
    except OSError:
        if os.path.exists(results_folder + "/count-based.topics"):
            raise Exception("folder " + results_folder + " already exists")

    with open(results_folder + "/args.txt", "w") as args_file:
        args_file.write(str(vars(args)) + "\n")
    log_file = open(results_folder + "/log", "w")

    ########### Runner
    embedding_corpus, words_corpus = build_embedding_corpus(corpus, embeddings)
    embedding_corpus = zip(embedding_corpus, range(len(embedding_corpus)))
    num_docs = len(embedding_corpus)
    vocabulary = np.unique([word for doc in words_corpus for word in doc])

    log_file.write("{'num_docs': %d, 'num_dim': %d, 'num_embeddings': %d}\n" % (num_docs, num_dim, len(embeddings)))

    d = np.random.rand(num_dim)
    d = d / np.linalg.norm(d)

    obs_hypparams = dict(mu_0=d, C_0=1, m_0=2, sigma_0=0.25)
    components = [vonMisesFisherLogNormal(**obs_hypparams) for _ in range(K)]
    HDP = models.HDP(alpha=alpha, gamma=gamma, obs_distns=components, num_docs=num_docs + 1)

    sgdseq = sgd_passes(tau=tau, kappa=kappa_sgd, datalist=embedding_corpus, minibatchsize=multibatch_size, npasses=1)
    for data, rho_t in progprint(sgdseq, log_file):
        HDP.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(num_docs), rho_t)


    ############# Add data and do mean field

    corpus_topic_predictions = get_topic_predictions(HDP, embedding_corpus)

    write_document_topics(K, corpus_topic_predictions, results_folder)
    write_count_based_topics(corpus_topic_predictions, results_folder, words_corpus)
    log_file.write('################################\n')
    write_prob_based_topics(HDP, K, embedding_corpus, vocabulary, words_corpus, results_folder)
    log_file.close()

    end_time = time()
    duration = int(end_time - start_time)

    with open(results_folder + "/runtime.txt", "w") as runtime_file:
        runtime_file.write(str(duration) + "\n")

def write_prob_based_topics(HDP, K, embedding_corpus, vocabulary, words_corpus, results_folder):
    prob_based_topics_file = open("%s/prob-based.topics" % results_folder, "wb")
    topics_dict = {}
    for word in range(K):
        topics_dict[word] = {}
        for k in vocabulary:
            topics_dict[word][k] = 0
    for doc_nr, doc in enumerate(words_corpus):
        HDP.add_data(np.atleast_2d(embedding_corpus[doc_nr][0].squeeze()), doc_nr)
        HDP.states_list[-1].meanfieldupdate()
        temp_exp = HDP.states_list[-1].all_expected_stats[0]
        for word_nr, word in enumerate(doc):
            for t in range(K):
                try:
                    topics_dict[t][word] += temp_exp[word_nr, t]
                except:
                    pass

    sorted_topics_dict = []
    for t in range(K):
        sorted_topics_dict.append(sorted(topics_dict[t].items(), key=operator.itemgetter(1), reverse=True)[:10])
        # print [word for word, prob in sorted_topics_dict[-1]]

    for t in range(K):
        if len(sorted_topics_dict[t]) > 5:
            top_ordered_words = sorted_topics_dict[t][:20]
            prob_based_topics_file.write(' '.join([doc[0] for doc in top_ordered_words]))
            prob_based_topics_file.write('\n')

    prob_based_topics_file.close()


def write_count_based_topics(corpus_topic_predictions, results_folder, words_corpus):
    count_based_topics_file = open("%s/count-based.topics" % results_folder, "wb")
    unique_topics = np.unique([item for sublist in corpus_topic_predictions for item in sublist])
    topics_dict = {}
    for topic_id in unique_topics:
        topics_dict[topic_id] = []
    for doc_nr in range(len(corpus_topic_predictions)):
        for kk in range(len(corpus_topic_predictions[doc_nr])):
            topics_dict[corpus_topic_predictions[doc_nr][kk]].append(words_corpus[doc_nr][kk])
    for topic_id in unique_topics:
        topics_dict[topic_id] = Counter(topics_dict[topic_id]).most_common(10)
        print [word for word, count in topics_dict[topic_id]]
    for topic_id in unique_topics:
        if len(topics_dict[topic_id]) > 5:
            top_ordered_words = topics_dict[topic_id][:20]
            # print top_ordered_words
            count_based_topics_file.write(' '.join([word for word, count in top_ordered_words]))
            count_based_topics_file.write('\n')
    count_based_topics_file.close()


def write_document_topics(K, corpus_topic_predictions, results_folder):
    documents_topics_file = open("%s/document-topics" % results_folder, "wb")
    for topic_predictions in corpus_topic_predictions:
        doc_length = len(topic_predictions)
        counter = Counter(topic_predictions)
        topic_proportions = [float(counter[k]) / doc_length for k in range(K)]

        documents_topics_file.write(" ".join([str(f) for f in topic_proportions]))
        documents_topics_file.write("\n")
    documents_topics_file.close()


def get_topic_predictions(HDP, embedding_corpus):
    corpus_topic_predictions = []
    for doc in range(len(embedding_corpus)):
        HDP.add_data(np.atleast_2d(embedding_corpus[doc][0].squeeze()), doc)
        HDP.states_list[-1].meanfieldupdate()
        topics = np.argmax(HDP.states_list[-1].all_expected_stats[0], 1).tolist()
        corpus_topic_predictions.append(topics)

    return corpus_topic_predictions


def run_shdp(params, args):
    kappa, tau, alpha, gamma = params
    args.kappa_sgd = kappa
    args.tau = tau
    args.alpha = alpha
    args.gamma = gamma
    number_args = {k: v for k,v in vars(args).iteritems() if "/" not in str(v)}
    print str(datetime.now()) + ": " + str(number_args)
    sys.stdout.flush()
    try:
        HDPRunner(args)
        print str(datetime.now()) + ": " + str(number_args) + " finished!"
        sys.stdout.flush()
    except Exception as e:
        print str(datetime.now()) + ": " + str(number_args) + " failed: " + str(e)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="""This program runs sHDP on a prepared corpus.""",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-embedding-model', type=str, required=True)
    parser.add_argument('-corpus', type=str, required=True)
    parser.add_argument('-vocabulary', type=str, required=True)
    parser.add_argument('-seed', help='Seed for running the model', type=np.int32, required=True)
    parser.add_argument('-alpha', '--alpha', help='alpha hyperparameter for the low level stick breaking process',
                        type=np.float,
                        required=True)
    parser.add_argument('-gamma', '--gamma', help='gamma hyperparameter for the top level stick breaking process',
                        type=np.float, required=True)
    parser.add_argument('-K', help='maximum number of states',
                        type=np.int32, required=True)
    parser.add_argument('-kappa-sgd', help='kappa for SGD', type=np.float, required=True)
    parser.add_argument('-tau', help='tau for SGD', type=np.float, required=True)
    parser.add_argument('-multibatch-size', help='mbsize for SGD', type=np.float, required=True)
    args = parser.parse_args()

    orig_corpus = args.corpus
    orig_vocabulary = args.vocabulary
    orig_embeddings = args.embedding_model

    for dim in [200]:
        args.vocabulary = orig_vocabulary.replace("dim-XXX", "dim-%d" % dim)
        args.embedding_model = orig_embeddings.replace("dim-XXX", "dim-%d" % dim)
        args.corpus = orig_corpus.replace("dim-XXX", "dim-%d" % dim)
        print args.corpus
        sys.stdout.flush()
        HDPRunner(args)

    # params = []
    # for kappa, tau in [(0.6, 0.8), (0.505, 0.8), (0.6, 0.1), (0.6, 10), (0.6, 100)]:
    #     for alpha in [0.5, 0.9, 1.0, 1.5, 10]:
    #         for gamma in [0.5, 1.0, 1.5, 10]:
    #             params.append((kappa, tau, alpha, gamma))
    #
    # p = Pool(5)
    # p.map(partial(run_shdp, args=args), params)
    # HDPRunner(args)

if __name__ == '__main__':
    main()
