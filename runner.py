#!/usr/bin/env python2.7

import argparse
import csv
import operator
import os
import pickle as pk
import sys
from argparse import RawTextHelpFormatter
from collections import Counter

import numpy as np
from sklearn.preprocessing import normalize

from HDP import models
from HDP.util.general import sgd_passes
from HDP.util.text import progprint
from core.distributions import vonMisesFisherLogNormal

project_path = ''
results_path = project_path + 'results/'

def build_embedding_corpus(corpus, embeddings):
    corpus_embeddings = []
    words = []
    for doc in corpus:
        temp_list = []
        words_doc = []
        for word, count in doc:
            if word in embeddings:
                pass
            elif word.capitalize() in embeddings:
                word = word.capitalize()
            elif word.upper() in embeddings:
                word = word.upper()

            try:
                temp_list.append((np.array(embeddings[word]).astype(float), count))
            except:
                pass
            words_doc.append(word)
        corpus_embeddings.append(np.array(temp_list))
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

    print "Nr. embeddings: %d, Nr. words in vocabulary: %d" % (len(embeddings), len(vocab))
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
    seed = args.seed
    K = args.K
    alpha = args.alpha
    gamma = args.gamma
    tau = args.tau
    kappa_sgd = args.kappa_sgd
    multibatch_size = args.multibatch_size

    corpus_x, embeddings = read_corpus(args)
    corpus, embeddings_x = read_other_corpus()
    print "other embeddings norm"
    for word in embeddings_x:
        norm = np.linalg.norm(embeddings_x[word])
        print norm
    num_dim = len(embeddings["word"])

    results_folder = "results/%s/dim-%d.seed-%d.topics-%d.alpha-%s.gamma-%s.kappa-%s.tau-%s.batch-%d" % (
        "20news",
        num_dim,
        seed,
        K,
        str(float(alpha)).replace(".", "-"),
        str(float(gamma)).replace(".", "-"),
        str(float(kappa_sgd)).replace(".", "-"),
        str(float(tau)).replace(".", "-"),
        multibatch_size
    )
    try:
        os.mkdir(results_folder)
    except Exception:
        pass

    ########### Runner
    embedding_corpus, words = build_embedding_corpus(corpus, embeddings)
    embedding_corpus = zip(embedding_corpus, range(len(embedding_corpus)))
    num_docs = len(embedding_corpus)
    words = words[:num_docs]
    vocabulary = np.unique([j for i in words for j in i])

    print "{'num_docs': %d, 'num_dim': %d}" % (num_docs, num_dim)

    training_size = num_docs
    np.random.seed(seed)

    d = np.random.rand(num_dim)
    d = d / np.linalg.norm(d)
    obs_hypparams = dict(mu_0=d, C_0=1, m_0=2, sigma_0=0.25)
    components = [vonMisesFisherLogNormal(**obs_hypparams) for _ in range(K)]

    HDP = models.HDP(alpha=alpha, gamma=gamma, obs_distns=components, num_docs=num_docs + 1)

    sgdseq = sgd_passes(tau=tau, kappa=kappa_sgd, datalist=embedding_corpus, minibatchsize=multibatch_size, npasses=1)
    for t, (data, rho_t) in progprint(enumerate(sgdseq)):
        # print rho_t
        HDP.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(training_size), rho_t)

    print "Finished Training"


    count_based_topics_file = open("%s/count-based.topics" % results_folder, "wb")
    prob_based_topics_file = open("%s/prob-based.topics" % results_folder, "wb")
    documents_topics_file = open("%s/document-topics" % results_folder, "wb")

    ############# Add data and do mean field

    all_topics_pred = []
    all_topics_unique = []
    for i in range(num_docs):
        HDP.add_data(np.atleast_2d(embedding_corpus[i][0].squeeze()), i)
        HDP.states_list[-1].meanfieldupdate()
        predictions = np.argmax(HDP.states_list[-1].all_expected_stats[0], 1)
        all_topics_pred.append(predictions)
        all_topics_unique.extend(np.unique(predictions))
        # print predictions
        topics = predictions.tolist()
        doc_length = len(topics)
        counter = Counter(topics)
        topic_proportions = [float(counter[k]) / doc_length for k in range(K)]
        documents_topics_file.write(" ".join([str(f) for f in topic_proportions]))
        documents_topics_file.write("\n")

    documents_topics_file.close()
    print "Finshed document topics"

    ### count based topics
    unique_topics = np.unique(all_topics_unique)
    topics_dict = {}
    for j in unique_topics:
        topics_dict[j] = []
    for k in range(num_docs):
        for kk in range(len(all_topics_pred[k])):
            topics_dict[all_topics_pred[k][kk]].append(words[k][kk])

    for t in unique_topics:
        topics_dict[t] = Counter(topics_dict[t]).most_common(30)
        print topics_dict[t]

    # now there is a dictionary
    for t in unique_topics:
        if len(topics_dict[t]) > 5:
            top_ordered_words = topics_dict[t][:20]
            # print top_ordered_words
            count_based_topics_file.write(' '.join([i[0] for i in top_ordered_words]))
            count_based_topics_file.write('\n')
    count_based_topics_file.close()
    print "Finshed count-based topics"

    ### prob based topics
    topics_dict = {}
    for j in range(K):
        topics_dict[j] = {}
        for k in vocabulary:
            topics_dict[j][k] = 0

    for idx, doc in enumerate(words):
        HDP.add_data(np.atleast_2d(embedding_corpus[idx][0].squeeze()), idx)
        HDP.states_list[-1].meanfieldupdate()
        temp_exp = HDP.states_list[-1].all_expected_stats[0]
        for idw, word in enumerate(doc):
            for t in range(K):
                try:
                    topics_dict[t][word] += temp_exp[idw, t]
                except:
                    pass

    print '################################'

    sorted_topics_dict = []
    for t in range(K):
        sorted_topics_dict.append(sorted(topics_dict[t].items(), key=operator.itemgetter(1), reverse=True)[:20])
        print sorted_topics_dict[-1]

    for t in range(K):
        if len(sorted_topics_dict[t]) > 5:
            top_ordered_words = sorted_topics_dict[t][:20]
            prob_based_topics_file.write(' '.join([i[0] for i in top_ordered_words]))
            prob_based_topics_file.write('\n')
    prob_based_topics_file.close()


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
    print vars(args)
    HDPRunner(args)


if __name__ == '__main__':
    main()
