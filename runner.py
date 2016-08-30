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


def HDPRunner(args):
    seed = args.seed
    K = args.K
    alpha = args.alpha
    gamma = args.gamma
    tau = args.tau
    kappa_sgd = args.kappa_sgd
    multibatch_size = args.multibatch_size

    ################# Data generation
    texts, vectors_dict = read_corpus(args)
    num_dim = len(vectors_dict["word"])

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
    count_based_topics_file = open("%s/count-based.topics" % results_folder, "wb")
    prob_based_topics_file = open("%s/prob-based.topics" % results_folder, "wb")
    documents_topics_file = open("%s/document-topics" % results_folder, "wb")

    ########### Runner

    def glovize_data(list_of_texts):
        all_data = []
        for text in list_of_texts:
            temp_list = []
            for word, count in text:
                if word in vectors_dict:
                    pass
                elif word.capitalize() in vectors_dict:
                    word = word.capitalize()
                elif word.upper() in vectors_dict:
                    word = word.upper()

                temp_list.append((np.array(vectors_dict[word]).astype(float), count))
            all_data.append(np.array(temp_list))
        return all_data

    def glovize_data_wo_count(list_of_texts):
        all_data = []
        all_avail_words = []
        for text in list_of_texts:
            temp_list = []
            temp_list_words = []
            for word, count in text:
                if word in vectors_dict:
                    pass
                elif word.capitalize() in vectors_dict:
                    word = word.capitalize()
                elif word.upper() in vectors_dict:
                    word = word.upper()

                temp_list.append((np.array(vectors_dict[word]).astype(float), count))
                temp_list_words.append(word)
            all_data.append(np.array(temp_list))
            all_avail_words.append(np.array(temp_list_words))
        return all_data, all_avail_words

    temp1 = glovize_data(texts)
    temp2 = glovize_data_wo_count(texts)[0]
    temp2 = zip(temp2, range(len(temp2)))
    real_data = temp2[:]
    num_docs = len(real_data)
    print 'num_docs', num_docs
    temp_words = glovize_data_wo_count(texts)[1]
    temp_words = temp_words[:num_docs]
    vocabulary = np.unique([j for i in temp_words for j in i])

    training_size = num_docs
    all_words = []
    for d in temp1:
        for w in d:
            all_words.append(w[0])

    np.random.seed(seed)

    d = np.random.rand(num_dim, )
    d = d / np.linalg.norm(d)
    obs_hypparams = dict(mu_0=d, C_0=1, m_0=2, sigma_0=0.25)
    components = [vonMisesFisherLogNormal(**obs_hypparams) for itr in range(K)]

    HDP = models.HDP(alpha=alpha, gamma=gamma, obs_distns=components, num_docs=num_docs + 1)

    sgdseq = sgd_passes(tau=tau, kappa=kappa_sgd, datalist=real_data, minibatchsize=multibatch_size, npasses=1)
    for t, (data, rho_t) in progprint(enumerate(sgdseq)):
        HDP.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(training_size), rho_t)

    print "Finished Training"

    ############# Add data and do mean field

    ### count based topics
    all_topics_pred = []
    all_topics_unique = []
    for i in range(num_docs):
        HDP.add_data(np.atleast_2d(real_data[i][0].squeeze()), i)
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

    unique_topics = np.unique(all_topics_unique)
    topics_dict = {}
    for j in unique_topics:
        topics_dict[j] = []
    for k in range(num_docs):
        for kk in range(len(all_topics_pred[k])):
            topics_dict[all_topics_pred[k][kk]].append(temp_words[k][kk])

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

    for idx, doc in enumerate(temp_words):
        HDP.add_data(np.atleast_2d(real_data[idx][0].squeeze()), idx)
        HDP.states_list[-1].meanfieldupdate()
        temp_exp = HDP.states_list[-1].all_expected_stats[0]
        for idw, word in enumerate(doc):
            for t in range(K):
                topics_dict[t][word] += temp_exp[idw, t]

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
