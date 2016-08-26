#!/usr/bin/env python2.7

import argparse
from argparse import RawTextHelpFormatter
from collections import Counter
from core.distributions import vonMisesFisherLogNormal
import csv
import sys
import pickle as pk
import numpy as np
import os
from HDP import models
from HDP.util.general import sgd_passes
from HDP.util.text import progprint
import operator

project_path = ''
results_path = project_path + 'results/'


def HDPRunner(args):
    infseed = args['infSeed']  # 1
    K = args['Nmax']
    alpha = args['alpha']  # 1
    gamma = args['gamma']  # 2
    tau = args['tau']
    kappa_sgd = args['kappa_sgd']
    mbsize = args['mbsize']
    dataset = args['dataset']

    results_folder = "results/%s/infseed-%d.topics-%d.alpha-%s.gamma-%s.kappa-%s.tau-%s.batch-%d" % (
        dataset,
        infseed,
        K,
        str(float(alpha)).replace(".", "-"),
        str(float(gamma)).replace(".", "-"),
        str(float(kappa_sgd)).replace(".", "-"),
        str(float(tau)).replace(".", "-"),
        mbsize
    )
    try:
        os.mkdir(results_folder)
    except:
        pass
    count_based_topics_file = open("%s/count-based.topics" % results_folder, "wb")
    prob_based_topics_file = open("%s/prob-based.topics" % results_folder, "wb")
    documents_topics_file = open("%s/document-topics" % results_folder, "wb")

    ################# Data generation
    temp_file = open(project_path + 'data/' + dataset + '/texts.pk', 'rb')
    texts = pk.load(temp_file)
    # print len(texts)
    # print texts[0]
    temp_file.close()

    print 'Loading the glove dict file....'
    csv.field_size_limit(sys.maxsize)
    vectors_file = open(project_path + 'data/' + dataset + '/wordvec.pk', 'rb')
    vectors_dict = pk.load(vectors_file)
    num_dim = len(vectors_dict["word"])

    # TODO: normalize word vectors to size 1

    ########### Runner

    def glovize_data(list_of_texts):
        all_data = []
        for text in list_of_texts:
            temp_list = []
            for word in text:
                try:
                    temp_list.append((np.array(vectors_dict[word[0]]).astype(float), word[1]))
                except:
                    pass
            all_data.append(np.array(temp_list))
        return all_data

    def glovize_data_wo_count(list_of_texts):
        all_data = []
        all_avail_words = []
        for text in list_of_texts:
            temp_list = []
            temp_list_words = []
            for word in text:
                try:
                    temp_list.append((np.array(vectors_dict[word[0]]).astype(float), word[1]))
                    temp_list_words.append(word[0])
                except:
                    pass
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

    np.random.seed(infseed)

    d = np.random.rand(num_dim, )
    d = d / np.linalg.norm(d)
    obs_hypparams = dict(mu_0=d, C_0=1, m_0=2, sigma_0=0.25)
    components = [vonMisesFisherLogNormal(**obs_hypparams) for itr in range(K)]

    HDP = models.HDP(alpha=alpha, gamma=gamma, obs_distns=components, num_docs=num_docs + 1)

    sgdseq = sgd_passes(tau=tau, kappa=kappa_sgd, datalist=real_data, minibatchsize=mbsize, npasses=1)
    for t, (data, rho_t) in progprint(enumerate(sgdseq)):
        HDP.meanfield_sgdstep(data, np.array(data).shape[0] / np.float(training_size), rho_t)

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

    sorted_topics_dict = []
    print '################################'
    for t in range(K):
        sorted_topics_dict.append(sorted(topics_dict[t].items(), key=operator.itemgetter(1), reverse=True)[:20])
        print sorted_topics_dict[-1]

    for t in range(K):
        if len(sorted_topics_dict[t]) > 5:
            top_ordered_words = sorted_topics_dict[t][:20]
            prob_based_topics_file.write(' '.join([i[0] for i in top_ordered_words]))
            prob_based_topics_file.write('\n')
    prob_based_topics_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This program runs sHDP on a prepared corpus.
    sample argument setting is as follows:
    python runner.py -is 1 -alpha 1 -gamma 2 -Nmax 40 -kappa_sgd 0.6 -tau 0.8 -mbsize 10 -dataset nips
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-is', '--infSeed', help='Seed for running the model', type=np.int32, required=True)
    parser.add_argument('-alpha', '--alpha', help='alpha hyperparameter for the low level stick breaking process',
                        type=np.float,
                        required=True)
    parser.add_argument('-gamma', '--gamma', help='gamma hyperparameter for the top level stick breaking process',
                        type=np.float, required=True)
    parser.add_argument('-Nmax', '--Nmax', help='maximum number of states',
                        type=np.int32, required=True)
    parser.add_argument('-kappa_sgd', '--kappa_sgd', help='kappa for SGD', type=np.float, required=True)
    parser.add_argument('-tau', '--tau', help='tau for SGD', type=np.float, required=True)
    parser.add_argument('-mbsize', '--mbsize', help='mbsize for SGD', type=np.float, required=True)
    parser.add_argument('-dataset', '--dataset', help='choose one of nips 20news wiki', required=True)
    args = vars(parser.parse_args())
    print args
    HDPRunner(args)
