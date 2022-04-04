# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:24:01 2021

@author: ktlco
"""
import numpy as np
import os
from os.path import abspath, exists
import csv
from scipy import sparse
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

def build_clusters(k, x):
    x = x[:, 0:k]    
    x = x/np.repeat(np.sqrt(np.sum(x*x, axis=1).reshape(-1, 1)), k, axis=1)
    # scatter
    #plt.scatter(x[:, 0], x[:, 1])
    #plt.show()
    
    # k-means
    kmeans = KMeans(n_clusters=k).fit(x)
    c_idx = kmeans.labels_
    return c_idx

def main():
    # inspiration taken from demo code test_football.py
    dirpath = os.getcwd()    
    node_file = dirpath + '//data//nodes.txt'
    edge_file = dirpath + '//data//edges.txt'
    
    # create graph from edge file
    if exists(edge_file):
        with open(edge_file) as e:
            reader = csv.reader(e, delimiter="\t")
            lines = list(reader)
        edges = np.array(lines).astype(int)
    
    # load node data file
    if exists(node_file):
        with open(node_file) as nf:
            reader = csv.reader(nf, delimiter="\t")
            nodes = [[int(row[0]), int(row[2])] for row in reader if row] 
        nodes = np.array(nodes).astype(int)
    
    
    # re-index nodes
    n = len(np.unique(edges))
    node_ids = np.unique(edges)
    node_ids.sort()
    adj_ids = np.array(range(0, n))
    node_to_adj = dict(zip(node_ids, adj_ids))
    adj_to_node = dict(zip(adj_ids, node_ids))    
    node_to_label = dict(zip(nodes[:,0], nodes[:,1]))   
    
    # get indices of 1's for adjacency matrix
    i = [node_to_adj[x] for x in edges[:,0]]
    j = [node_to_adj[x] for x in edges[:,1]]
    v = np.ones((edges.shape[0], 1)).flatten()
    
    # create adjacency matrix
    A = sparse.coo_matrix((v, (i, j)), shape=(n, n))
    A = (A + np.transpose(A))/2
    
    D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
    L = D @ A @ D
    
    # get eigenvectors
    x, v, _= np.linalg.svd(L)
    
    # build clusters
    k_list = [2, 5, 10, 20]
    #k_list=[2]
    for k in k_list:
        c_idx = build_clusters(k, x)
        # calculate majority labels for each cluster in k
        node_ids = np.array([adj_to_node[z] for z in range(0, len(c_idx))]    )
        node_labels = np.array([node_to_label[z] for z in node_ids])
        for i in range(0, k):
            idx = c_idx==i
            cur_nodes = node_ids[idx]
            cur_labels = node_labels[idx]
            if len(cur_labels)==0:
                maj_label = "n/a"
                mm_rate = 0
            else:
                c = Counter(cur_labels)
                maj_label = c.most_common(1)[0][0]
                mm_rate = 1.0 * sum(cur_labels!=maj_label) / len(cur_labels)
            print(f"For k = {k}, the majority label in cluster {i} is {maj_label} with mismatch rate {mm_rate:.2f}")   
    
    # determine best k based on mismatch rate
    k_list = range(2, 101)
    mm_plotY = np.empty((len(k_list), 1))
    mm_plotX = np.empty((len(k_list), 1))
    q = 0
    for k in k_list:
        mm_plotX[q] = k
        c_idx = build_clusters(k, x)
        # calculate majority labels for each cluster in k
        node_ids = np.array([adj_to_node[z] for z in range(0, len(c_idx))]    )
        node_labels = np.array([node_to_label[z] for z in node_ids])
        mm_rates = np.empty((k, 1))
        for i in range(0, k):
            idx = c_idx==i
            cur_nodes = node_ids[idx]
            cur_labels = node_labels[idx]
            if len(cur_labels)==0:
                    maj_label = "n/a"
                    mm_rates[i] = 0
            else:
                c = Counter(cur_labels)
                maj_label = c.most_common(1)[0][0]
                mm_rates[i] = 1.0 * sum(cur_labels!=maj_label) / len(cur_labels)
        mm_rate = np.average(mm_rates)
        mm_plotY[q] = mm_rate
        q = q+1
        print(f"For k = {k}, average mismatch rate is {mm_rate:.2f}")   
    
    # plot change in mismatch rate by k
    plt.plot(mm_plotX, mm_plotY)
    plt.xlabel("k")
    plt.ylabel("avg mismatch rate")
    plt.axis([0, 100, 0, 1])
    plt.show()
    

if __name__ == '__main__':
    main()
