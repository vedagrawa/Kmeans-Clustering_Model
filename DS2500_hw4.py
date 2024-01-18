#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ved Agrawal

DS 2500

11/7

Creating a KMeans clustering model based on 
the content of each article

"""

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator

def get_news_data(api_key, keyword, page_size=100, page=1):
    """
    Fetches news articles data from the NewsAPI.

    Args:
    - api_key (str): API key for NewsAPI.
    - keyword (str): Keyword to search for in the articles.
    - page_size (int): The number of articles to fetch per page.
    - page (int): Page number to fetch the data from.

    Returns:
    - list: A list of articles fetched from the API.
    """
    
    endpoint = "https://newsapi.org/v2/everything"
    
    params = {
        
        'apiKey': api_key,
        
        'q': keyword,
        
        'pageSize': page_size,
        
        'page': page,
        
        'language': "en"
    }
    
    response = requests.get(endpoint, params=params)
    
    return response.json()['articles']

def process_data(articles):
    """
    Processes the articles by extracting content and applying TF-IDF vectorization.

    Args:
    - articles (list): A list of articles.

    Returns:
    - sparse matrix: The TF-IDF weighted feature matrix.
    """
    
    contents = [article['content'] for article in articles if article['content']]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    
    X = vectorizer.fit_transform(contents)
    
    return X

def apply_pca(X, n_components=2):
    """
    Applies PCA (Principal Component Analysis) to reduce dimensions of the data.

    Args:
    - X (sparse matrix): The TF-IDF feature matrix.
    - n_components (int): Number of components for PCA.

    Returns:
    - ndarray: The PCA transformed data.
    """
    
    pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X.toarray())
    
    return X_pca

def find_optimal_k(X_pca, max_k=10):
    """
    Finds the optimal number of clusters (k) using the Elbow Method.

    Args:
    - X_pca (ndarray): The PCA transformed data.
    - max_k (int): Maximum number of clusters to test.

    Returns:
    - int: The optimal number of clusters (k).
    """
    
    inertias = []
    
    range_k = range(1, max_k + 1)
    
    for k in range_k:
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        
        kmeans.fit(X_pca)
        
        inertias.append(kmeans.inertia_)
    
    kn = KneeLocator(range_k, inertias, curve='convex', direction='decreasing')
    
    return kn.knee

def fit_kmeans(X_pca, optimal_k):
    """
    Fits the KMeans clustering algorithm to the PCA transformed data.

    Args:
    - X_pca (ndarray): The PCA transformed data.
    - optimal_k (int): The optimal number of clusters.

    Returns:
    - array: The cluster assignments for each data point.
    """
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    
    kmeans.fit(X_pca)
    
    return kmeans.labels_

def plot_pca_clusters(X_pca, cluster_assignments):
    """
    Plots the PCA clusters with a legend indicating each cluster.

    Args:
    - X_pca (ndarray): The PCA transformed data.
    - cluster_assignments (array): The cluster assignments from KMeans.
    """
    
    plt.figure(figsize=(8, 6))
    
    unique_clusters = np.unique(cluster_assignments)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    for cluster, color in zip(unique_clusters, colors):
        
        # Selecting only data points that belong to the current cluster
        
        cluster_data = X_pca[cluster_assignments == cluster]
        
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=[color], label=f'Cluster {cluster+1}')

    plt.title('PCA Scatterplot with KMeans Clusters')
    
    plt.xlabel('PCA Component 1')
    
    plt.ylabel('PCA Component 2')
    
    plt.legend()
    
    plt.show()


def plot_elbow_method(X_pca, max_k=10):
    """
    Plots the Elbow Method graph to determine the optimal number of clusters.

    Args:
    - X_pca (ndarray): The PCA transformed data.
    - max_k (int): Maximum number of clusters to test.
    """
    
    inertias = []
    
    range_k = range(1, max_k + 1)

    for k in range_k:
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        
        kmeans.fit(X_pca)
        
        inertias.append(kmeans.inertia_)

    kn = KneeLocator(range_k, inertias, curve='convex', direction='decreasing')
    
    optimal_k = kn.knee

    plt.figure(figsize=(8, 6))
    
    plt.plot(range_k, inertias, marker='o')
    
    plt.title('Elbow Method For Optimal k')
    
    plt.xlabel('Number of clusters')
    
    plt.ylabel('Inertia')
    
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    
    plt.legend()
    
    plt.show()

def main():
    """
    Main function to execute the analysis process.
    """
    
    API_KEY = '0d0a505b113f464b92b76e51061493a1'
    
    keyword = "technology", "politics"
    
    articles = get_news_data(API_KEY, keyword)
    
    X = process_data(articles)
   
    X_pca = apply_pca(X)
    
    optimal_k = find_optimal_k(X_pca)
    
    cluster_assignments = fit_kmeans(X_pca, optimal_k)

    first_article_pca = X_pca[0]
    
    print("First article PCA components:", first_article_pca)

    print("Optimal number of clusters (k):", optimal_k)

    # Counting articles in each cluster
    
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    
    cluster_distribution = dict(zip(unique + 1, counts))
    
    print("Number of articles in each cluster:", cluster_distribution)
    
    plot_pca_clusters(X_pca, cluster_assignments)

    plot_elbow_method(X_pca)

if __name__ == "__main__":
    main()
