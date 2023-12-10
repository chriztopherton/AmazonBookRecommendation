# -*- coding: utf-8 -*-
"""Data245_Project_Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GNNuwWakSaZ8GBvHRTF1VfrMkHhTNzEo

# Install & Import Packages
"""

!pip install pyspark stemming sparknlp langchain sentence-transformers faiss-gpu llama-cpp-python transformers

# Pre-processing
import glob
import random
import re
import string
import json
from stemming.porter2 import stem
from nltk.corpus import stopwords

# Spectral clustering
import numpy as np
import pandas as pd
from sklearn import cluster
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh
from matplotlib import pylab, pyplot as plt

# Progress bar
from tqdm import tqdm
from time import sleep

import pickle

import os
import gzip
from urllib.request import urlopen

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords = set(STOPWORDS)

# Data manipulation and visualization
import pandas as pd
import numpy as np
import random
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
from tqdm.notebook import tqdm

# Natural Language Processing
from nltk import download
download('stopwords')

#pyspark
from pyspark.sql import SparkSession

#KMeans clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

#DeepWalk skip-gram
from gensim.models import Word2Vec

# langchain methods for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import pipeline
import textwrap
from pprint import pprint

"""# Connect to Google Drive and navigate to shared folder"""

from google.colab import drive

drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/Shareddrives/'DATA 245 Team 6'/Datasets/
# %ls

"""# Read in source data files and pre-process"""

filepath = glob.glob("raw files/amazon-meta.txt*")
fil = filepath[0]
text = open(fil,'r',encoding='utf-8',errors='ignore')

"""## pre-process `amazon-meta.txt`"""

amazon = {}
# Declaring the tuple of various feature values
(Id, ASIN, title, categories, group, connections, salesrank, tot_reviews, avg_rating, degree_centrality, clustering_coeff) = ("","","","","","", 0, 0, 0.0, 0, 0.0)

with open(fil,'r',encoding='utf-8',errors='ignore') as text:
  for line in text:
      line = line.strip()
      # Strip ID
      if(line.startswith("Id")):
          Id = line[3:].strip()
      # Strip ASIN, which is the node
      elif(line.startswith("ASIN")):
          ASIN = line[5:].strip()
      # Strip Title
      elif(line.startswith("title")):
          title = line[6:].strip()
          title = ' '.join(title.split())
      # Strip Group
      elif(line.startswith("group")):
          group = line[6:].strip()
      # Strip sales rank
      elif(line.startswith("salesrank")):
          salesrank = line[10:]
      # Strip similar, which are the connections
      elif(line.startswith("similar")):
          a = line.split()
          connections = ' '.join([i for i in a[2:]])
      # Strip categories
      elif(line.startswith("categories")):
          b = line.split()
          # Converted to lowercase
          categories = ' '.join((text.readline()).lower() for i in range(int(b[1].strip())))
          # Remove punctuations
          categories = re.compile('[%s]' % re.escape(string.digits + string.punctuation)).sub(' ', categories)
          # Remove stopwords
          categories = ' '.join(set(categories.split()) - set(stopwords.words('english')))
          # Concatenate the words
          categories = ' '.join(stem(word) for word in categories.split())
      # Count the reviews, and average rating
      elif(line.startswith("reviews")):
          c = line.split()
          tot_reviews = c[2].strip()
          avg_rating = c[7].strip()
      # Handling the exception for cases where instances are missing
      elif(line==""):
          try:
              exception = {}
              if(ASIN!=""):
                  amazon[ASIN] = exception
              exception['Id'] = Id
              exception['title'] = title
              exception['group'] = group
              exception['categories'] = ' '.join(set(categories.split()))
              exception['connections'] = connections
              exception['salesrank'] = int(salesrank)
              #exception['similar'] = ' '.join(set(connections.split()))
              exception['tot_reviews'] = int(tot_reviews)
              exception['avg_rating'] = float(avg_rating)
              exception['degree_centrality'] = degree_centrality
              exception['clustering_coeff'] = clustering_coeff
          except NameError:
              continue
          (Id, ASIN, title, categories, group, connections, salesrank, tot_reviews, avg_rating, degree_centrality, clustering_coeff) = ("","","","","","", 0, 0, 0.0, 0, 0.0)

with open('amazon_products.pickle', 'wb') as handle:
    pickle.dump(amazon, handle, protocol=pickle.HIGHEST_PROTOCOL)

def product_df(amzn,grp):
  prod = {}
  for key, value in amzn.items():
    # Check for the case where value is "Music", and only then choose
      if(value['group'] == grp):
          prod[key] = amzn[key]

  # Join the connections
  for key, value in prod.items():
      prod[key]['connections'] = ' '.join([connection for connection in value['connections'].split() if connection in prod.keys()])

  # Convert to dataframe
  prod_df = pd.DataFrame.from_dict(prod)
  prod_df = prod_df.transpose()

  return prod,prod_df

with open('raw files/amazon_products.pickle', 'rb') as handle:
    amzn = pickle.load(handle)

books_dict,books = product_df(amzn, "Book")
len(books) #393561

books.to_csv("books_amzn_meta.csv")

books_meta = pd.read_csv("dataset_for_modeling/books_amzn_meta.csv").rename(columns={'Unnamed: 0':"asin"})

"""## pre-process `Books_5.json.gz`"""

array= []
count = 1

# Create a JSON file to store the extracted data
jsonFile = open("./Books_rating_category.json", "w")
jsonFile.write("[") # Opening bracket to denote a list of JSON objects

# Open the gzipped file and extract the required fields
with gzip.open('Books_5.json.gz') as f:
    for l in f:
        # Load each line as a JSON object
        d = json.loads(l.strip())

        # Check if the required fields are present in the JSON object
        if 'reviewText' in d and 'asin' in d and 'overall' in d and 'reviewerID' in d and 'summary' in d:
            et_dict = {"asin":  d["asin"],
                        "overall" : d["overall"],
                        "reviewText" : d["reviewText"],
                        "reviewerID" : d["reviewerID"],
                        "summary" : d["summary"],}
            # Append the extracted fields to the array
            array.append(et_dict)
            # Convert the dictionary to a JSON string
            jsonString = json.dumps(et_dict)+",\n"
            # Write the JSON string to the file
            jsonFile.write(jsonString)

# Closing bracket to denote the end of the list of JSON objects
jsonFile.write("]")
# Close the JSON file
jsonFile.close()

"""## Convert to pyspark for big data processing"""

# create a SparkSession
spark = SparkSession.builder.appName("booksSpark").getOrCreate()

df = spark.read.json("dataset_for_modeling/Books_rating_category.json")

# show the first 10 rows of the DataFrame
df.show(10)

df.printSchema()

books_meta = pd.read_csv("books_preprocessed.csv").rename(columns={'Unnamed: 0':'asin'}).drop('degree_centrality',axis=1)
print(books_meta.columns)

books_meta_spark =spark.createDataFrame(books_meta)
books_meta_spark.printSchema()
books_meta_spark.show()

print((books_meta_spark.count(), len(books_meta_spark.columns)))

books_nx_reviews = books_meta_spark.join(df,['asin'],"inner")
print((books_nx_reviews.count(), len(books_nx_reviews.columns)))
books_nx_reviews.printSchema()

books_ex = books_nx_reviews.filter(books_nx_reviews.asin == "0932081258").toPandas()

wordcloud = WordCloud().generate(str(books_ex['reviewText'].values))
plt.imshow(wordcloud)

"""# Graph-Based method

## Jaccard Coefficient: Transforming into Networkx data structure
"""

Copurchase_Graph = nx.Graph()
for asin, metadata in books_dict.items():
    Copurchase_Graph.add_node(asin)
    for a in metadata['connections'].split():
        Copurchase_Graph.add_node(a.strip())
        similarity = 0
        n1 = set((books_dict[asin]['categories']).split())
        n2 = set((books_dict[a]['categories']).split())
        n1In2 = n1 & n2
        n1Un2 = n1 | n2
        if(len(n1In2)) > 0:
            similarity = round(len(n1In2) / len(n1Un2), 2)
            #print(similarity)
            Copurchase_Graph.add_edge(asin, a.strip(), weight = similarity)

dc = nx.degree(Copurchase_Graph)
for asin in nx.nodes(Copurchase_Graph):
    metadata = books_dict[asin]
    metadata['DegreeCentrality'] = int(dc[asin])
    ego = nx.ego_graph(Copurchase_Graph, asin, radius = 1)
    cluster_coeff = round(nx.average_clustering(ego), 2)
    #print(cluster_coeff)
    metadata['clustering_coeff'] = cluster_coeff
    books_dict[asin] = metadata

Amazon_Books_File = open('books_clustering_coeff.txt', 'w', encoding = 'utf-8', errors = 'ignore')

Amazon_Books_File.write("Id\t" + "ASIN\t" + "title\t" + "categories\t" + "group\t" + "similar\t" + "salesrank\t" + "tot_reviews\t" + "avg_rating\t" "degree_centrality\t" +
                        "clustering_coeff\n")

for asin, metadata in books_dict.items(): # converting the meta-data into txt file
     Amazon_Books_File.write(metadata['Id'] + "\t" + \
                             asin + "\t" +  \
                             metadata['title'] + "\t" + \
                             metadata['categories'] + "\t" + \
                             metadata['group'] + "\t" +  \
                             metadata['similar'] + "\t" + \
                             str(metadata['salesrank']) + "\t" + \
                             str(metadata['tot_reviews']) + "\t" +
                             str(metadata['avg_rating']) + "\t" + \
                             str(metadata['degree_centrality']) + "\t" + \
                             str(metadata['clustering_coeff']) + "\n")

Amazon_Books_File.close()

# writing the adjacency edge list
Amazon_Books_File = open("amazon-books-copurchase.edgelist", 'wb')
nx.write_weighted_edgelist(Copurchase_Graph, Amazon_Books_File)
Amazon_Books_File.close()

Books_File = open('books_clustering_coeff.txt', 'r', encoding = 'utf-8', errors = 'ignore')
Books = {}
Books_File.readline()
for line in Books_File:
    cell = line.split("\t")
    MetaData = {}
    MetaData['Id'] = cell[0].strip()
    ASIN = cell[1].strip()
    MetaData['title'] = cell[2].strip()
    MetaData['categories'] = cell[3].strip()
    MetaData['group'] = cell[4].strip()
    MetaData['similar'] = cell[5].strip()
    MetaData['salesrank'] = int(cell[6].strip())
    MetaData['tot_reviews'] = int(cell[7].strip())
    MetaData['avg_rating'] = float(cell[8].strip())
    MetaData['degree_centrality'] = int(cell[9].strip())
    MetaData['clustering_coeff'] = float(cell[10].strip())
    Books[ASIN] = MetaData
Books_File.close()

Books_File = open("amazon-books-copurchase.edgelist", "rb")
Copurchase_Graph = nx.read_weighted_edgelist(Books_File)
Books_File.close()

Copurchase_Graph = nx.read_weighted_edgelist("dataset_for_modeling/amazon-books-copurchase.edgelist")

print("Looking for Recommendations for Customer Purchasing this Book: ")
print("---------------------------------------------------------------")
Purchased_ASIN = '0805047905'
print("ASIN = ", Purchased_ASIN)
print("title = ", Books[Purchased_ASIN]['title'])
print("salesrank = ", Books[Purchased_ASIN]['salesrank'])
print("tot_reviews = ", Books[Purchased_ASIN]['tot_reviews'])
print("avg_rating = ", Books[Purchased_ASIN]['avg_rating'])
print("degree_centrality = ", Books[Purchased_ASIN]['degree_centrality'])
print("clustering_coeff = ", Books[Purchased_ASIN]['clustering_coeff'])

with open("books_dict.json", "w") as outfile:
    outfile.write(json.dumps(books_dict, indent=4))

books_df = pd.DataFrame.from_dict(books_dict)
books_df = books_df.transpose()
books_df.to_csv("books_preprocessed.csv")

"""## DeepWalk: Transforming into Networkx data structure

"""

train = books_meta.dropna()
title_conn = train[['asin','title','categories','connections']].copy()
title_conn['categories_split'] = title_conn['categories'].apply(lambda x:x.split())
asin_title = dict(zip(title_conn.asin, title_conn.title))
title_cat = dict(zip(title_conn.title, title_conn.categories_split))

co_purchase_pairs = {}

for i,row in tqdm(title_conn.iterrows()):
  conns = row['connections'].split(' ')
  pair = [(row['title'],asin_title.get(i)) for i in conns if asin_title.get(i)is not None ]
  for p in pair:
    if p[0] != p[1]:
      if p in co_purchase_pairs.keys():
        co_purchase_pairs[p] +=1
      else:
        co_purchase_pairs[p] = 1

with open('word2vec/co_purchase_pairs.pkl', 'wb') as f:
  pickle.dump(co_purchase_pairs, f)

from math import log, e

def entropy(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent

# cat1 = train.query("asin == '0827229534'").categories.values[0].split()
# cat2 = train.query("asin == '0738700797'").categories.values[0].split()
# print(cat1 + cat2)
# entropy(cat1 + cat2)

weighted_co_purchase_pairs =  {}

for i,row in tqdm(title_conn.iterrows()):
  conns = row['connections'].split(' ')
  pair = [(row['title'],asin_title.get(i)) for i in conns if asin_title.get(i)is not None ]
  for p in pair:
    if p[0] != p[1]:
      if p in weighted_co_purchase_pairs.keys():
        weighted_co_purchase_pairs[p] += (1 * entropy(title_cat.get(p[0]) + title_cat.get(p[1])))
      else:
        weighted_co_purchase_pairs[p] = (1 * entropy(title_cat.get(p[0]) + title_cat.get(p[1])))

with open('word2vec/weighted_co_purchase_pairs.pkl', 'wb') as f:
  pickle.dump(weighted_co_purchase_pairs, f)

d = [dict(sorted(co_purchase_pairs.items(), key=lambda x:x[1], reverse=True))]
d_weighted = [dict(sorted(weighted_co_purchase_pairs.items(), key=lambda x:x[1], reverse=True))]

# Create and populate the graph object
G = nx.Graph()

for key, val in d[0].items():
    G.add_edge(key[0], key[1], weight = val)

# Take a look at how many nodes there are in the graph; too many and it's uncomfortable to visualise
nodes = list(G.nodes)
len(nodes)

# Prune the plot so we only have items that are matched with at least two others
for node in nodes:
    try:
        if G.degree[node] <= 1:
            G.remove_node(node)
    except:
        print(f'error with node {node}')

nodes = list(G.nodes)
len(nodes)

def random_walk(graph, node, weighted=False, n_steps = 5):
    ''' Function that takes a random walk along a graph
    Most code borrowed from: https://towardsdatascience.com/deepwalk-its-behavior-and-how-to-implement-it-b5aac0290a15
    params: graph - the networkx graph object
            node - the node to start on
            weighted - whether the probability of moving to a new node is determined by the edge weight
            n_steps - the number of steps to take
    Returns a "string" of the nodes visited
    '''
    local_path = [str(node),]
    target_node = node
      # Take n_steps random walk away from the node (can return to the node)
    for _ in range(n_steps):
        #print(list(nx.all_neighbors(graph, target_node)))
        neighbours = list(nx.all_neighbors(graph, target_node))
        if weighted:
            # sample in a weighted manner
            target_node = random.choices(neighbours,weights[target_node])[0]
        else:
            target_node = random.choice(neighbours)
        local_path.append(str(target_node))

    return local_path

##  Take a weighted walk  ##
walk_paths_weighted = []

i = 0
for node in tqdm(G.nodes()):
    # We take 10 random walk from each node
    try:
      for _ in range(10):
          walk_paths_weighted.append(random_walk(G, node, weighted=True))
    except:
      pass

# with open('walk_paths_weighted.pkl', 'wb') as f:
#   pickle.dump(walk_paths_weighted, f)

with open("word2vec/walk_paths_weighted.pkl", "rb") as input_file:
  walk_paths_weighted = pickle.load(input_file)

"""## FAISS: create vector embeddings"""

train_sample = train.sample(n=2000)
train_spark =spark.createDataFrame(train_sample)
train_nx_reviews = train_spark.join(reviews,['asin'],"inner")
train_nx_reviews_df = train_nx_reviews.toPandas()
train_nx_reviews_full = train_nx_reviews_df.drop(['reviewText','summary'],axis=1)
train_nx_reviews_combined = train_nx_reviews_full.groupby(core_columns).agg({'summary_review':lambda x: ' ### '.join(x)}).reset_index()
num_books_ve = train_nx_reviews_combined.shape[0]
train_nx_reviews_combined.head()

question_loader = DataFrameLoader(train_nx_reviews_combined, page_content_column="summary_review")
question_data = question_loader.load()
max_length = 500
splitter = RecursiveCharacterTextSplitter(
                                          chunk_size=max_length,
                                          chunk_overlap=400)
texts = splitter.split_documents(question_data)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(texts, embeddings)
db.save_local(f"train_{num_books_ve}_books_summary_review_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'})

sr_db = FAISS.load_local(f"review vector db/train_269_books_summary_review_db", embeddings)

"""## KMeans: combine and vectorize feature columns"""

train['combined'] = train[['title', 'categories']].agg(' '.join, axis=1)
# Build the tfidf matrix with the review texts
start_time = time.time()
text_content = train['combined']
vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                             min_df=1,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
tfidf = vector.fit_transform(text_content)

"""# Model Training and Generate Predictions

*No training required for Jaccard or FAISS

## Jaccard
"""

def find_recommendation(simgraph,query_id,thres,top_n):

  ego = nx.ego_graph(simgraph, query_id, radius = 1)
  Purchased_ASIN_Ego_Graph = nx.Graph(ego)

  Purchased_ASIN_Ego_Trim_Graph = nx.Graph()
  for f, t, e in Purchased_ASIN_Ego_Graph.edges(data = True):
      if e['weight'] >= thres:
          Purchased_ASIN_Ego_Trim_Graph.add_edge(f, t)

  Purchased_ASIN_Neighbors = Purchased_ASIN_Ego_Trim_Graph.neighbors(Purchased_ASIN)

  weights = [1 if Purchased_ASIN_Ego_Trim_Graph[u][v] == {} else Purchased_ASIN_Ego_Trim_Graph[u][v]['weight'] for u,v in Purchased_ASIN_Ego_Trim_Graph.edges()]

  pos = nx.spring_layout(Purchased_ASIN_Ego_Trim_Graph)
  nx.draw(Purchased_ASIN_Ego_Trim_Graph, pos, node_color = "lavender",  width=weights,node_size = 800, with_labels = True)

  options = {"node_size": 1200, "node_color": "r"}
  nx.draw_networkx_nodes(Purchased_ASIN_Ego_Trim_Graph, pos, nodelist=[query_id], **options)

  #nx.draw_networkx_nodes(Purchased_ASIN_Ego_Trim_Graph)

  ASIN_Meta = [[asin,
                books_dict[asin]['title'],
                books_dict[asin]['salesrank'],
                books_dict[asin]['tot_reviews'],
                books_dict[asin]['avg_rating'],
                books_dict[asin]['degree_centrality'],
                books_dict[asin]['clustering_coeff']
                ] for asin in Purchased_ASIN_Neighbors]

  res = sorted(ASIN_Meta, key = lambda x: (x[4], x[3]), reverse = True)[:top_n]

  print(f"Top 5 recommendations by average rating & total reviews from users who have purchased '{books_dict[Purchased_ASIN]['title']}' ")

  output = pd.DataFrame(res,columns = ['asin','title',
       'salesrank','tot_reviews','avg_rating', 'degree_centrality',
       'clustering_coeff'])
  display(output)
  return output

print("Looking for Recommendations for Customer Purchasing this Book: ")
print("---------------------------------------------------------------")
Purchased_ASIN = '0395485908'
print("ASIN = ", Purchased_ASIN)
print("title = ", books_dict[Purchased_ASIN]['title'])
print("salesrank = ", books_dict[Purchased_ASIN]['salesrank'])
print("tot_reviews = ", books_dict[Purchased_ASIN]['tot_reviews'])
print("avg_rating = ", books_dict[Purchased_ASIN]['avg_rating'])
print("degree_centrality = ", books_dict[Purchased_ASIN]['degree_centrality'])
print("clustering_coeff = ", books_dict[Purchased_ASIN]['clustering_coeff'])
print()

Copurchase_Graph = nx.read_weighted_edgelist("dataset_for_modeling/amazon-books-copurchase.edgelist")
res1 = find_recommendation(Copurchase_Graph,query_id = Purchased_ASIN, thres = 0.5,top_n = 5)

"""## FAISS"""

docs = sr_db.similarity_search("cooking with an airfryer")
for d in docs:
  print(d.metadata['title'])

"""## DeepWalk"""

## Create your node embeddings ##
# Instantiate the embedder
embedder = Word2Vec(window = 4, sg=1, negative=10, alpha=0.03, min_alpha=0.0001, seed=42)
# Build the vocab
embedder.build_vocab(walk_paths_weighted, progress_per=2)
# Train the embedder to build the word embeddings
embedder.train(walk_paths_weighted, total_examples=embedder.corpus_count, epochs=20, report_delay=1)

embedder = Word2Vec.load("word2vec/books_word2vec.model")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 100), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=arrays.shape[0]).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(10)


    # plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    # plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)

    plt.title('t-SNE visualization for {}'.format(word.title()))

query = "Patterns of Preaching: A Sermon Sampler"

# show original copurchased items
query_conns = title_conn.query(f'title == "{query}"')['connections'].values[0].split()
[asin_title.get(i) for i in query_conns]

x_ = embedder.wv.most_similar(query, topn=20)

#return similar titles from random walk
tsnescatterplot(embedder, query, [i[0] for i in x_])

bq = title_conn['title'].sample(1).values[0]
print(bq,'\n')
try:
  res = embedder.wv.most_similar(negative=[bq], topn=20)

  for r in res:
    print('\t - ',r[0],': ',embedder.wv.similarity(bq, r[0]))
    print()
except Exception as e:
    print(repr(e))

bq = title_conn['title'].sample(1).values[0]
try:
  tsnescatterplot(embedder, bq, [i[0] for i in embedder.wv.most_similar(negative=[bq], topn=20)][10:])
except Exception as e:
    print(repr(e))

"""## KMeans"""

# Clustering  Kmeans
k = 1000
kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]


terms = vector.get_feature_names_out()

# print the centers of the clusters
# for i in range(0,k):
#     word_list=[]
#     print("cluster%d:"% i)
#     for j in centers[i,:10]:
#         word_list.append(terms[j])
#     print(word_list)

request_transform = vector.transform(train['combined'])
# new column cluster based on the description
train['cluster'] = kmeans.predict(request_transform)

train['cluster'].value_counts().head()

# Find similar : get the top_n book with title + categories similar to the target title + categories
def find_similar(books, asin, top_n = 5):
  book_cluster = books.loc[books['asin'] == asin]['cluster'].iloc[0]
  book_connections = books.loc[books['asin'] == asin]['connections'].iloc[0]

  books_same_cluster = books.loc[books['cluster'] == book_cluster]
  books_same_cluster = books_same_cluster.sort_values(by=['avg_rating', 'tot_reviews'], ascending=[False, False])
  books_same_cluster_lst = books_same_cluster['asin'].tolist()
  book_connections_lst = book_connections.split(' ')

  similar_asin = []

  # check if the connection book and the given book are in the same cluster
  for bk in book_connections_lst:
    if len(similar_asin) < top_n:
      if bk in books_same_cluster_lst and bk != asin:
        similar_asin.append(bk)
        books_same_cluster_lst.remove(bk)

  if len(similar_asin) < top_n:
    for bk in books_same_cluster_lst:
      if len(similar_asin) < top_n and bk != asin:
        similar_asin.append(bk)

  return similar_asin

print("Top 5 Recommendations By connections, Then By AvgRating, and Then By TotalReviews for Users Purchased The Book: ")
print("--------------------------------------------------------------------------------------")

similarities = find_similar(train, '0827229534', 5)

top5_similar_book = train[train['asin'].isin(similarities)]
top5_similar_book

"""# Evaluation"""

kmeans = pd.read_csv("dataset_for_modeling/books_kmeans_cluster.csv")
train_kmeans = train.merge(kmeans[['asin','cluster']], left_on="asin",right_on="asin")
df = train_kmeans[['asin','title','categories','connections','cluster']].copy()
df['categories_split'] = df['categories'].apply(lambda x:x.split())

embedder_w2v = Word2Vec.load("word2vec/books_word2vec.model")

def common_items(list1, list2, list3):
  set1 = set(list1)
  set2 = set(list2)
  set3 = set(list3)
  common_in_at_least_two = set1.intersection(set2) | set1.intersection(set3) | set2.intersection(set3)
  return list(common_in_at_least_two)

asin_title = dict(zip(df.asin, df.title))
title_asin = dict(zip(df.title, df.asin))

rand_sample = train.sample(n=2000).reset_index().drop('index',axis=1).drop(['Id','group','salesrank', 'tot_reviews', 'avg_rating', 'clustering_coeff','DegreeCentrality'],axis=1)
rand_sample_individual_eval = rand_sample.copy()

for i,r in rand_sample.iterrows():
  title = r['title']
  try:
    # get ground truths
    query_conns = df.query(f'title == "{title}"')['connections'].values[0].split()
    ground_truths = set([asin_title.get(i) for i in query_conns])

    #generate predictions
    jaccard_results = find_recommendation(books_dict,Copurchase_Graph,r['asin'],thres=0.5,top_n=10)['title'].tolist()
    kmeans_results = df[df['asin'].isin(find_similar(df,r['asin'], 10))]['title'].tolist()
    deepwalk_results = pd.DataFrame(embedder_w2v.wv.most_similar(r['title'],topn=10),columns=['title','similarity_score'])['title'].tolist()
    ensemble_titles = set(common_items(jaccard_results,kmeans_results,deepwalk_results))
    ensemble_asin = ' '.join([title_asin.get(i) for i in list(ensemble_titles)])

    #calculate Jaccard score itself
    n1In2 = ground_truths & set(jaccard_results)
    n1Un2 = ground_truths| set(jaccard_results)
    if(len(n1In2)) > 0:
      sim = round(len(n1In2) / len(n1Un2), 2)
      #rand_sample_individual_eval.loc[rand_sample_eval['asin'] == r['asin'],'ensemble_recs']= ensemble_asin
      rand_sample_individual_eval.loc[rand_sample_individual_eval['asin'] == r['asin'],'jaccard_sim_score']= sim


    #calculate kmeans Jaccard score
    n1In2 = ground_truths & set(kmeans_results)
    n1Un2 = ground_truths| set(kmeans_results)
    if(len(n1In2)) > 0:
      sim = round(len(n1In2) / len(n1Un2), 2)
      #rand_sample_individual_eval.loc[rand_sample_eval['asin'] == r['asin'],'ensemble_recs']= ensemble_asin
      rand_sample_individual_eval.loc[rand_sample_individual_eval['asin'] == r['asin'],'kmeans_sim_score']= sim


    #calculate deepwalk Jaccard score
    n1In2 = ground_truths & set(deepwalk_results)
    n1Un2 = ground_truths| set(deepwalk_results)
    if(len(n1In2)) > 0:
      sim = round(len(n1In2) / len(n1Un2), 2)
      #rand_sample_individual_eval.loc[rand_sample_eval['asin'] == r['asin'],'ensemble_recs']= ensemble_asin
      rand_sample_individual_eval.loc[rand_sample_individual_eval['asin'] == r['asin'],'deepwalk_sim_score']= sim

    #calculate ensembled Jaccard score
    n1In2 = ground_truths & ensemble_titles
    n1Un2 = ground_truths| ensemble_titles
    if(len(n1In2)) > 0:
      sim = round(len(n1In2) / len(n1Un2), 2)
      rand_sample_individual_eval.loc[rand_sample_individual_eval['asin'] == r['asin'],'ensembled_sim_score']= sim
      rand_sample_individual_eval.loc[rand_sample_individual_eval['asin'] == r['asin'],'ensemble_recs']= ensemble_asin

  except Exception as e:
    pass

rand_sample_individual_eval = rand_sample_individual_eval.dropna().drop(['asin','categories'],axis=1)
rand_sample_individual_eval

import seaborn as sns
x = rand_sample_individual_eval["jaccard_sim_score"].tolist()
y = rand_sample_individual_eval["kmeans_sim_score"].tolist()
z = rand_sample_individual_eval["deepwalk_sim_score"].tolist()
all = rand_sample_individual_eval["ensembled_sim_score"].tolist()
ggg = [x,y,z,all]
fig, ax = plt.subplots(figsize=(7,7))
for a in ggg:
    sns.histplot(a, ax=ax, kde=True)
fig.legend(labels=["jaccard_sim_score","kmeans_sim_score","deepwalk_sim_score","ensembled_sim_score"],bbox_to_anchor=(1, 0.5))
plt.title("Distribution of Jaccard Evaluation Metric")