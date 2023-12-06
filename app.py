from utils import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from gensim.models import Word2Vec
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def ref_dict():
    with open('dataset_for_modeling/books_dict.json', 'r') as openfile:
        books_dict = json.load(openfile)
    return books_dict

@st.cache_data
def load_data():
    train = pd.read_csv("dataset_for_modeling/books_amzn_meta.csv").rename(columns={'Unnamed: 0':"asin"}).dropna()
    kmeans = pd.read_csv("kmeans/books_kmeans_cluster.csv")
    train_kmeans = train.merge(kmeans[['asin','cluster']], left_on="asin",right_on="asin")
    df = train_kmeans[['asin','title','categories','connections','cluster']].copy()
    df['categories_split'] = df['categories'].apply(lambda x:x.split())
    return df

@st.cache_data
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {'device': 'cpu'})

    sr_db = FAISS.load_local(f"train_269_books_summary_review_db", embeddings)

    embedder_w2v = Word2Vec.load("word2vec/books_word2vec.model")

    return sr_db,embedder_w2v

@st.cache_data
def find_recommendation(books_dict,_simgraph,Purchased_ASIN,thres,top_n):

  ego = nx.ego_graph(_simgraph, Purchased_ASIN, radius = 1)
  Purchased_ASIN_Ego_Graph = nx.Graph(ego)

  Purchased_ASIN_Ego_Trim_Graph = nx.Graph()
  for f, t, e in Purchased_ASIN_Ego_Graph.edges(data = True):
      if e['weight'] >= thres:
          Purchased_ASIN_Ego_Trim_Graph.add_edge(f, t)

  Purchased_ASIN_Neighbors = Purchased_ASIN_Ego_Trim_Graph.neighbors(Purchased_ASIN)

  weights = [1 if Purchased_ASIN_Ego_Trim_Graph[u][v] == {} else Purchased_ASIN_Ego_Trim_Graph[u][v]['weight'] for u,v in Purchased_ASIN_Ego_Trim_Graph.edges()]

  pos = nx.spring_layout(Purchased_ASIN_Ego_Trim_Graph)
  options = {"node_size": 1200, "node_color": "r"}

  fig, ax = plt.subplots()
  nx.draw(Purchased_ASIN_Ego_Trim_Graph, pos, node_color = "lavender",  width=weights,node_size = 800, with_labels = True)
  nx.draw_networkx_nodes(Purchased_ASIN_Ego_Trim_Graph, pos, nodelist=[Purchased_ASIN], **options)

  ASIN_Meta = [[asin,
                books_dict[asin]['title'],
                books_dict[asin]['salesrank'],
                books_dict[asin]['tot_reviews'],
                books_dict[asin]['avg_rating'],
                books_dict[asin]['degree_centrality'],
                books_dict[asin]['clustering_coeff']
                ] for asin in Purchased_ASIN_Neighbors]

  res = sorted(ASIN_Meta, key = lambda x: (x[4], x[3]), reverse = True)[:top_n]

  #print(f"Top 5 recommendations by average rating & total reviews from users who have purchased '{books_dict[Purchased_ASIN]['title']}' ")

  return (pd.DataFrame(res,columns = ['asin','title',
       'salesrank','tot_reviews','avg_rating', 'degree_centrality',
       'clustering_coeff']), fig)



def main():
    st.sidebar.header("DATA 245: Machine Learning")
    st.sidebar.divider()
    st.title("Amazon Products: Books Discovery")

    tab1, tab2, tab3, tab4 = st.tabs(["Set Theory","KMeans","DeepWalk","Ensemble"])


    description_text = st.sidebar.text_area('Please describe what you would like to read and we will provide an initial list of recommendations from our collection!')
    st.sidebar.divider()
    
    df = load_data()
    asin_title = dict(zip(df.asin, df.title))
    title_asin = dict(zip(df.title, df.asin))

    sr_db, embedder_w2v = load_embeddings()
    docs = sr_db.similarity_search(description_text)


    st.sidebar.caption("Here are the best matched results from your description using FAISS:")
    for d in docs:
        st.sidebar.markdown("- " + d.metadata['title'])

    query = st.sidebar.selectbox(
                'Select book title for recommendations:',
                [d.metadata['title'] for d in docs],None)


    st.sidebar.divider()



    if st.sidebar.button('Generate Predictions'):
        with st.spinner("Recommendations ongoing"):
            # show original copurchased items
            query_conns = df.query(f'title == "{query}"')['connections'].values[0].split()


            with tab1:

                st.subheader("Set Theory:")
                books_dict = ref_dict()
                Copurchase_Graph = nx.read_weighted_edgelist("dataset_for_modeling/amazon-books-copurchase.edgelist")

                Purchased_ASIN = title_asin.get(query)

                col1, col2 = st.columns(2)

                try:
                    with col1:
                        st.caption("Looking for Recommendations for Customer Purchasing this Book: ")
                        st.caption("ASIN = " + Purchased_ASIN)
                        st.caption("title = " + books_dict[Purchased_ASIN]['title'])
                        st.caption("salesrank = " + str(books_dict[Purchased_ASIN]['salesrank']))
                        st.caption("tot_reviews = " + str(books_dict[Purchased_ASIN]['tot_reviews']))
                        st.caption("avg_rating = " + str(books_dict[Purchased_ASIN]['avg_rating']))
                        st.caption("degree_centrality = " + str(books_dict[Purchased_ASIN]['degree_centrality']))
                        st.caption("clustering_coeff = " + str(books_dict[Purchased_ASIN]['clustering_coeff']))
                        set_res,fig_nx = find_recommendation(books_dict,Copurchase_Graph,Purchased_ASIN,thres=0.5,top_n=10)
                        st.dataframe(set_res)

                    with col2:
                        st.pyplot(fig_nx)
                except Exception as e: print(e)

                
            with tab2:

                st.subheader("KMeans:")
                similarities = find_similar(df, title_asin.get(query), 10)
                top5_similar_book = df[df['asin'].isin(similarities)]
                st.dataframe(top5_similar_book[['title','categories_split']])

            with tab3:

                st.subheader("DeepWalk:")
                most_sim = embedder_w2v.wv.most_similar(query,topn=10)
                most_sim_df = pd.DataFrame(most_sim,columns=['title','similarity_score'])
                col1,col2 = st.columns(2)

                with col1:
                    st.dataframe(most_sim)

                with col2:
                    fig = tsnescatterplot(embedder_w2v, query, [i[0] for i in most_sim])
                    st.pyplot(fig)

            with tab4:
                st.subheader("Final Recommendation")

                try:
                    res1 = set_res['title'].tolist()
                except:
                    pass

                res2 = top5_similar_book['title'].tolist()
                res3 = most_sim_df['title'].tolist()

                try:
                    set_theory_kmeans = list(set(res1) & set(res2))
                    if len(list(set(res1) & set(res2))) >= 1:
                        st.text("Between set theory and kmeans")
                        for i in set_theory_kmeans:
                            st.markdown("- " + i)
                except:
                    pass
                
                kmeans_deepwalk = list(set(res2) & set(res3))
                if len(kmeans_deepwalk) >= 1:
                    st.text("Between kmeans and deepwalk")
                    for i in kmeans_deepwalk:
                            st.markdown("- " + i)

                try:
                    set_theory_deepwalk = list(set(res1) & set(res3))
                    if len(set_theory_deepwalk) >= 1:
                        st.text("Between set theory and deepwalk")
                        for i in set_theory_deepwalk:
                            st.markdown("- " + i)
                except:
                    pass


                st.divider()

                st.caption(f"Below are books that have purchased alongside {query}")
                for i in query_conns:
                    st.markdown("- " + asin_title.get(i))






if __name__ == "__main__":
    main()

