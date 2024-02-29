import pandas as pd

games_df = pd.read_csv("D:/N4HK2/HTGY/GameRCM/datasets-20240125T131440Z-001/datasets/Games_dataset.csv", index_col=0)

print('Number of games loaded: %s ' % (len(games_df)), '\n')

# Display the data
games_df.head()


import nltk
nltk.download('punkt')
import re
from nltk.stem.snowball import SnowballStemmer

# Create an English language SnowballStemmer object
stemmer = SnowballStemmer("english")

# Define a function to perform both stemming and tokenization
def tokenize_and_stem(text):
    
    # Tokenize by sentence, then by word - thực hiện tokenization (spliting token)
    tokens = [word for sent in nltk.sent_tokenize(text) 
              for word in nltk.word_tokenize(sent)]
    
    # Filter out raw tokens to remove noise - Chuẩn hóa các token (token normalization)
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stems = [stemmer.stem(word) for word in filtered_tokens]
    
    return stems

# kỹ thuật extract features từ input text
# create input features to train NLP models
# Transform token into features
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer object with stopwords and tokenizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))

# Fit and transform the tfidf_vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in games_df["Plots"]])
# ==================================KMeans==================================================================
from sklearn.cluster import KMeans

km = KMeans(n_clusters=7)

# Fit the k-means object with tfidf_matrix
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

games_df["cluster"] = clusters
games_df['cluster'].value_counts() 

# =================================Hierarchy===================================================================

# Import matplotlib.pyplot for plotting graphs
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Create mergings matrix 
mergings = linkage(similarity_distance, method='complete')

# ============================Plot the dendrogram, using title as label column=======================================
dendrogram_ = dendrogram(mergings,
               labels=[x for x in games_df["Title"]],
               leaf_rotation=90,
               leaf_font_size=16,
)

# Adjust the plot
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

plt.savefig('dendo.png', dpi=100)
plt.show()

# ===========================================Create a dataframe from the similarity matrix to export========================
vals = games_df.Title.tolist()
similarity_df = pd.DataFrame(similarity_distance, columns=vals, index=vals)
# Export
similarity_df.to_csv('datasets/sim_matrix.csv')


# ========================Recommendation example===========================================
title = "Payday 2"

matches = similarity_df[title].sort_values()[1:6]
matches = matches.index.tolist()
games_df.set_index('Title').loc[matches]
