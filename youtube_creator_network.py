import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import numpy as np

def is_mostly_english(text):
    if len(text) == 0:
        return False
    return sum(c.isascii() for c in text) / len(text) > 0.9

french_stopwords = [
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "ce", "qui", "que", "quoi", "ou", "où", "est", "sur", "avec"
]

valid_categories = [21, 22, 23, 26] #video blogging, people and blogs, comedy, howto and style
#removed 20 is gaming
#removed enterntainment 24
#removed science and technology 28
df = pd.read_csv("CAvideos.csv")

#sorting data
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df = df.sort_values('trending_date') #keep more recent videos
df = df.drop_duplicates(subset='video_id', keep='first') #remove all video duplicates
df = df.sort_values('views', ascending=False) #sort by views
df = df[df['category_id'].isin(valid_categories)] #keep videos only in categories valid_categories

exclude_titles = ["trailer", "teaser", "music video", "series", "getting over it", "galaxy s9", "episode", "épisode", "youtube rewind"]
exclude_channels = ["Youtube Spotlight", "KSI", "Philip DeFranco", "Samsung", "HUM TV", "ARY Digital"]
title_mask = ~df['title'].str.contains('|'.join(exclude_titles), case=False, na=False)
channel_mask = ~df['channel_title'].str.contains('|'.join(exclude_channels), case=False, na=False) #remove certain channels
#only take english titles
df["text"] = df["title"]
df = df[df["text"].apply(is_mostly_english)].reset_index(drop=True)
df = df.head(5000) #take 5000 most popular videos after filtering

print(len(df))

creators = df[['video_id','channel_title','title','tags','category_id','trending_date']]

df.to_csv("filtered_youtube_trending.csv") #write data to new csv file

#removes stop words (the, an, etc), combines title and tags and calculates the similarity matrix
combined_stopwords = list(ENGLISH_STOP_WORDS) + french_stopwords
vectorizer = TfidfVectorizer(stop_words=combined_stopwords)
tfidf_matrix = vectorizer.fit_transform(df["text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

print(similarity_matrix)

dates = df['trending_date'].values.astype('datetime64[D]')
categories = df['category_id'].values
channels = df['channel_title'].values

rows, cols = np.triu_indices(len(df), k=1)

sims       = similarity_matrix[rows, cols]
time_diffs = (dates[cols] - dates[rows]).astype(int)
same_cat   = categories[rows] == categories[cols]
diff_chan  = channels[rows] != channels[cols]

mask = (sims > 0.5) & same_cat & diff_chan & (time_diffs > 0) & (time_diffs <= 14)

#apply mask to get only valid pairs
valid_rows = rows[mask]
valid_cols = cols[mask]
valid_sims = sims[mask]

#build graph and print edges
G = nx.DiGraph()
for creator in df['channel_title'].unique():
    G.add_node(creator)

for i, j, sim in zip(valid_rows, valid_cols, valid_sims):
    creator_a, creator_b = channels[i], channels[j]
    print(f"{creator_a} -> {creator_b}")
    print(f"    {df.iloc[i]['title']}  -->  {df.iloc[j]['title']}")
    print(f"    similarity = {sim:.2f}")
    print(f"    category = {categories[i]}\n")
    G.add_edge(creator_a, creator_b)

G.remove_nodes_from(list(nx.isolates(G)))

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

category_colors = {
    21: '#e74c3c',  # video blogging
    22: '#3498db',  # entertainment
    23: '#2ecc71',  # people & blogs
    26: '#f39c12',  # comedy
}

category_map = dict(zip(df['channel_title'], df['category_id']))

net = Network(height="950px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")

for node in G.nodes():
    cat = category_map.get(node, 0)
    color = category_colors.get(cat, '#cccccc')
    
    net.add_node(node, label=node, color=color, title=f"{node}\nCategory: {cat}")

for src, dst in G.edges():
    net.add_edge(src, dst, color="rgba(255,255,255,0.3)", arrows="to")

net.show_buttons(filter_=['physics'])
net.save_graph("youtube_network.html")
print("Saved to youtube_network.html")