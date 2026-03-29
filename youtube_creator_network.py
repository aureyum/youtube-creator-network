import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import numpy as np
import json

#function to check if a text is mostly English (at least 90% ASCII characters)
def is_mostly_english(text):
    if len(text) == 0:
        return False
    return sum(c.isascii() for c in text) / len(text) > 0.9

#list of common French stop words to remove from titles
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

exclude_titles = ["trailer", "teaser", "music video", "series", "getting over it", "galaxy s9", "episode", "épisode", "deleted video"]
exclude_channels = ["Youtube Spotlight", "Samsung", "HUM TV", "ARY Digital", "Country Christmas Music", "Merry Christmas",
                    # News re-upload channels
                    "Elijah Price", "TVT Network", "manna7654", "Doug Ketchum",
                    "Dennis Boyd", "Derek Guyton", "Fox News Live Stream", "ThingsToKnow",
                    "artis storm", "Any Bill Maher Fans", "Hawkz80", "political-news",
                    "CaxiasNET", "Real Time Season 15", "Bucking Trump", "Herry Saputra",
                    "info boxxx",
                    # Sports
                    "SPS Proudly Present", "Neplanet Football"]

#remove videos with certain words in the title and from certain channels to avoid noise
title_mask = ~df['title'].str.contains('|'.join(exclude_titles), case=False, na=False)
channel_mask = ~df['channel_title'].str.contains('|'.join(exclude_channels), case=False, na=False) #remove certain channels
df = df[title_mask & channel_mask].reset_index(drop=True)

#only take english titles
df = df[df["title"].apply(is_mostly_english)].reset_index(drop=True)

print(len(df))

#keep only relevant columns for later use and save to new csv file
df = df[['video_id','channel_title','title','tags','category_id','trending_date']]
df.to_csv("filtered_youtube_trending.csv")

#removes stop words (the, an, etc), combines title and calculates the similarity matrix
combined_stopwords = list(ENGLISH_STOP_WORDS) + french_stopwords
vectorizer = TfidfVectorizer(stop_words=combined_stopwords)
tfidf_matrix = vectorizer.fit_transform(df["title"])
similarity_matrix = cosine_similarity(tfidf_matrix)

#extract dates, categories, and channels as numpy arrays for faster access
dates = df['trending_date'].values.astype('datetime64[D]')
categories = df['category_id'].values
channels = df['channel_title'].values

#get upper triangle indices to avoid duplicate pairs and self-similarity
rows, cols = np.triu_indices(len(df), k=1)

#calculate similarities, time differences, and channel differences for all pairs
sims       = similarity_matrix[rows, cols]
time_diffs = (dates[cols] - dates[rows]).astype(int)
diff_chan  = channels[rows] != channels[cols] #only connect different channels to avoid self-loops

#apply thresholds: similarity > 0.5 && similarity < 1.0, time difference between 0 and 14 days, and different channels
mask = (sims > 0.5) & (sims < 1.0) & diff_chan & (time_diffs > 0) & (time_diffs <= 14)

#apply mask to get only valid pairs
valid_rows = rows[mask]
valid_cols = cols[mask]
valid_sims = sims[mask]

#build graph
G = nx.DiGraph()
for creator in df['channel_title'].unique():
    G.add_node(creator)

for i, j, sim in zip(valid_rows, valid_cols, valid_sims):
    creator_a, creator_b = channels[i], channels[j]
    G.add_edge(creator_a, creator_b)

G.remove_nodes_from(list(nx.isolates(G))) #remove isolated nodes

category_map = dict(zip(df['channel_title'], df['category_id']))

# Build nodes list
nodes_data = []
for node in G.nodes():
    cat = category_map.get(node, 0)
    nodes_data.append({
        "id": node,
        "category_id": int(cat),
        "in_degree": G.in_degree(node),
        "out_degree": G.out_degree(node)
    })

# Build edges list
edges_data = []
for i, j, sim in zip(valid_rows, valid_cols, valid_sims):
    creator_a, creator_b = channels[i], channels[j]
    if G.has_edge(creator_a, creator_b):  # only save edges that survived isolate removal
        edges_data.append({
            "source": creator_a,
            "target": creator_b,
            "similarity": round(float(sim), 2),
            "category_id": int(categories[i]),
            "source_title": df.iloc[i]['title'],
            "target_title": df.iloc[j]['title'],
            "source_date": str(df.iloc[i]['trending_date'].date()),
            "target_date": str(df.iloc[j]['trending_date'].date())
        })

# Save to JSON
output = {"edges": edges_data}
with open("youtube_network.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {len(nodes_data)} nodes and {len(edges_data)} edges to youtube_network.json")

# In-degree and out-degree for all nodes
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# Print sorted by out-degree (most influential first) top 10
print("=== Out-Degree (Trend Setters) ===")
for creator, degree in sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]:
    if degree > 0:
        print(f"{creator}: {degree}")

print("\n=== In-Degree (Trend Followers) ===")
for creator, degree in sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]:
    if degree > 0:
        print(f"{creator}: {degree}")

#betweenness centrality
betweenness = nx.betweenness_centrality(G)

# Print top 5 by betweenness
print("\n=== Betweenness Centrality (Bridge Creators) ===")
for creator, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{creator}: {score:.4f}")

category_colors = {
    21: '#e74c3c',  # video blogging
    22: '#3498db',  # entertainment
    23: '#2ecc71',  # people & blogs
    26: '#f39c12',  # comedy
}

#building visualization with pyvis
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