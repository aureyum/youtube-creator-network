import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import numpy as np
import json
from networkx.algorithms import community

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

valid_categories = [2, 15, 19, 21, 22, 23, 26, 27, 28, 29] 
#2: autos and vehicles
#15: pets and animals
#19: travel and events
#21: video blogging
#22: people and blogs
#23: comedy
#24: entertainment (REMOVED)
#26: howto and style
#27: news and politics
#28: science and technology
#29: Nonprofits & Activism

#---------------------DATA PREPROCESSING--------------------
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

#df = df.head(5000)  # Limit to top 5000 videos
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

#--------------------GRAPH CONSTRUCTION--------------------
#build graph
G = nx.DiGraph()
for creator in df['channel_title'].unique():
    G.add_node(creator)

for i, j, sim in zip(valid_rows, valid_cols, valid_sims):
    creator_a, creator_b = channels[i], channels[j]
    G.add_edge(creator_a, creator_b)

G.remove_nodes_from(list(nx.isolates(G))) #remove isolated nodes

category_map = dict(zip(df['channel_title'], df['category_id']))

#-------------------INFLUENCE ANALYSIS--------------------
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

# betweenness centrality
betweenness = nx.betweenness_centrality(G)

print("\n=== Betweenness Centrality (Bridge Creators) ===")
for creator, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
    if score > 0:
        print(f"{creator}: {score:.4f}")

print("\n=== Louvain Communities ===")
G_undirected = G.to_undirected()
communities = community.louvain_communities(G_undirected)
print(f"Number of communities: {len(communities)}")
for i, comm in enumerate(communities):
    print(f"Community {i}: {list(comm)}")

# density - how connected is the network relative to its maximum possible connections
print("\nDensity:", nx.density(G))

# average degree
avg_in = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
avg_out = sum(dict(G.out_degree()).values()) / G.number_of_nodes()
print(f"\nAverage in-degree: {avg_in:.2f}")
print(f"\nAverage out-degree: {avg_out:.2f}")

# weakly connected components - how many separate clusters exist
components = list(nx.weakly_connected_components(G))
print(f"\nNumber of weakly connected components: {len(components)}")
print(f"\nLargest component size: {max(len(c) for c in components)}")

#--------------------JSON OUTPUT--------------------
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

#---------------------NETWORK VISUALIZATION--------------------
category_colors = {
    2: "#17159b",  # autos & vehicles
    15: "#ff56da",  # pets & animals
    19: "#2bd1b0",  # travel & events
    21: "#0e4d08",  # video blogging
    22: '#3498db',  # people and blogs
    23: "#F7F32D",  # comedy
    24: '#8e44ad',  # entertainment
    26: "#80f751",  # 	Howto & Style
    27: "#e66510",  # news & politics
    28: "#f5131e",  # science & technology
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

categories_to_plot = {
    2: "autos and vehicles",
    15: "pets and animals",
    19: "travel and events",
    21: "video blogging",
    22: "people and blogs",
    23: "comedy",
    24: "entertainment",
    26: "howto and style",
    27: "news and politics",
    28: "science and technology",
    29: "Nonprofits & Activism"
}

for cat_id, cat_name in categories_to_plot.items():
    cat_nodes = [node for node in G.nodes() if category_map.get(node) == cat_id]
    subgraph = G.subgraph(cat_nodes)

    if subgraph.number_of_nodes() == 0:
        print(f"\nNo nodes for category {cat_name}")
        continue

    print(f"\n=== {cat_name} (Category {cat_id}) ===")
    print(f"Nodes: {subgraph.number_of_nodes()}, Edges: {subgraph.number_of_edges()}")

    sub_net = Network(height="950px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")  # renamed to sub_net
    color = category_colors.get(cat_id, "#cccccc")

    for node in subgraph.nodes():
        sub_net.add_node(node, label=node, color=color, title=node)
    for src, dst in subgraph.edges():
        sub_net.add_edge(src, dst, color="rgba(255,255,255,0.3)", arrows="to")

    sub_net.show_buttons(filter_=['physics'])
    filename = f"network_cat_{cat_id}.html"
    sub_net.save_graph(filename)
    print(f"Saved to {filename}")