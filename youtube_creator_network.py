import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import numpy as np
import json
from networkx.algorithms import community
import os

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

exclude_categories = [1, 10] 
#1: film and animation
#10: music

#---------------------DATA PREPROCESSING--------------------
df = pd.read_csv("CAvideos.csv")

#sorting data
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df = df.sort_values('trending_date') #keep more recent videos
df = df.drop_duplicates(subset='video_id', keep='first') #remove all video duplicates
df = df.sort_values('views', ascending=False) #sort by views
df = df[~df['category_id'].isin(exclude_categories)] #exclude videos in exclude_categories

exclude_titles = ["reupload", "highlights"]

# channels that are legitimate EVENT SOURCES but not creators
# keep as SOURCE only, never as TARGET
event_source_channels = [
    # news organizations
    "ABC News", "CBC News", "BBC", "CNN", "MSNBC", "NDTV",
    "FOX 10 Phoenix", "FOX 11 Los Angeles", "SKY NEWS",
    "Channel 4 News", "Guardian News", "Washington Post",
    "Daily Mail", "USA TODAY", "VOA News", "TODAY",
    # sports organizations  
    "WWE", "NBA", "NHL", "Olympic", "NBA on TNT",
    # film studios
    "Universal Pictures", "Warner Bros. Pictures",
    "Marvel Entertainment", "Star Wars", "EA Star Wars",
    # tv show official channels
    "American Idol", "The View", "Entertainment Tonight",
    "Jimmy Kimmel Live", "The Tonight Show Starring Jimmy Fallon",
    "The Late Show with Stephen Colbert",
]

# channels that are legitimate TARGETS only
# they react to creators, showing influence received
react_channels = [
    "FBE", "REACT", "Reaction Time", "Michael McCrudden", "ZIAS!", "RDCworld1",
]

# channels that are aggregators of content from other creators, not original sources
aggregator_channels = [
    "WatchMojo.com", "MsMojo", "TheTalko", "Nicki Swift", "Brand Sins", "What's Trending", 
    "DramaAlert", "Scarce", "Top Clips YT", "9 Media", "AB Network", "Chat Sports", 
    "FreeDawkins", "Motion Station", "TMZ", "DramaAlert", "Scarce", "TheTalko", 
    "Nicki Swift", "MsMojo", "Brand Sins", "Motion Station", "FilmSelect Trailer",
    "Wrestling Time", "VA10HD", "Ximo Pierto", "RonFilms7", "Premium Sintesi", "AB Network", 
    "WatchMojo.com", "9 Media", "NTA Channel", "FootballHD", "FreeDawkins", "Real Sportz",
    "Plug Trending", "GOLAZO TV", "Legendary Football", "Meghan Markle & Prince Harry News", "Royal Insider",
    "RWANDA5 TV"
]

# channels to exclude entirely — neither source nor target
exclude_channels = [
    #re-upload channels
    "TVT Network", "manna7654", "Doug Ketchum", "Elijah Price", "Dennis Boyd", "Derek Guyton", 
    "ThingsToKnow", "Hawkz80", "Any Bill Maher Fans", "political-news", "Bucking Trump",
    "Herry Saputra", "info boxxx", "George E. Creed", "HeeYuHee [DO NOT RE-UPLOAD & SECONDARY PROCESSING]",
    "TOP SHOWS", "Best Christmas Movies 2018", "Dans nos Laurentides.com", "Country Christmas Music",
    "Merry Christmas", "FanBillMahar", "Dans nos Laurentides.com", "Christmas Time", 
    "Resumenes FUT", "highlights", "Akashi Tv", "hallodoremi", "Chat Sports", "MrMatador", 
    "Last Week Tonight Season 4", "Neplanet Football",

    # spam/low quality
    "D", "6.8M Views", "New Channel", "black bull", "ANT8061", "bongizzlez", "valtairvasconcelos1",
    "gero97440", "mikhailsur123", "faisal8912", "sirens999sirens", "AntAhprernA", "Ahacrh Channel",
    "OlipBeautyCorner", "AresPromo", "NugoBasilaiaa"
]


#remove videos with certain words in the title and from certain channels to avoid noise
title_mask = ~df['title'].str.contains('|'.join(exclude_titles), case=False, na=False)
channel_pattern_mask = ~df['channel_title'].str.contains('highlights|recap|full match|full game|re-upload|reupload', case=False, na=False)
channel_mask = ~df['channel_title'].isin(exclude_channels)
df = df[title_mask & channel_mask & channel_pattern_mask].reset_index(drop=True)

#only take english titles
df = df[df['title'].apply(is_mostly_english)].reset_index(drop=True)

# remove videos with very short or vague titles
df = df[df['title'].str.split().str.len() >= 4].reset_index(drop=True)

#keep only relevant columns for later use and save to new csv file
df = df[['video_id','channel_title','title','tags','category_id','trending_date']]
df.to_csv("filtered_youtube_trending.csv")

#removes stop words (the, an, etc), combines title and calculates the similarity matrix
combined_stopwords = list(ENGLISH_STOP_WORDS) + french_stopwords
vectorizer = TfidfVectorizer(stop_words=combined_stopwords)
tfidf_matrix = vectorizer.fit_transform(df["title"])
similarity_matrix = cosine_similarity(tfidf_matrix)

#extract dates, categories, and channels as numpy arrays
dates = df['trending_date'].values.astype('datetime64[D]')
categories = df['category_id'].values
channels = df['channel_title'].values

#get upper triangle indices to avoid duplicate pairs and self-similarity
rows, cols = np.triu_indices(len(df), k=1)

#calculate similarities, time differences, and channel differences for all pairs
sims       = similarity_matrix[rows, cols]
time_diffs = (dates[cols] - dates[rows]).astype(int)
diff_chan  = channels[rows] != channels[cols] #only connect different channels to avoid self-loops

#apply thresholds: similarity > 0.5 && similarity < 1.0, time difference between 2 and 10 days, and different channels
mask = (sims > 0.50) & (sims < 1.0) & diff_chan & (time_diffs > 2) & (time_diffs <= 10)

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
    

    # skip if target is an event source channel
    # (news/studio/sports org should never be a target)
    if creator_b in event_source_channels:
        continue

    # skip if source is a react channel
    # (react channels should never initiate influence)
    if creator_a in react_channels:
        continue

    # skip if source is an aggregator channel    
    # # (aggregators should never initiate influence)
    if creator_a in aggregator_channels:
        continue  

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
            "source_category_id": int(categories[i]),
            "target_category_id": int(categories[j]),
            "source_title": df.iloc[i]['title'],
            "target_title": df.iloc[j]['title'],
            "source_date": str(df.iloc[i]['trending_date'].date()),
            "target_date": str(df.iloc[j]['trending_date'].date())
        })

# Save to JSON
output = {"nodes": nodes_data, "edges": edges_data}
with open("youtube_network.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {len(nodes_data)} nodes and {len(edges_data)} edges to youtube_network.json")

#---------------------NETWORK VISUALIZATION--------------------
category_colors = {
    2: "#17159b",  # autos & vehicles
    17: "#ff56da",  # pets & animals
    19: "#2bd1b0",  # travel & events 
    20: "#0e4d08",  # gaming
    22: '#3498db',  # people and blogs
    23: "#F7F32D",  # comedy
    24: "#cff529",  # entertainment
    25: '#8e44ad',  # News & Politics
    26: "#80f751",  # 	Howto & Style
    27: "#e66510",  # education
    28: "#f82424",  # science & technology
    29: "#E6E6E6"   # nonprofits & Activism
}

categories_to_plot = {
    2: "autos and vehicles",
    17: "Sports",
    19: "travel and events",
    20: "Gaming",
    22: "people and blogs",
    23: "comedy",
    24: "entertainment",
    25: "News & Politics",
    26: "howto and style",
    27: "Education",
    28: "science and technology",
    29: "Nonprofits & Activism"
}

#create folders
os.makedirs("graphs/communities", exist_ok=True)
os.makedirs("graphs/categories", exist_ok=True)

#building visualization with pyvis
net = Network(height="950px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")

## CORE NETWORK
for node in G.nodes():
    cat = category_map.get(node, 0)
    color = category_colors.get(cat, '#cccccc')
    tooltip = f"{node}\nCategory: {cat}\n"
    net.add_node(node, label=node, color=color, title=tooltip)

# build edge tooltip map
edge_title_map = {}
for e in edges_data:
    key = (e['source'], e['target'])
    edge_title_map[key] = (
        f"Similarity: {e['similarity']}\n"
        f"Source: {e['source_title']}\n"
        f"Target: {e['target_title']}\n"
        f"Date: {e['source_date']} → {e['target_date']}"
    )

for src, dst in G.edges():
    edge_tooltip = edge_title_map.get((src, dst), "")
    net.add_edge(src, dst, color="rgba(255,255,255,0.3)", arrows="to", title=edge_tooltip)

net.show_buttons(filter_=['physics'])
net.save_graph("graphs/youtube_network.html")
print("Saved to youtube_network.html")

## COMMUNITY SUBGRAPHS
#get community subgraphs and visualize separately
print(f"\n=== Louvain Communities ===")
G_undirected = G.to_undirected()
communities = community.louvain_communities(G_undirected)
print(f"Number of communities: {len(communities)}")
for i, comm in enumerate(communities):
    # skip communities that are too small
    if len(comm) < 3:
        continue
    
    comm_subgraph = G.subgraph(comm).copy()

    print(f"\nCommunity {i}: {list(comm)}")
    print(f"Nodes: {comm_subgraph.number_of_nodes()}, Edges: {comm_subgraph.number_of_edges()}")
    
    comm_net = Network(height="950px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
    
    for node in comm_subgraph.nodes():
        cat = category_map.get(node, 0)
        color = category_colors.get(cat, '#cccccc')
        comm_net.add_node(node, label=node, color=color, title=f"{node}\nCategory: {cat}")
    
    for src, dst in comm_subgraph.edges():
        edge_tooltip = edge_title_map.get((src, dst), "")
        comm_net.add_edge(src, dst, color="rgba(255,255,255,0.3)", arrows="to", title=edge_tooltip)
    
    comm_net.show_buttons(filter_=['physics'])
    filename = f"graphs/communities/community_{i}.html"
    comm_net.save_graph(filename)
    print(f"Saved to {filename}")

## CATEGORY SUBGRAPHS
#plot per category graph
for cat_id, cat_name in categories_to_plot.items():
    cat_nodes = [node for node in G.nodes() if category_map.get(node) == cat_id]
    subgraph = G.subgraph(cat_nodes)

    if subgraph.number_of_nodes() < 3:
        print(f"\nNo nodes for category {cat_name}")
        continue

    print(f"\n=== {cat_name} (Category {cat_id}) ===")
    print(f"Nodes: {subgraph.number_of_nodes()}, Edges: {subgraph.number_of_edges()}")

    sub_net = Network(height="950px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")  # renamed to sub_net
    color = category_colors.get(cat_id, "#cccccc")

    for node in subgraph.nodes():
        sub_net.add_node(node, label=node, color=color, title=node)
    for src, dst in subgraph.edges():
        edge_tooltip = edge_title_map.get((src, dst), "")
        sub_net.add_edge(src, dst, color="rgba(255,255,255,0.3)", arrows="to", title=edge_tooltip)

    sub_net.show_buttons(filter_=['physics'])
    filename = f"graphs/categories/network_cat_{cat_id}.html"
    sub_net.save_graph(filename)
    print(f"Saved to {filename}")