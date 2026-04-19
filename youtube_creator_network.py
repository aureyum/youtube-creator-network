import shutil

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import numpy as np
import json
from networkx.algorithms import community
import os

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

#list of common French stop words to remove from titles
french_stopwords = [
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "ce", "qui", "que", "quoi", "ou", "où", "est", "sur", "avec"
]

# Categories treated as event sources — can only be SOURCE nodes, never targets
event_source_categories = [1, 10, 17, 25] 
#1: film and animation
#10: music
#17: sports
#25: news and politics

#titles to be excluded, are mostly reupload contents
exclude_titles = ["reupload", "highlights", "lyrics"]

# channels that are legitimate EVENT SOURCES but not creators
# keep as SOURCE only, never as TARGET
event_source_channels = [
    # news organizations
    "ABC News", "CBC News", "BBC", "CNN", "MSNBC", "NDTV", "FOX 10 Phoenix", "FOX 11 Los Angeles", 
    "SKY NEWS", "Channel 4 News", "Guardian News", "Washington Post", "Daily Mail", "USA TODAY", 
    "VOA News", "TODAY",
    # sports organizations  
    "WWE", "NBA", "NHL", "Olympic", "NBA on TNT",
    # film studios
    "Universal Pictures", "Warner Bros. Pictures", "Marvel Entertainment", "Star Wars", 
    "EA Star Wars", "Marvel Studios Movies",
    # tv show official channels
    "American Idol", "The View", "Entertainment Tonight", "Jimmy Kimmel Live", "The Tonight Show Starring Jimmy Fallon",
    "The Late Show with Stephen Colbert",
    #entertainment companies
    "ibighit", "ChildishGambinoVEVO",
    #others
    "HeyUGuys", "The Verge"
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
    "RWANDA5 TV", "Furious Trailer"
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

# Color mapping for each YouTube category ID used in network visualization
category_colors = {
    1: "#17159b",   # film and animation
    2: "#f7146b",   # autos & vehicles
    10: "#8f4c16",  # music
    15: "#ec9191",  # pets and animals
    17: "#ff56da",  # sports
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

# Human-readable names for each category used in subgraph visualization
categories_to_plot = {
    1: "film and animation",
    2: "autos and vehicles",
    10: "music",
    15: "pets and animals",
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#function to check if a text is mostly English (at least 90% ASCII characters)
def is_mostly_english(text):
    if len(text) == 0:
        return False
    return sum(c.isascii() for c in text) / len(text) > 0.9

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data():
    """
    Loads the raw Canadian YouTube Trending dataset and applies all filtering
    and cleaning steps. Returns a cleaned DataFrame ready for graph construction.
 
    Steps:
    1. Load raw CSV and parse trending dates
    2. Remove duplicate video IDs keeping earliest trending date
    3. Sort by view count descending
    4. Remove videos with excluded title keywords
    5. Remove channels matching excluded channel patterns
    6. Remove non-English titles
    7. Remove titles with fewer than 4 words
    8. Select only relevant columns and save to CSV
    """

    df = pd.read_csv("CAvideos.csv")

    #sorting data by date then drop duplicates then sort by views
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
    df = df.sort_values('trending_date')
    df = df.drop_duplicates(subset='video_id', keep='first')
    df = df.sort_values('views', ascending=False)

    #remove videos with certain words in the title and from certain channels to avoid noise
    title_mask = ~df['title'].str.contains('|'.join(exclude_titles), case=False, na=False)
    channel_pattern_mask = ~df['channel_title'].str.contains('highlights|recap|full match|full game|re-upload|reupload', case=False, na=False)
    channel_mask = ~df['channel_title'].isin(exclude_channels)
    df = df[title_mask & channel_mask & channel_pattern_mask].reset_index(drop=True)

    #only take english titles
    df = df[df['title'].apply(is_mostly_english)].reset_index(drop=True)

    # remove videos with very short or vague titles
    df = df[df['title'].str.split().str.len() >= 4].reset_index(drop=True)

    # keep only the columns needed for graph construction and analysis
    df = df[['video_id','channel_title','title','tags','category_id','trending_date']]
    df.to_csv("filtered_youtube_trending.csv")
    print(f"Dataset size after preprocessing: {len(df)}")
    return df

def build_graph(df):
    """
    Constructs a directed influence graph from the preprocessed DataFrame.
 
    Edge construction:
    - Computes TF-IDF cosine similarity between all pairs of video titles
    - Applies edge mask: similarity > 0.50 and < 1.0, time difference 3-10 days,
      different channels
    - Enforces role constraints: event source channels can only be sources,
      react/aggregator channels can only be targets
    - Removes isolated nodes (no connections)
 
    Returns the directed graph G and the valid edge indices/similarities.
    """
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

    #build graph
    G = nx.DiGraph()
    for creator in df['channel_title'].unique():
        G.add_node(creator)

    for i, j in zip(valid_rows, valid_cols):
        creator_a, creator_b = channels[i], channels[j]
        cat_b = categories[j]
        
        # skip if target is an event source channel
        # (news/studio/sports org should never be a target)
        if creator_b in event_source_channels:
            continue

        # skip if target is in an event source category
        if cat_b in event_source_categories:
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

    return G, valid_rows, valid_cols, valid_sims, categories, channels

# =============================================================================
# NETWORK ANALYSIS
# =============================================================================

def analyze_network(G, edges_data):
    """
    Computes and prints key social network metrics for the influence graph.
 
    Metrics computed:
    - Out-degree centrality: identifies trend setters (high out-degree = many creators
      posted similar content after them)
    - In-degree centrality: identifies trend followers (high in-degree = they frequently
      post after others on similar topics)
    - Network density: fraction of possible edges that actually exist
    - Average degree: average number of connections per creator
    - Weakly connected components: number of disconnected clusters in the network
    """
    # Print sorted by out-degree (most influential first) top 10
    print("=== Out-Degree (Trend Setters) ===")
    top_out = []
    for creator, degree in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]:
        if degree > 0:
            print(f"{creator}: {degree}")
            top_out.append(creator)

    print("\n=== In-Degree (Trend Followers) ===")
    top_in = []
    for creator, degree in sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10]:
        if degree > 0:
            print(f"{creator}: {degree}")
            top_in.append(creator)

    # print all edges for top out-degree creators
    print("\n=== Edges for Top Out-Degree Creators ===")
    for e in edges_data:
        if e['source'] in top_out:
            print(f"  {e['source']} -> {e['target']}")
            print(f"    Source: {e['source_title']}")
            print(f"    Target: {e['target_title']}")
            print(f"    Similarity: {e['similarity']} | Dates: {e['source_date']} -> {e['target_date']}")

    # print all edges for top in-degree creators
    print("\n=== Edges for Top In-Degree Creators ===")
    for e in edges_data:
        if e['target'] in top_in:
            print(f"  {e['source']} -> {e['target']}")
            print(f"    Source: {e['source_title']}")
            print(f"    Target: {e['target_title']}")
            print(f"    Similarity: {e['similarity']} | Dates: {e['source_date']} -> {e['target_date']}")

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

# =============================================================================
# JSON OUTPUT
# =============================================================================

def save_json(G, df, valid_rows, valid_cols, valid_sims, categories, channels):
    """
    Saves the network nodes and edges to a JSON file for external analysis
    or visualization. Each node includes degree metrics and each edge includes
    similarity score, video titles and dates.
    """
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
                "source_category_id": int(categories[i]),
                "target_category_id": int(categories[j]),
                "source_title": df.iloc[i]['title'],
                "target_title": df.iloc[j]['title'],
                "source_date": str(df.iloc[i]['trending_date'].date()),
                "target_date": str(df.iloc[j]['trending_date'].date())
            })

    # Save to JSON
    with open("youtube_network.json", "w") as f:
        json.dump({"nodes": nodes_data, "edges": edges_data}, f, indent=2)

    print(f"Saved {len(nodes_data)} nodes and {len(edges_data)} edges to youtube_network.json")
    return nodes_data, edges_data

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def build_edge_tooltip_map(edges_data):
    """
    Builds a dictionary mapping (source, target) pairs to hover tooltip text.
    Used to display edge metadata when hovering over edges in the pyvis network.
    """

    edge_title_map = {}
    for e in edges_data:
        key = (e['source'], e['target'])
        edge_title_map[key] = (
            f"Similarity: {e['similarity']}\n"
            f"Source: {e['source_title']}\n"
            f"Target: {e['target_title']}\n"
            f"Date: {e['source_date']} → {e['target_date']}"
        )
    return edge_title_map

# =============================================================================
# NETWORK VISUALIZATION
# =============================================================================

def build_main_network(G, df, edges_data):
    """
    Builds and saves the full network visualization as an interactive HTML file.
    Nodes are colored by category and edges show hover tooltips with video titles,
    similarity scores and dates.
    """
    category_map = dict(zip(df['channel_title'], df['category_id']))
    edge_title_map = build_edge_tooltip_map(edges_data)

    net = Network(height="950px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")

    for node in G.nodes():
        cat = category_map.get(node, 0)
        color = category_colors.get(cat, '#cccccc')
        tooltip = f"{node}\nCategory: {cat}\n"
        net.add_node(node, label=node, color=color, title=tooltip)

    for src, dst in G.edges():
        edge_tooltip = edge_title_map.get((src, dst), "")
        net.add_edge(src, dst, color="rgba(255,255,255,0.3)", arrows="to", title=edge_tooltip)

    net.show_buttons(filter_=['physics'])
    net.save_graph("graphs/youtube_network.html")
    print("Saved to youtube_network.html")
    return edge_title_map

def build_community_graphs(G, df, edge_title_map):
    """
    Runs Louvain community detection on the undirected projection of the graph
    and saves a separate visualization for each community with 3 or more nodes.
    Communities are detected without enforcing category constraints — if genre-based
    clusters emerge naturally, this provides evidence of homophily.
    """
    category_map = dict(zip(df['channel_title'], df['category_id']))
    G_undirected = G.to_undirected()
    communities = community.louvain_communities(G_undirected)

    print(f"\n=== Louvain Communities ===")
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

def build_category_graphs(G, df, edge_title_map):
    """
    Generates a separate network visualization for each YouTube content category.
    Only includes nodes within the same category, showing intra-genre influence patterns.
    Note: because same-category constraint was removed from edge construction,
    most interesting cross-genre edges will not appear in these subgraphs.
    Categories with fewer than 3 nodes are skipped.
    """
    category_map = dict(zip(df['channel_title'], df['category_id']))

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

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # clean up old graph files before generating new ones
    # this ensures stale community/category graphs from previous runs are removed
    if os.path.exists("graphs"):
        shutil.rmtree("graphs")
        print("Deleted old graphs folder")
        
    #create folders for graph files
    os.makedirs("graphs/communities", exist_ok=True)
    os.makedirs("graphs/categories", exist_ok=True)

    # step 1: load and clean the dataset
    df = preprocess_data()

    # step 2: build the directed influence graph
    G, valid_rows, valid_cols, valid_sims, categories, channels = build_graph(df)

    # step 3: save nodes and edges to JSON for external use
    nodes_data, edges_data = save_json(G, df, valid_rows, valid_cols, valid_sims, categories, channels)
    
    # step 4: compute and print network metrics
    analyze_network(G, edges_data)

    # step 5: generate all network visualizations
    edge_title_map = build_main_network(G, df, edges_data) # full network
    build_community_graphs(G, df, edge_title_map) # one graph per community
    build_category_graphs(G, df, edge_title_map) # one graph per category