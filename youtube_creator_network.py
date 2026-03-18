import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

valid_categories = [21, 20, 24, 22, 23, 26, 28] #video blogging, gaming, entertainment, people and blogs, comedy, howto and style, science and technology

df = pd.read_csv("CAvideos.csv")

#sorting data
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df = df.sort_values('trending_date') #keep more recent videos
df = df.drop_duplicates(subset='video_id', keep='first') #remove all video duplicates
df = df.sort_values('views', ascending=False) #sort by views
df = df[df['category_id'].isin(valid_categories)] #keep videos only in categories valid_categories
df = df[~df['title'].str.contains("trailer", case=False, na=False)] #remove movie trailers, sometimes it's in people and blogs
df = df[~df['title'].str.contains("teaser", case=False, na=False)] #remove movie teasers
df = df[~df['title'].str.contains("music video", case=False, na=False)] #remove music videos
df = df[~df['title'].str.contains("series", case=False, na=False)] #remove movie teasers
df = df.head(2000) #take first 1000 most popular views

print(len(df))

creators = df[['video_id','channel_title','title','tags','category_id','trending_date']]

df.to_csv("filtered_youtube_trending.csv") #write data to new csv file

#removes stop words (the, an, etc), combines title and tags and calculates the similarity matrix
vectorizer = TfidfVectorizer(stop_words='english')
df["text"] = df["title"] + " " + df["tags"] 
tfidf_matrix = vectorizer.fit_transform(df["text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

print(similarity_matrix)

#creates graph and add creators as nodes
G = nx.DiGraph()
for creator in df['channel_title'].unique():
    G.add_node(creator)

# Print all edges with video titles
for i in range(len(df)):
    for j in range(i+1, len(df)):
        time_diff = (df.iloc[j]['trending_date'] - df.iloc[i]['trending_date']).days
        creator_a = df.iloc[i]['channel_title']
        creator_b = df.iloc[j]['channel_title']

        if creator_a == creator_b:
            continue
        
        sim = similarity_matrix[i][j]
        
        if (sim > 0.15 and df.iloc[i]['category_id'] == df.iloc[j]['category_id'] and 0 < time_diff <= 7):
            title_a = df.iloc[i]['title']
            title_b = df.iloc[j]['title']
            print(f"{creator_a} -> {creator_b}")
            print(f"    {title_a}  -->  {title_b}")
            print(f"    similarity = {sim:.2f}\n")
            G.add_edge(creator_a, creator_b)

#removing the creators that have no relations to other creators
G.remove_nodes_from(list(nx.isolates(G)))     

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=50,
    font_size=6
)

plt.savefig("youtube_network.png", dpi=300)

plt.show()