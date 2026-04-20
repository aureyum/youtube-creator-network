
# YouTube Creator Network Analysis

A Python tool that analyzes influence patterns between YouTube content creators by detecting similar trending videos and building a directed influence graph.

## Overview

This project constructs a network where:
- **Nodes** represent YouTube channels
- **Edges** represent potential influence (creator A likely inspired creator B)
- **Edge weight** is based on video title similarity, timing, and content category

## Key Features

- **TF-IDF similarity matching** on video titles to detect related content
- **Temporal constraints** (edges only connect videos 3-10 days apart)
- **Role-based filtering**: event sources only as sources, reaction channels only as targets
- **Community detection** using Louvain algorithm
- **Interactive visualizations** with PyVis for network exploration
- **JSON export** for external analysis

## Installation

```bash
pip install pandas scikit-learn networkx pyvis numpy
```

## Data

Requires `CAvideos.csv` (Canadian YouTube trending dataset) from https://www.kaggle.com/datasets/datasnaek/youtube-new with columns:
- `video_id`, `channel_title`, `title`, `tags`, `category_id`, `trending_date`, `views`

## Usage

```bash
python youtube_creator_network.py
```

Generates:
- `filtered_youtube_trending.csv` - cleaned dataset
- `youtube_network.json` - nodes and edges
- `graphs/youtube_network.html` - full network visualization
- `graphs/communities/community_*.html` - per-community visualizations
- `graphs/categories/network_cat_*.html` - per-category visualizations

## Configuration

Edit global constants for:
- Stop words and exclusion lists
- Event source categories and channels
- Similarity thresholds
- Category color mapping
