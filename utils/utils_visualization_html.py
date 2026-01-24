#!/usr/bin/env python3
"""
Interactive HTML Visualization for BERTopic
=============================================
Creates pyLDAvis-style interactive HTML visualizations for BERTopic models.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def prepare_visualization_data(topic_model, topics: np.ndarray, umap_embeddings: np.ndarray,
                                df: pd.DataFrame, topics_desc: dict,
                                top_artists: int = 10, top_examples: int = 5) -> dict:
    """
    Prepare all data needed for the interactive HTML visualization.

    Args:
        topic_model: Fitted BERTopic model
        topics: Topic assignments for each document
        umap_embeddings: UMAP-reduced embeddings
        df: DataFrame with document metadata (artist, year, lyrics_cleaned)
        topics_desc: Topic descriptions dict from display_topics()
        top_artists: Number of top artists to show per topic
        top_examples: Number of example documents per topic

    Returns:
        Dict with all visualization data ready for JSON embedding
    """
    # Get 2D coordinates for scatter plot
    if umap_embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(umap_embeddings)
    else:
        coords_2d = umap_embeddings

    # Prepare document data for scatter plot
    docs_data = []
    for i in range(len(df)):
        text_preview = str(df.iloc[i]['lyrics_cleaned'])[:150] + "..." if len(str(df.iloc[i]['lyrics_cleaned'])) > 150 else str(df.iloc[i]['lyrics_cleaned'])
        docs_data.append({
            'x': float(coords_2d[i, 0]),
            'y': float(coords_2d[i, 1]),
            'topic': int(topics[i]),
            'artist': str(df.iloc[i]['artist']),
            'year': int(df.iloc[i]['year']) if pd.notna(df.iloc[i]['year']) else None,
            'title': str(df.iloc[i].get('title', '')),
            'text_preview': text_preview,
        })

    # Prepare per-topic data
    topics_data = {}
    unique_topics = sorted([int(t) for t in np.unique(topics) if t >= 0])

    for topic_id in unique_topics:
        topic_mask = topics == topic_id
        topic_df = df[topic_mask]

        # Get topic description if available
        topic_info = topics_desc.get(topic_id, {})

        # Keywords from different representations
        keywords = {}
        if 'ctfidf' in topic_info:
            ctfidf_data = topic_info['ctfidf']
            if isinstance(ctfidf_data, dict):
                keywords['c-TF-IDF'] = ctfidf_data.get('words', [])[:20]
            elif isinstance(ctfidf_data, list):
                keywords['c-TF-IDF'] = ctfidf_data[:20]
        if 'mmr' in topic_info:
            mmr_data = topic_info['mmr']
            if isinstance(mmr_data, dict):
                keywords['MMR'] = mmr_data.get('words', [])[:20]
            elif isinstance(mmr_data, list):
                keywords['MMR'] = mmr_data[:20]
        if 'keybert' in topic_info:
            keybert_data = topic_info['keybert']
            if isinstance(keybert_data, dict):
                keywords['KeyBERT'] = keybert_data.get('words', [])[:20]
            elif isinstance(keybert_data, list):
                keywords['KeyBERT'] = keybert_data[:20]

        # OpenAI label if available
        openai_label = topic_info.get('openai', None)

        # Top artists
        artist_counts = topic_df['artist'].value_counts().head(top_artists)
        top_artists_list = [
            {'name': artist, 'count': int(count)}
            for artist, count in artist_counts.items()
        ]

        # Year distribution
        year_dist = topic_df['year'].value_counts().sort_index()
        year_distribution = {int(year): int(count) for year, count in year_dist.items() if pd.notna(year)}

        # Example documents
        examples = []
        sample_df = topic_df.sample(min(top_examples, len(topic_df)), random_state=42) if len(topic_df) > 0 else topic_df
        for _, row in sample_df.iterrows():
            text = str(row['lyrics_cleaned'])
            examples.append({
                'artist': str(row['artist']),
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'title': str(row.get('title', '')),
                'text': text[:300] + "..." if len(text) > 300 else text,
            })

        topics_data[int(topic_id)] = {
            'count': int(topic_mask.sum()),
            'label': openai_label,
            'keywords': keywords,
            'top_artists': top_artists_list,
            'year_distribution': year_distribution,
            'examples': examples,
        }

    # Metadata
    metadata = {
        'n_documents': len(df),
        'n_topics': len(unique_topics),
        'year_range': [int(df['year'].min()), int(df['year'].max())] if df['year'].notna().any() else None,
        'timestamp': datetime.now().isoformat(),
    }

    return {
        'documents': docs_data,
        'topics': topics_data,
        'metadata': metadata,
    }


def create_interactive_bertopic_html(vis_data: dict, output_path: str,
                                      title: str = "BERTopic Interactive Visualization"):
    """
    Generate a self-contained interactive HTML file for BERTopic visualization.

    Args:
        vis_data: Visualization data from prepare_visualization_data()
        output_path: Path to save the HTML file
        title: Title for the visualization
    """
    # Generate topic colors (Plotly qualitative palette)
    n_topics = vis_data['metadata']['n_topics']
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    ]
    topic_colors = {}
    topic_ids = sorted(vis_data['topics'].keys())
    for i, tid in enumerate(topic_ids):
        topic_colors[tid] = colors[i % len(colors)]
    topic_colors[-1] = '#cccccc'  # Outliers

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #ffffff;
            color: #333333;
            line-height: 1.5;
        }}
        .container {{
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        .left-panel {{
            width: 45%;
            padding: 20px;
            background: #ffffff;
            min-width: 0;
        }}
        .right-panel {{
            width: 55%;
            background: #f8f9fa;
            border-left: 1px solid #e0e0e0;
            overflow-y: auto;
            padding: 20px;
        }}
        .topic-selector {{
            margin-bottom: 15px;
        }}
        .topic-selector select {{
            width: 100%;
            padding: 10px;
            font-size: 0.9rem;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background: white;
            cursor: pointer;
        }}
        .topic-selector select:focus {{
            outline: none;
            border-color: #1976d2;
        }}
        h1 {{
            font-size: 1.5rem;
            color: #1976d2;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        h2 {{
            font-size: 1.2rem;
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        h3 {{
            font-size: 1rem;
            color: #555;
            margin: 15px 0 8px 0;
            font-weight: 600;
        }}
        .metadata {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 20px;
        }}
        .topic-header {{
            background: linear-gradient(135deg, #1976d2, #1565c0);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .topic-header h2 {{
            color: white;
            margin: 0;
        }}
        .topic-header .label {{
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 5px;
        }}
        .topic-header .count {{
            font-size: 0.85rem;
            opacity: 0.8;
            margin-top: 3px;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .keywords-section {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .keyword-type {{
            margin-bottom: 8px;
        }}
        .keyword-type .type-label {{
            font-size: 0.75rem;
            color: #1976d2;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .keyword-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .keyword {{
            background: #e3f2fd;
            color: #1565c0;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }}
        .artist-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #eee;
            font-size: 0.9rem;
        }}
        .artist-item:last-child {{
            border-bottom: none;
        }}
        .artist-name {{
            font-weight: 500;
        }}
        .artist-count {{
            color: #666;
        }}
        .year-chart {{
            height: 80px;
            margin-top: 10px;
        }}
        .example {{
            padding: 10px;
            background: #fafafa;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 3px solid #1976d2;
        }}
        .example-meta {{
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 5px;
        }}
        .example-text {{
            font-size: 0.85rem;
            color: #444;
            font-style: italic;
        }}
        .placeholder {{
            text-align: center;
            color: #999;
            padding: 40px;
        }}
        #plot {{
            width: 100%;
            height: calc(100% - 80px);
        }}
        .legend-info {{
            font-size: 0.8rem;
            color: #666;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <h1>{title}</h1>
            <div class="metadata">
                {vis_data['metadata']['n_documents']:,} documents &bull;
                {vis_data['metadata']['n_topics']} topics &bull;
                Years {vis_data['metadata']['year_range'][0]}-{vis_data['metadata']['year_range'][1] if vis_data['metadata']['year_range'] else 'N/A'}
            </div>
            <div id="plot"></div>
            <div class="legend-info">Click on a cluster to see topic details</div>
        </div>
        <div class="right-panel">
            <div class="topic-selector">
                <select id="topic-dropdown" onchange="onTopicSelect(this.value)">
                    <option value="">-- Select a Topic --</option>
                </select>
            </div>
            <div id="details-panel">
                <div class="placeholder">
                    <h2>Select a Topic</h2>
                    <p>Click on a point cluster or use the dropdown above to view topic details.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const visData = {json.dumps(vis_data)};
        const topicColors = {json.dumps(topic_colors)};

        // Populate topic dropdown
        function populateDropdown() {{
            const dropdown = document.getElementById('topic-dropdown');
            const topicIds = Object.keys(visData.topics).map(Number).sort((a, b) => a - b);

            for (const tid of topicIds) {{
                const topic = visData.topics[tid];
                // Get KeyBERT terms, or fall back to c-TF-IDF
                let terms = topic.keywords['KeyBERT'] || topic.keywords['c-TF-IDF'] || [];
                const termStr = terms.slice(0, 3).join(' | ');
                const option = document.createElement('option');
                option.value = tid;
                option.textContent = `Topic n°${{tid}}: ${{termStr}}`;
                dropdown.appendChild(option);
            }}
        }}

        function onTopicSelect(value) {{
            if (value !== '') {{
                showTopicDetails(parseInt(value));
            }}
        }}

        populateDropdown();

        // Prepare scatter plot data
        const docs = visData.documents;
        const x = docs.map(d => d.x);
        const y = docs.map(d => d.y);
        const colors = docs.map(d => topicColors[d.topic] || '#cccccc');
        const hoverText = docs.map(d =>
            `Topic ${{d.topic}}<br>Artist: ${{d.artist}}<br>Year: ${{d.year || 'N/A'}}<br>${{d.text_preview}}`
        );

        const trace = {{
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            marker: {{
                color: colors,
                size: 5,
                opacity: 0.6,
            }},
            text: hoverText,
            hoverinfo: 'text',
            customdata: docs.map(d => d.topic),
        }};

        const layout = {{
            showlegend: false,
            hovermode: 'closest',
            xaxis: {{
                title: 'UMAP 1',
                showgrid: true,
                gridcolor: '#f0f0f0',
                zeroline: false,
            }},
            yaxis: {{
                title: 'UMAP 2',
                showgrid: true,
                gridcolor: '#f0f0f0',
                zeroline: false,
            }},
            margin: {{ l: 50, r: 20, t: 20, b: 50 }},
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
        }};

        const config = {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        }};

        Plotly.newPlot('plot', [trace], layout, config);

        // Click handler
        document.getElementById('plot').on('plotly_click', function(data) {{
            const topicId = data.points[0].customdata;
            showTopicDetails(topicId);
            // Sync dropdown
            document.getElementById('topic-dropdown').value = topicId;
        }});

        function showTopicDetails(topicId) {{
            const panel = document.getElementById('details-panel');
            const topic = visData.topics[topicId];

            if (!topic) {{
                panel.innerHTML = '<div class="placeholder"><h2>Outlier Topic (-1)</h2><p>This cluster contains documents that don\\'t fit well into any topic.</p></div>';
                return;
            }}

            let html = `
                <div class="topic-header" style="background: linear-gradient(135deg, ${{topicColors[topicId]}}, ${{topicColors[topicId]}}dd);">
                    <h2>Topic ${{topicId}}</h2>
                    ${{topic.label ? `<div class="label">${{topic.label}}</div>` : ''}}
                    <div class="count">${{topic.count.toLocaleString()}} documents</div>
                </div>
            `;

            // Keywords section
            if (Object.keys(topic.keywords).length > 0) {{
                html += '<div class="section"><h3>Keywords</h3><div class="keywords-section">';
                for (const [type, words] of Object.entries(topic.keywords)) {{
                    if (words && words.length > 0) {{
                        html += `
                            <div class="keyword-type">
                                <div class="type-label">${{type}}</div>
                                <div class="keyword-list">
                                    ${{words.slice(0, 20).map(w => `<span class="keyword">${{w}}</span>`).join('')}}
                                </div>
                            </div>
                        `;
                    }}
                }}
                html += '</div></div>';
            }}

            // Top Artists
            if (topic.top_artists && topic.top_artists.length > 0) {{
                html += '<div class="section"><h3>Top Artists</h3>';
                for (const artist of topic.top_artists.slice(0, 15)) {{
                    html += `
                        <div class="artist-item">
                            <span class="artist-name">${{artist.name}}</span>
                            <span class="artist-count">${{artist.count}} docs</span>
                        </div>
                    `;
                }}
                html += '</div>';
            }}

            // Year distribution mini chart
            if (topic.year_distribution && Object.keys(topic.year_distribution).length > 0) {{
                html += `<div class="section"><h3>Year Distribution</h3><div id="year-chart-${{topicId}}" class="year-chart"></div></div>`;
            }}

            // Example documents
            if (topic.examples && topic.examples.length > 0) {{
                html += '<div class="section"><h3>Example Documents</h3>';
                for (const ex of topic.examples) {{
                    html += `
                        <div class="example">
                            <div class="example-meta">${{ex.artist}} ${{ex.year ? '(' + ex.year + ')' : ''}} ${{ex.title ? '- ' + ex.title : ''}}</div>
                            <div class="example-text">"${{ex.text}}"</div>
                        </div>
                    `;
                }}
                html += '</div>';
            }}

            panel.innerHTML = html;

            // Render year chart if data exists
            if (topic.year_distribution && Object.keys(topic.year_distribution).length > 0) {{
                const years = Object.keys(topic.year_distribution).map(Number).sort((a, b) => a - b);
                const counts = years.map(y => topic.year_distribution[y]);

                Plotly.newPlot(`year-chart-${{topicId}}`, [{{
                    x: years,
                    y: counts,
                    type: 'bar',
                    marker: {{ color: topicColors[topicId] }},
                }}], {{
                    margin: {{ l: 30, r: 10, t: 5, b: 25 }},
                    xaxis: {{ tickfont: {{ size: 10 }}, dtick: 5 }},
                    yaxis: {{ tickfont: {{ size: 10 }} }},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                }}, {{
                    responsive: true,
                    displayModeBar: false,
                }});
            }}
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  Interactive HTML saved to: {output_path}")
