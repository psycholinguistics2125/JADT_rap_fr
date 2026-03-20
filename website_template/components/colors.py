# 20 topic colors (consistent across all models)
TOPIC_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

MODEL_COLORS = {"BERTopic": "#1f77b4", "LDA": "#ff7f0e", "IRAMUTEQ": "#2ca02c"}
MODEL_MARKERS = {"BERTopic": "circle", "LDA": "square", "IRAMUTEQ": "triangle-up"}


def topic_color(topic_id):
    """Get color for a topic by its ID (0-indexed internally)."""
    idx = int(topic_id) % len(TOPIC_COLORS)
    return TOPIC_COLORS[idx]
