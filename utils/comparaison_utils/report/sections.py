#!/usr/bin/env python3
"""
Report section generators: corpus description, run description,
distance appendix, intra-topic distance, and helper utilities.
"""

import shutil
import numpy as np
from pathlib import Path

from ..constants import METRIC_REFERENCES
from ..vocabulary import extract_topic_words
from ..visualization import create_corpus_year_distribution, create_decade_breakdown_plot

from .translations import get_text, get_metric_description


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_topic_distribution_metrics(doc_assignments) -> dict:
    """
    Compute imbalance ratio and distribution entropy from document assignments.

    Returns
    -------
    dict with 'imbalance_ratio', 'distribution_entropy', 'n_topics'
    """
    if doc_assignments is None or 'topic' not in doc_assignments.columns:
        return {'imbalance_ratio': 'N/A', 'distribution_entropy': 'N/A', 'n_topics': 0}

    topic_counts = doc_assignments['topic'].value_counts()
    n_topics = len(topic_counts)

    if n_topics == 0:
        return {'imbalance_ratio': 'N/A', 'distribution_entropy': 'N/A', 'n_topics': 0}

    imbalance_ratio = topic_counts.max() / topic_counts.min() if topic_counts.min() > 0 else float('inf')

    proportions = topic_counts / topic_counts.sum()
    entropy = -np.sum(proportions * np.log(proportions + 1e-10))
    max_entropy = np.log(n_topics) if n_topics > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        'imbalance_ratio': imbalance_ratio,
        'distribution_entropy': normalized_entropy,
        'n_topics': n_topics
    }


# =============================================================================
# FIGURE COPYING
# =============================================================================

def copy_run_figures(run_dir: str, figures_dir: str, model_prefix: str) -> dict:
    """Copy existing figures from a run directory to the comparison figures directory."""
    run_dir = Path(run_dir)
    figures_dir = Path(figures_dir)
    copied = {}

    figure_files = [
        ('topic_distribution.png', 'topic_distribution'),
        ('year_topic_heatmap.png', 'year_topic_heatmap'),
        ('artist_topics_heatmap.png', 'artist_topics_heatmap'),
        ('artist_specialization.png', 'artist_specialization'),
        ('annual_js_divergence.png', 'annual_js_divergence'),
        ('silhouette_plot.png', 'silhouette_plot'),
        ('coherence_plot.png', 'coherence_plot'),
        ('umap_topics.png', 'umap_topics'),
        ('topic_pca.png', 'topic_pca'),
        ('class_distribution.png', 'class_distribution'),
    ]

    # Fallback mappings: if primary file not found, try alternative
    fallbacks = {
        'annual_js_divergence.png': 'biannual_js_divergence.png',
    }

    for filename, fig_type in figure_files:
        src = run_dir / filename
        if not src.exists() and filename in fallbacks:
            src = run_dir / fallbacks[filename]
        if src.exists():
            dest = figures_dir / f'{model_prefix}_{filename}'
            shutil.copy2(src, dest)
            copied[fig_type] = f'figures/{model_prefix}_{filename}'

    return copied


# =============================================================================
# APPENDIX: MATHEMATICAL COMPARISON OF DISTANCES
# =============================================================================

def generate_distance_appendix(lang: str = 'fr') -> str:
    """Generate appendix section explaining Labbé vs Jensen-Shannon distances."""
    t = lambda key: get_text(key, lang)

    if lang == 'fr':
        md = f"""### {t('appendix_distance_title')}

#### Labbé vs Jensen-Shannon : deux regards sur les fréquences

**Différences fondamentales**

Les deux métriques mesurent la similarité lexicale entre textes mais avec des approches différentes.

---

#### Distance de Labbé (algorithme IRAMUTEQ)

La distance de Labbé, implémentée selon l'algorithme original d'IRAMUTEQ, gère explicitement
l'asymétrie de longueur entre textes. L'algorithme procède comme suit :

1. **Identifier** le texte le plus petit ($N_{{small}}$) et le plus grand ($N_{{large}}$)
2. **Normaliser** les comptages du texte plus grand : $n'_i = n_i \\times U$ où $U = N_{{small}} / N_{{large}}$
3. **Calculer** la somme des différences absolues sur les comptages normalisés
4. **Normaliser** par le dénominateur ajusté

$$D_{{\\text{{Labbé}}}}(A, B) = \\frac{{\\sum_{{i=1}}^{{V}} |n_{{small}}(i) - n'_{{large}}(i)|}}{{N_{{small}} + \\sum_{{n'_i \\geq 1}} n'_i}}$$

Cette approche est particulièrement adaptée à la comparaison de textes de longueurs différentes.

---

#### Divergence de Jensen-Shannon

Mesure la divergence informationnelle entre les distributions de probabilité.

$$D_{{\\text{{JS}}}}(A, B) = \\frac{{1}}{{2}} D_{{\\text{{KL}}}}(P_A \\| M) + \\frac{{1}}{{2}} D_{{\\text{{KL}}}}(P_B \\| M)$$

où $M = (P_A + P_B) / 2$ est la distribution moyenne et $P$ représente les fréquences relatives.

---

#### Différence fondamentale

| Aspect | Labbé | Jensen-Shannon |
|--------|-------|----------------|
| **Entrée** | Comptages bruts normalisés | Fréquences relatives |
| **Asymétrie** | Gère explicitement | Symétrique |
| **Sensibilité** | Linéaire | Logarithmique |
| **Mots rares** | Faible impact | **Fort impact** |
| **Fondement** | Algorithme IRAMUTEQ | Théorie de l'information |

---

#### Application au rap français

**JS est plus sensible aux mots d'argot spécifiques** à certains artistes/thèmes.
**Labbé capture mieux l'homogénéité globale** du vocabulaire courant et gère mieux
les différences de longueur entre couplets.

**Recommandation :** Utiliser les deux métriques en complément :
- **Labbé** pour l'homogénéité lexicale générale (robuste aux différences de longueur)
- **JS** pour détecter les vocabulaires distinctifs

"""
    else:
        md = f"""### {t('appendix_distance_title')}

#### Labbé vs Jensen-Shannon: Two Perspectives on Lexical Similarity

**Fundamental Differences**

Both metrics measure lexical similarity between texts but with different approaches.

---

#### Labbé Distance (IRAMUTEQ algorithm)

The Labbé distance, implemented according to the original IRAMUTEQ algorithm, explicitly
handles length asymmetry between texts. The algorithm proceeds as follows:

1. **Identify** the smaller ($N_{{small}}$) and larger ($N_{{large}}$) texts
2. **Scale** the larger text's counts: $n'_i = n_i \\times U$ where $U = N_{{small}} / N_{{large}}$
3. **Compute** sum of absolute differences on scaled counts
4. **Normalize** by the adjusted denominator

$$D_{{\\text{{Labbé}}}}(A, B) = \\frac{{\\sum_{{i=1}}^{{V}} |n_{{small}}(i) - n'_{{large}}(i)|}}{{N_{{small}} + \\sum_{{n'_i \\geq 1}} n'_i}}$$

This approach is particularly suited for comparing texts of different lengths.

---

#### Jensen-Shannon Divergence

Measures the informational divergence between probability distributions.

$$D_{{\\text{{JS}}}}(A, B) = \\frac{{1}}{{2}} D_{{\\text{{KL}}}}(P_A \\| M) + \\frac{{1}}{{2}} D_{{\\text{{KL}}}}(P_B \\| M)$$

where $M = (P_A + P_B) / 2$ is the mean distribution and $P$ represents relative frequencies.

---

#### Fundamental Difference

| Aspect | Labbé | Jensen-Shannon |
|--------|-------|----------------|
| **Input** | Scaled raw counts | Relative frequencies |
| **Asymmetry** | Explicitly handled | Symmetric |
| **Sensitivity** | Linear | Logarithmic |
| **Rare words** | Low impact | **High impact** |
| **Foundation** | IRAMUTEQ algorithm | Information theory |

---

#### Application to French Rap

**JS is more sensitive to slang words specific** to certain artists/themes.
**Labbé better captures the global homogeneity** of common vocabulary and handles
length differences between verses better.

**Recommendation:** Use both metrics as complements:
- **Labbé** for general lexical homogeneity (robust to length differences)
- **JS** to detect distinctive vocabularies

"""
    return md


# =============================================================================
# INTRA-TOPIC DISTANCE SECTION (Q5)
# =============================================================================

def generate_intra_topic_distance_section(distance_results: dict, lang: str = 'fr') -> str:
    """Generate markdown section for intra-topic distance analysis (Q5)."""
    t = lambda key: get_text(key, lang)

    def get_interp_js(mean_dist):
        if mean_dist < 0.3:
            return t('high_homogeneity')
        elif mean_dist < 0.5:
            return t('moderate_homogeneity')
        elif mean_dist < 0.7:
            return t('low_homogeneity')
        else:
            return t('very_heterogeneous')

    def get_interp_labbe(mean_dist):
        if mean_dist < 0.4:
            return t('high_homogeneity')
        elif mean_dist < 0.6:
            return t('moderate_homogeneity')
        elif mean_dist < 0.8:
            return t('low_homogeneity')
        else:
            return t('very_heterogeneous')

    md = f"""### {t('q5_title')}

{t('q5_research')}

{t('q5_intro')}

#### {t('q5_method')}

{t('q5_method_text')}

| Distance | {t('what_it_captures')} | {t('scientific_justification')} |
|----------|------------------|--------------------------|
| **Jensen-Shannon** | {t('distributional_divergence')} | {t('js_justification')} |
| **Labbé** | {t('lexical_homogeneity')} | {t('labbe_justification')} |

"""

    md += f"""**Jensen-Shannon** — {METRIC_REFERENCES['js_divergence']['citation']}

{get_metric_description('js_divergence', lang)}

**Labbé** — {METRIC_REFERENCES['labbe_distance']['citation']}

{get_metric_description('labbe_distance', lang)}

#### {t('q5_results')}

"""

    # Summary table for Jensen-Shannon
    md += f"**{t('js_distributional')}**\n\n"
    md += f"| {t('model')} | {t('mean_distance')} | {t('std_dev')} | {t('n_topics')} | {t('interpretation')} |\n"
    md += "|-------|---------------|---------|----------|----------------|\n"

    for model in ['bertopic', 'lda', 'iramuteq']:
        js_results = distance_results.get(f'{model}_js', {})
        mean_dist = js_results.get('mean', 0)
        std_dist = js_results.get('std', 0)
        n_topics = js_results.get('n_topics', 0)
        interp = get_interp_js(mean_dist)
        md += f"| {model.upper()} | {mean_dist:.4f} | {std_dist:.4f} | {n_topics} | {interp} |\n"

    md += "\n"

    # Summary table for Labbé
    md += f"**{t('labbe_lexical')}**\n\n"
    md += f"| {t('model')} | {t('mean_distance')} | {t('std_dev')} | {t('n_topics')} | {t('interpretation')} |\n"
    md += "|-------|---------------|---------|----------|----------------|\n"

    for model in ['bertopic', 'lda', 'iramuteq']:
        labbe_results = distance_results.get(f'{model}_labbe', {})
        mean_dist = labbe_results.get('mean', 0)
        std_dist = labbe_results.get('std', 0)
        n_topics = labbe_results.get('n_topics', 0)
        interp = get_interp_labbe(mean_dist)
        md += f"| {model.upper()} | {mean_dist:.4f} | {std_dist:.4f} | {n_topics} | {interp} |\n"

    # Find best model for each metric
    best_js = min(
        ['bertopic', 'lda', 'iramuteq'],
        key=lambda x: distance_results.get(f'{x}_js', {}).get('mean', 1.0)
    )
    best_labbe = min(
        ['bertopic', 'lda', 'iramuteq'],
        key=lambda x: distance_results.get(f'{x}_labbe', {}).get('mean', 1.0)
    )

    js_val = distance_results.get(f'{best_js}_js', {}).get('mean', 0)
    labbe_val = distance_results.get(f'{best_labbe}_labbe', {}).get('mean', 0)

    md += f"""
{t('q5_key_obs')}

1. {t('q5_obs1').format(metric=t('q5_best_js'), model=best_js.upper(), val=js_val)}

2. {t('q5_obs1').format(metric=t('q5_best_labbe'), model=best_labbe.upper(), val=labbe_val)}

3. {t('q5_obs2')}

#### {t('q5_per_topic')}

{t('q5_per_topic_intro')}

"""

    # Per-topic details for each model
    for model in ['bertopic', 'lda', 'iramuteq']:
        js_results = distance_results.get(f'{model}_js', {})
        per_topic = js_results.get('per_topic', {})

        if not per_topic:
            continue

        sorted_topics = sorted(
            per_topic.items(),
            key=lambda x: x[1].get('mean_distance', 1.0)
        )

        md += f"**{model.upper()}**\n\n"

        if len(sorted_topics) >= 3:
            md += f"{t('q5_most_homogeneous')}\n\n"
            md += f"| {t('topic')} | {t('mean_js_distance')} | {t('n_documents')} |\n"
            md += "|-------|------------------|-------------|\n"
            for topic_id, stats in sorted_topics[:5]:
                md += f"| {topic_id} | {stats.get('mean_distance', 0):.4f} | {stats.get('n_documents', 0)} |\n"

            md += f"\n{t('q5_least_homogeneous')}\n\n"
            md += f"| {t('topic')} | {t('mean_js_distance')} | {t('n_documents')} |\n"
            md += "|-------|------------------|-------------|\n"
            for topic_id, stats in sorted_topics[-5:]:
                md += f"| {topic_id} | {stats.get('mean_distance', 0):.4f} | {stats.get('n_documents', 0)} |\n"

            md += "\n"

    md += f"\n{t('q5_appendix_ref')}\n\n"

    return md


def generate_topic_distance_4configs_section(
    topic_distance_results: dict,
    aggregation_size: int = 20,
    lang: str = 'fr'
) -> str:
    """
    Generate markdown section for all 4 distance configurations (Q5 extended).

    Parameters
    ----------
    topic_distance_results : dict
        Results from evaluate_topic_distances for each model.
        Structure: {model: {config_name: {'js': {...}, 'labbe': {...}}}}
    aggregation_size : int
        Number of verses aggregated (for display).
    lang : str
        Language for translations.

    Returns
    -------
    str
        Markdown section content.
    """
    t = lambda key: get_text(key, lang)

    # Build config names dynamically
    configs = [
        ('intra_all_paired', t('config_intra_all_paired'), t('config_intra_all_paired_desc'), 'homogeneity'),
        ('inter_all_paired', t('config_inter_all_paired'), t('config_inter_all_paired_desc'), 'separation'),
        (f'intra_aggregated_{aggregation_size}', t('config_intra_aggregated') + f' (n={aggregation_size})',
         t('config_intra_aggregated_desc'), 'homogeneity'),
        (f'inter_aggregated_{aggregation_size}', t('config_inter_aggregated') + f' (n={aggregation_size})',
         t('config_inter_aggregated_desc'), 'separation'),
    ]

    md = f"""#### {t('q5_4configs_title')}

{t('q5_4configs_intro')}

| {t('config_table_header')} | {t('what_it_measures')} | {t('interpretation_guide')} |
|--------------|----------------------|----------------------|
| **{t('config_intra_all_paired')}** | {t('homogeneity')} | {t('lower_better')} |
| **{t('config_inter_all_paired')}** | {t('separation')} | {t('higher_better')} |
| **{t('config_intra_aggregated')}** (n={aggregation_size}) | {t('homogeneity')} | {t('lower_better')} |
| **{t('config_inter_aggregated')}** (n={aggregation_size}) | {t('separation')} | {t('higher_better')} |

{t('aggregation_note').format(n=aggregation_size)}

"""

    # Generate tables for each configuration
    for config_key, config_label, config_desc, config_type in configs:
        md += f"**{config_label}**\n\n"
        md += f"*{config_desc}*\n\n"

        # Jensen-Shannon table
        md += f"| {t('model')} | JS | Labbé |\n"
        md += "|-------|------|-------|\n"

        for model in ['bertopic', 'lda', 'iramuteq']:
            model_results = topic_distance_results.get(model, {})
            config_results = model_results.get(config_key, {})
            js_mean = config_results.get('js', {}).get('mean', 0)
            labbe_mean = config_results.get('labbe', {}).get('mean', 0)
            md += f"| {model.upper()} | {js_mean:.4f} | {labbe_mean:.4f} |\n"

        md += "\n"

    # Summary: find best model for each configuration type
    md += f"{t('q5_summary_4configs')}\n\n"

    # Best homogeneity (lowest intra distance)
    best_homo_js = None
    best_homo_js_val = float('inf')
    best_homo_labbe = None
    best_homo_labbe_val = float('inf')

    for model in ['bertopic', 'lda', 'iramuteq']:
        model_results = topic_distance_results.get(model, {})
        intra_results = model_results.get('intra_all_paired', {})
        js_val = intra_results.get('js', {}).get('mean', float('inf'))
        labbe_val = intra_results.get('labbe', {}).get('mean', float('inf'))
        if js_val < best_homo_js_val:
            best_homo_js_val = js_val
            best_homo_js = model
        if labbe_val < best_homo_labbe_val:
            best_homo_labbe_val = labbe_val
            best_homo_labbe = model

    # Best separation (highest inter distance)
    best_sep_js = None
    best_sep_js_val = 0
    best_sep_labbe = None
    best_sep_labbe_val = 0

    for model in ['bertopic', 'lda', 'iramuteq']:
        model_results = topic_distance_results.get(model, {})
        inter_results = model_results.get('inter_all_paired', {})
        js_val = inter_results.get('js', {}).get('mean', 0)
        labbe_val = inter_results.get('labbe', {}).get('mean', 0)
        if js_val > best_sep_js_val:
            best_sep_js_val = js_val
            best_sep_js = model
        if labbe_val > best_sep_labbe_val:
            best_sep_labbe_val = labbe_val
            best_sep_labbe = model

    if best_homo_js:
        md += f"- **{t('q5_best_homogeneity')} (JS):** {best_homo_js.upper()} ({best_homo_js_val:.4f})\n"
    if best_homo_labbe:
        md += f"- **{t('q5_best_homogeneity')} (Labbé):** {best_homo_labbe.upper()} ({best_homo_labbe_val:.4f})\n"
    if best_sep_js:
        md += f"- **{t('q5_best_separation')} (JS):** {best_sep_js.upper()} ({best_sep_js_val:.4f})\n"
    if best_sep_labbe:
        md += f"- **{t('q5_best_separation')} (Labbé):** {best_sep_labbe.upper()} ({best_sep_labbe_val:.4f})\n"

    md += "\n"

    return md


# =============================================================================
# AGGREGATION STABILIZATION CURVE SECTION
# =============================================================================

def generate_aggregation_curve_section(
    multi_agg_results: dict,
    agg_metadata: dict = None,
    lang: str = 'fr',
) -> str:
    """
    Generate markdown section for the aggregation size vs Labbé distance curve.

    Includes the plot reference and per-aggregation-size inter-topic tables
    (one table per aggregation size, transposed: rows=topics).

    Parameters
    ----------
    multi_agg_results : dict
        {model_name: {agg_size: {mode: {'labbe': {'mean', 'std', 'per_topic'}}}}}
    agg_metadata : dict, optional
        Metadata from compute_aggregation_range: 'agg_min', 'agg_max',
        'min_topic_size', 'mean_doc_length'.
    lang : str
        Language for translations.
    """
    t = lambda key: get_text(key, lang)

    md = f"\n#### {t('q5_agg_curve_title')}\n\n"
    md += f"{t('q5_agg_curve_intro')}\n\n"

    # Show range metadata if available
    if agg_metadata:
        md += t('q5_agg_curve_range').format(
            min_agg=agg_metadata.get('agg_min', '?'),
            max_agg=agg_metadata.get('agg_max', '?'),
            min_words=agg_metadata.get('min_words_per_unit', 1000),
            min_units=agg_metadata.get('min_units_per_topic', 10),
            min_topic_size=agg_metadata.get('min_topic_size', '?'),
            n_points=agg_metadata.get('n_points', '?'),
        ) + "\n\n"

    # Figure reference
    md += f"![Aggregation Curve](figures/aggregation_curve.png)\n\n"
    md += f"{t('q5_agg_curve_fig_caption')}\n\n"

    return md


# =============================================================================
# INTER-TOPIC SEPARATION RANKING SECTION
# =============================================================================

def generate_inter_topic_ranking_section(
    centroid_results: dict,
    topic_labels_per_model: dict = None,
    lang: str = 'fr',
) -> str:
    """
    Generate markdown section for the inter-topic separation ranking.

    Uses centroid distances (each topic merged into one doc vs rest)
    for maximum discrimination.

    Parameters
    ----------
    centroid_results : dict
        {model: {'labbe': {'per_topic': {...}}, 'js': {'per_topic': {...}}}}
    topic_labels_per_model : dict, optional
        {model: {topic_id: label_string}}
    lang : str
        Language for translations.
    """
    t = lambda key: get_text(key, lang)

    if topic_labels_per_model is None:
        topic_labels_per_model = {}

    md = f"\n#### {t('q5_inter_ranking_title')}\n\n"
    md += f"{t('q5_inter_ranking_intro')}\n\n"

    for model in ['bertopic', 'lda', 'iramuteq']:
        centroid_data = centroid_results.get(model, {})
        labbe_per_topic = centroid_data.get('labbe', {}).get('per_topic', {})

        if not labbe_per_topic:
            continue

        js_per_topic = centroid_data.get('js', {}).get('per_topic', {})
        labels = topic_labels_per_model.get(model, {})

        # Figure reference
        md += f"![{model.upper()} Ranking](figures/inter_topic_ranking_{model}.png)\n\n"
        md += f"{t('q5_inter_ranking_fig_caption').format(model=model.upper())}\n\n"

        # Sort by Labbé distance
        sorted_topics = sorted(
            labbe_per_topic.items(),
            key=lambda x: x[1].get('mean_distance', 0),
            reverse=True
        )

        n_show = min(3, len(sorted_topics))

        def _label(tid):
            return labels.get(str(tid), labels.get(tid, f"Topic {tid}"))

        # Most distinct (highest distance)
        md += f"{t('q5_inter_ranking_most_distinct')}\n\n"
        md += f"| Topic | {t('mean_labbe_distance')} | {t('mean_js_inter_distance')} |\n"
        md += "|-------|-----------------|----------------|\n"
        for tid, stats in sorted_topics[:n_show]:
            labbe_val = stats.get('mean_distance', 0)
            js_val = js_per_topic.get(tid, {}).get('mean_distance', 0)
            md += f"| {_label(tid)} | {labbe_val:.4f} | {js_val:.4f} |\n"
        md += "\n"

        # Least distinct (lowest distance)
        md += f"{t('q5_inter_ranking_least_distinct')}\n\n"
        md += f"| Topic | {t('mean_labbe_distance')} | {t('mean_js_inter_distance')} |\n"
        md += "|-------|-----------------|----------------|\n"
        for tid, stats in sorted_topics[-n_show:]:
            labbe_val = stats.get('mean_distance', 0)
            js_val = js_per_topic.get(tid, {}).get('mean_distance', 0)
            md += f"| {_label(tid)} | {labbe_val:.4f} | {js_val:.4f} |\n"
        md += "\n"

    return md


# =============================================================================
# χ²/n WORD × TOPIC INDEPENDENCE TEST SECTION
# =============================================================================

def generate_word_topic_chi2_section(
    chi2_results: dict,
    topic_labels_per_model: dict = None,
    lang: str = 'fr',
) -> str:
    """
    Generate markdown section for χ²/n word × topic independence test.

    Parameters
    ----------
    chi2_results : dict
        {'non_lemmatized': {model: result_dict}, 'lemmatized': {model: result_dict}}
    topic_labels_per_model : dict, optional
        {model: {topic_id: label_string}}
    lang : str
        Language for translations.
    """
    t = lambda key: get_text(key, lang)

    if topic_labels_per_model is None:
        topic_labels_per_model = {}

    md = f"\n#### {t('q5_chi2_title')}\n\n"
    md += f"{t('q5_chi2_intro')}\n\n"
    md += f"{t('q5_chi2_explanation')}\n\n"

    for variant_key in ['non_lemmatized', 'lemmatized']:
        variant_data = chi2_results.get(variant_key, {})
        if not variant_data:
            continue

        md += f"**{t(variant_key)}**\n\n"

        # Global summary table
        md += f"| Model | χ² | {t('total_tokens_label')} | {t('chi2_over_n_label')} | {t('vocab_size_label')} |\n"
        md += "|-------|-----|--------|--------|----------|\n"

        for model in ['bertopic', 'lda', 'iramuteq']:
            r = variant_data.get(model, {})
            if not r:
                continue
            md += (f"| {model.upper()} "
                   f"| {r.get('chi2', 0):,.0f} "
                   f"| {r.get('n', 0):,} "
                   f"| {r.get('chi2_over_n', 0):.4f} "
                   f"| {r.get('vocab_size', 0):,} |\n")

        md += "\n"

    return md


# =============================================================================
# CORPUS DESCRIPTION
# =============================================================================

def generate_corpus_description(df, figures_dir: str = None, lang: str = 'fr') -> str:
    """Generate markdown section describing the corpus with visualizations."""
    t = lambda key: get_text(key, lang)
    n_docs = len(df)
    n_artists = df['artist'].nunique()

    year_col = df['year'].dropna()
    if len(year_col) > 0:
        min_year = int(year_col.min())
        max_year = int(year_col.max())
        mean_year = year_col.mean()
        median_year = year_col.median()
    else:
        min_year, max_year = "N/A", "N/A"
        mean_year, median_year = "N/A", "N/A"

    artist_counts = df['artist'].value_counts()
    mean_docs_per_artist = artist_counts.mean()
    median_docs_per_artist = artist_counts.median()
    max_docs_artist = artist_counts.idxmax()
    max_docs_count = artist_counts.max()

    decade_counts = {}
    if len(year_col) > 0:
        decades = (year_col.astype(int) // 10 * 10)
        decade_counts = decades.value_counts().sort_index().to_dict()

    top_artists = artist_counts.head(10)

    if figures_dir:
        figures_dir = Path(figures_dir)
        create_corpus_year_distribution(df, str(figures_dir / 'corpus_year_distribution.png'))
        create_decade_breakdown_plot(df, str(figures_dir / 'corpus_decade_breakdown.png'))

    mean_year_str = f"{mean_year:.1f}" if isinstance(mean_year, float) else str(mean_year)
    median_year_str = str(int(median_year)) if isinstance(median_year, float) else str(median_year)

    md = f"""## {t('corpus_description')}

{t('corpus_intro')}

### {t('dataset_overview')}

| {t('metric')} | {t('value')} |
|--------|-------|
| **{t('total_documents')}** | {n_docs:,} |
| **{t('year_range')}** | {min_year} - {max_year} |
| **{t('unique_artists')}** | {n_artists:,} |
| **{t('mean_year')}** | {mean_year_str} |
| **{t('median_year')}** | {median_year_str} |
| **{t('mean_docs_per_artist')}** | {mean_docs_per_artist:.1f} |
| **{t('median_docs_per_artist')}** | {int(median_docs_per_artist)} |
| **{t('most_prolific_artist')}** | {max_docs_artist} ({max_docs_count:,} docs) |

### {t('temporal_coverage')}

{t('temporal_coverage_text')}

"""
    if figures_dir:
        md += f"""![Distribution](figures/corpus_year_distribution.png)

{t('fig_year_dist')}

![Decades](figures/corpus_decade_breakdown.png)

{t('fig_decade')}

"""

    if decade_counts:
        md += f"""| {t('decade')} | {t('documents')} | {t('pct_corpus')} |
|--------|-----------|-------------|
"""
        for decade, count in sorted(decade_counts.items()):
            pct = count / n_docs * 100
            md += f"| {int(decade)}s | {count:,} | {pct:.1f}% |\n"
        md += "\n"

    md += f"""### {t('top_artists')}

| {t('rank')} | {t('artist')} | {t('documents')} | {t('pct_corpus')} | {t('cumulative_pct')} |
|------|--------|-----------|-------------|--------------|
"""

    cumulative = 0
    for i, (artist, count) in enumerate(top_artists.items(), 1):
        pct = count / n_docs * 100
        cumulative += pct
        md += f"| {i} | {artist} | {count:,} | {pct:.1f}% | {cumulative:.1f}% |\n"

    concentration_type = t('concentrated') if cumulative > 15 else t('diverse')
    md += f"""
**{t('corpus_concentration')}** : {t('top_10_represent')} {cumulative:.1f}% {t('of_corpus')}, {t('indicating_distribution')} {concentration_type}.

"""
    return md


# =============================================================================
# RUN DESCRIPTION
# =============================================================================

def generate_run_description(run_data: dict, model_name: str, model_type: str,
                             figures_dir: str = None, comparison_figures_dir: str = None,
                             lang: str = 'fr') -> str:
    """Generate markdown section for a single model run."""
    t = lambda key: get_text(key, lang)

    metrics = run_data.get('metrics', {})
    params = metrics.get('parameters', {})
    topics = run_data.get('topics', {})
    run_dir = run_data.get('run_dir', '')

    copied_figures = {}
    if comparison_figures_dir and run_dir:
        copied_figures = copy_run_figures(run_dir, comparison_figures_dir, model_type)

    cluster_metrics = metrics.get('cluster_metrics', {})
    doc_assignments = run_data.get('doc_assignments')

    if not cluster_metrics or 'imbalance_ratio' not in cluster_metrics:
        computed_metrics = compute_topic_distribution_metrics(doc_assignments)
        n_topics = computed_metrics['n_topics'] if computed_metrics['n_topics'] > 0 else len(topics)
        imbalance = computed_metrics['imbalance_ratio']
        entropy = computed_metrics['distribution_entropy']
    else:
        n_topics = cluster_metrics.get('n_topics', len(topics))
        imbalance = cluster_metrics.get('imbalance_ratio', 'N/A')
        entropy = cluster_metrics.get('distribution_entropy', 'N/A')

    artist_metrics = metrics.get('artist_metrics', {})
    pct_spec = artist_metrics.get('pct_specialists', 0)
    pct_mod = artist_metrics.get('pct_moderate', 0)
    pct_gen = artist_metrics.get('pct_generalists', 0)
    specialization = artist_metrics.get('specialization', 'N/A')
    js_div = artist_metrics.get('js_divergence', 'N/A')

    temporal_metrics = metrics.get('temporal_metrics', {})
    mean_variance = temporal_metrics.get('mean_variance', 'N/A')
    # Backward compat: try annual first, fall back to biannual
    mean_annual_js = temporal_metrics.get('mean_annual_js',
                     temporal_metrics.get('mean_biannual_js', 'N/A'))

    coherence_metrics = metrics.get('coherence_metrics', {})
    silhouette_metrics = metrics.get('silhouette_metrics', {})

    md = f"""### {model_name}

**{t('run_folder')}:** `{run_dir}`

"""

    # Model description
    if model_type == 'bertopic':
        md += f"{t('bertopic_desc')}\n\n"
    elif model_type == 'lda':
        md += f"{t('lda_desc')}\n\n"
    elif model_type == 'iramuteq':
        md += f"{t('iramuteq_desc')}\n\n"

    md += f"""#### {t('parameters')}

| {t('parameter')} | {t('value')} |
|-----------|-------|
"""

    skip_params = ['run_dir', 'timestamp', 'num_documents', 'vocabulary_size']
    for key, value in params.items():
        if key not in skip_params:
            if isinstance(value, dict):
                value_str = ', '.join(f'{k}={v}' for k, v in value.items())
                md += f"| {key} | {value_str} |\n"
            else:
                md += f"| {key} | `{value}` |\n"

    # LDA coherence
    if model_type == 'lda' and coherence_metrics:
        cv = coherence_metrics.get('cv', 'N/A')
        umass = coherence_metrics.get('umass', 'N/A')
        cv_str = f"{cv:.4f}" if isinstance(cv, float) else str(cv)
        umass_str = f"{umass:.4f}" if isinstance(umass, float) else str(umass)
        cv_interp = t('good') if isinstance(cv, float) and cv > 0.5 else t('moderate_interp') if isinstance(cv, float) and cv > 0.4 else t('low')
        umass_interp = t('good_closer_to_0') if isinstance(umass, float) and umass > -2 else t('moderate_interp')
        md += f"""
#### {t('coherence_scores')}

| {t('metric')} | {t('value')} | {t('interpretation')} |
|--------|-------|----------------|
| {t('cv_coherence')} | {cv_str} | {cv_interp} |
| {t('umass_coherence')} | {umass_str} | {umass_interp} |

{t('coherence_note')}

"""
        if 'coherence_plot' in copied_figures:
            md += f"![Coherence]({copied_figures['coherence_plot']})\n\n"

    # BERTopic silhouette
    if model_type == 'bertopic' and silhouette_metrics:
        sil = silhouette_metrics.get('silhouette_umap', 'N/A')
        sil_str = f"{sil:.4f}" if isinstance(sil, float) else str(sil)
        sil_interp = t('good') if isinstance(sil, float) and sil > 0.3 else t('moderate_interp') if isinstance(sil, float) and sil > 0 else t('overlapping_clusters')
        md += f"""
#### {t('clustering_quality')}

| {t('metric')} | {t('value')} | {t('interpretation')} |
|--------|-------|----------------|
| {t('silhouette_umap')} | {sil_str} | {sil_interp} |

{t('silhouette_note')}

"""
        if 'silhouette_plot' in copied_figures:
            md += f"![Silhouette]({copied_figures['silhouette_plot']})\n\n"

    # Topic distribution
    imbalance_str = str(imbalance) if isinstance(imbalance, str) else f'{imbalance:.2f}'
    entropy_str = str(entropy) if isinstance(entropy, str) else f'{entropy:.3f}'

    if isinstance(imbalance, (int, float)) and imbalance != float('inf'):
        if imbalance < 2:
            imbalance_interp = t('well_balanced')
        elif imbalance < 5:
            imbalance_interp = t('moderately_balanced')
        elif imbalance < 10:
            imbalance_interp = t('imbalanced')
        else:
            imbalance_interp = t('highly_imbalanced')
    else:
        imbalance_interp = "N/A"

    if isinstance(entropy, (int, float)):
        if entropy > 0.9:
            entropy_interp = t('near_uniform')
        elif entropy > 0.7:
            entropy_interp = t('well_distributed')
        elif entropy > 0.5:
            entropy_interp = t('moderately_concentrated')
        else:
            entropy_interp = t('highly_concentrated')
    else:
        entropy_interp = "N/A"

    md += f"""
#### {t('topic_distribution')}

| {t('metric')} | {t('value')} | {t('interpretation')} |
|--------|-------|----------------|
| {t('n_topics_label')} | {n_topics} | - |
| {t('imbalance_ratio')} | {imbalance_str} | {imbalance_interp} |
| {t('distribution_entropy')} | {entropy_str} | {entropy_interp} |

{t('metric_definitions')}

{t('imbalance_ratio_def')}

{t('entropy_def')}

"""
    if 'topic_distribution' in copied_figures:
        md += f"![Topic Distribution]({copied_figures['topic_distribution']})\n\n"
        md += f"{t('topic_dist_caption')}\n\n"

    # Artist separation
    specialization_str = f"{specialization:.3f}" if isinstance(specialization, float) else 'N/A'
    js_div_str = f"{js_div:.3f}" if isinstance(js_div, float) else 'N/A'
    md += f"""
#### {t('artist_separation')}

| {t('metric')} | {t('value')} | {t('description')} |
|--------|-------|-------------|
| % {t('specialists')} | {pct_spec:.1f}% | {t('artists_50_one_topic')} |
| % {t('moderate')} | {pct_mod:.1f}% | {t('artists_25_50')} |
| % {t('generalists')} | {pct_gen:.1f}% | {t('artists_spread')} |
| {t('specialization_index')} | {specialization_str} | {t('mean_concentration')} |
| {t('js_divergence')} | {js_div_str} | {t('artist_profile_divergence')} |

"""
    if 'artist_topics_heatmap' in copied_figures:
        md += f"![Artist-Topic Heatmap]({copied_figures['artist_topics_heatmap']})\n\n"
        md += f"{t('artist_heatmap_caption')}\n\n"

    if 'artist_specialization' in copied_figures:
        md += f"![Artist Specialization]({copied_figures['artist_specialization']})\n\n"
        md += f"{t('artist_spec_caption')}\n\n"

    # Temporal dynamics
    mean_variance_str = f"{mean_variance:.6f}" if isinstance(mean_variance, float) else 'N/A'
    mean_annual_js_str = f"{mean_annual_js:.4f}" if isinstance(mean_annual_js, float) else 'N/A'
    md += f"""
#### {t('temporal_dynamics')}

| {t('metric')} | {t('value')} |
|--------|-------|
| {t('mean_topic_variance')} | {mean_variance_str} |
| {t('mean_annual_js')} | {mean_annual_js_str} |

"""
    if 'annual_js_divergence' in copied_figures:
        md += f"![Annual JS]({copied_figures['annual_js_divergence']})\n\n"
        md += f"{t('annual_caption')}\n\n"

    if 'year_topic_heatmap' in copied_figures:
        md += f"![Year-Topic Heatmap]({copied_figures['year_topic_heatmap']})\n\n"
        md += f"{t('year_heatmap_caption')}\n\n"

    # Topics Overview
    md += f"""
#### {t('topics_overview')}

"""

    topic_ids = sorted(topics.keys(), key=lambda x: int(x) if x.lstrip('-').isdigit() else 999)

    if model_type == 'bertopic':
        for tid in topic_ids[:15]:
            topic_data = topics.get(tid, {})
            if isinstance(topic_data, dict):
                openai = topic_data.get('openai', ['N/A'])
                openai_label = openai[0] if openai else 'N/A'
                openai_label = openai_label.strip('"')

                ctfidf = topic_data.get('ctfidf', {})
                ctfidf_words = ctfidf.get('words', [])[:10] if isinstance(ctfidf, dict) else []
                ctfidf_str = ', '.join(ctfidf_words)

                keybert = topic_data.get('keybert', [])
                keybert_str = ', '.join(keybert)

                md += f"""**Topic {tid}** — *{openai_label}*

- **c-TF-IDF:** {ctfidf_str}
- **KeyBERT ({len(keybert)} terms):** {keybert_str}

"""
    else:
        md += f"| {t('topic')} | {t('top_words')} |\n"
        md += "|-------|----------|\n"

        for tid in topic_ids[:15]:
            topic_data = topics.get(tid, {})
            if isinstance(topic_data, dict):
                words = extract_topic_words(topic_data, top_n=10)
                words_str = ', '.join(words)
            else:
                words_str = str(topic_data)[:60]
            md += f"| {tid} | {words_str} |\n"

    if len(topics) > 15:
        md += f"\n{t('more_topics').format(n=len(topics) - 15)}\n"

    if 'umap_topics' in copied_figures:
        md += f"\n![UMAP]({copied_figures['umap_topics']})\n\n"
        md += f"{t('umap_caption')}\n\n"
    elif 'topic_pca' in copied_figures:
        md += f"\n![PCA]({copied_figures['topic_pca']})\n\n"
        md += f"{t('pca_caption')}\n\n"

    return md
