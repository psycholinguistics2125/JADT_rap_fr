#!/usr/bin/env python3
"""
LaTeX report generator with proper equation rendering.
"""

from datetime import datetime
from pathlib import Path

from ..constants import METRIC_REFERENCES

from .translations import get_text
from .latex_helpers import (
    LATEX_PREAMBLE,
    LATEX_END,
    latex_escape,
    latex_safe_number,
    markdown_to_latex,
    generate_latex_table,
    generate_latex_figure,
)
from .sections import compute_topic_distribution_metrics


def generate_latex_report(results: dict, output_dir, figures_dir=None,
                          lang: str = 'fr') -> str:
    """
    Generate a pure LaTeX comparison report with proper equation rendering.

    Parameters
    ----------
    results : dict
        Results dictionary containing all comparison data.
    output_dir : str
        Path to output directory.
    figures_dir : str, optional
        Path to figures directory.
    lang : str, default='fr'
        Language for the report ('fr' for French, 'en' for English).

    Returns
    -------
    str
        Full LaTeX document content.
    """
    t = lambda key: get_text(key, lang)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    figures_dir_path = Path(output_dir) / 'figures' if not figures_dir else Path(figures_dir)
    babel_lang = 'french' if lang == 'fr' else 'english'

    # Start document
    tex = LATEX_PREAMBLE.replace('{babel_lang}', babel_lang)

    # Title
    tex += r'''
\title{\textbf{''' + latex_escape(t('title')) + r'''}}
\author{Topic Modeling Comparison Framework}
\date{''' + timestamp + r'''}
\maketitle

\tableofcontents
\newpage

'''

    # Abstract
    tex += r'\section*{' + latex_escape(t('abstract_title')) + r'}' + '\n'
    tex += r'\addcontentsline{toc}{section}{' + latex_escape(t('abstract_title')) + r'}' + '\n\n'
    tex += markdown_to_latex(t('abstract_text')) + '\n\n'
    tex += r'\newpage' + '\n\n'

    # Extract topic labels for use in all section generators
    topic_labels_per_model = results.get('topic_labels_per_model', {})
    centroid_results = results.get('centroid_results', {})

    # Section 1: Corpus Description
    tex += r'\clearpage' + '\n'
    tex += r'\section{' + latex_escape(t('corpus_description')) + r'}' + '\n\n'
    tex += _generate_latex_corpus_section(results, figures_dir_path, lang)

    # Section 2: Individual Models
    tex += r'\clearpage' + '\n'
    tex += r'\section{' + latex_escape(t('individual_models')) + r'}' + '\n\n'
    tex += markdown_to_latex(t('individual_models_intro')) + '\n\n'

    tex += r'\clearpage' + '\n'
    tex += r'\subsection{BERTopic}' + '\n'
    tex += _generate_latex_model_section(results['bertopic'], 'bertopic', figures_dir_path, lang)

    tex += r'\clearpage' + '\n'
    tex += r'\subsection{LDA}' + '\n'
    tex += _generate_latex_model_section(results['lda'], 'lda', figures_dir_path, lang)

    tex += r'\clearpage' + '\n'
    tex += r'\subsection{IRAMUTEQ}' + '\n'
    tex += _generate_latex_model_section(results['iramuteq'], 'iramuteq', figures_dir_path, lang)

    # Section 3: Comparative Analysis
    tex += r'\clearpage' + '\n'
    tex += r'\section{' + latex_escape(t('comparative_analysis')) + r'}' + '\n\n'
    tex += _generate_latex_comparison_section(results, lang)

    # Section 4: Intra-topic Distance (Q5)
    tex += r'\clearpage' + '\n'
    distance_results = results.get('intra_topic_distances', {})
    if distance_results:
        tex += _generate_latex_distance_section(distance_results, lang)

    # Q5 Extended: All 4 distance configurations (if available)
    topic_distance_results = results.get('topic_distance_results', {})
    aggregation_size = results.get('aggregation_size', 20)
    if topic_distance_results:
        tex += r'\FloatBarrier' + '\n'
        tex += _generate_latex_distance_4configs_section(
            topic_distance_results, aggregation_size, lang)

    # Q5 Feature: Aggregation stabilization curve
    multi_agg_results = results.get('multi_agg_results', {})
    if multi_agg_results:
        tex += r'\FloatBarrier' + '\n'
        tex += _generate_latex_aggregation_curve_section(
            multi_agg_results, results.get('agg_metadata'), figures_dir_path, lang)

    # Q5 Feature: Inter-topic separation ranking (uses centroid distances)
    if centroid_results:
        tex += r'\FloatBarrier' + '\n'
        tex += _generate_latex_inter_topic_ranking_section(
            centroid_results, figures_dir_path, topic_labels_per_model, lang)

    # Q5 Feature: χ²/n word × topic independence test
    chi2_results = results.get('chi2_results', {})
    if chi2_results:
        tex += r'\FloatBarrier' + '\n'
        tex += _generate_latex_chi2_section(chi2_results, topic_labels_per_model, lang)

    import re as _re

    def _strip_md_heading(s):
        """Strip markdown heading prefix (### ), numbered (1.) and letter (B.) prefixes from translation strings."""
        s = _re.sub(r'^#{1,5}\s+', '', s)
        s = _re.sub(r'^\d+\.\s*', '', s)
        s = _re.sub(r'^[A-Z]\.\s*', '', s)
        return s

    # Section 5: Summary — dynamic conclusion from computed results
    tex += r'\clearpage' + '\n'
    tex += r'\section{' + latex_escape(_strip_md_heading(t('summary_title'))) + r'}' + '\n\n'
    # Use the same dynamic conclusion builder as markdown, converted to LaTeX
    from .markdown_report import _build_dynamic_conclusion
    conclusion_md = _build_dynamic_conclusion(results, lang)
    tex += markdown_to_latex(conclusion_md) + '\n\n'

    # References (full citations: authors, title, journal)
    def latex_full_ref(key):
        ref = METRIC_REFERENCES[key]
        return latex_escape(f"{ref['citation']}. {ref['paper']}")

    tex += r'\section{' + latex_escape(_strip_md_heading(t('references_title'))) + r'}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('clustering_agreement_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('ari') + '\n'
    tex += r'\item ' + latex_full_ref('nmi') + '\n'
    tex += r'\item ' + latex_full_ref('ami') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('association_measures_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('cramers_v') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('info_theory_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('js_divergence') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('intertextual_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('labbe_distance') + '\n'
    for extra in METRIC_REFERENCES['labbe_distance'].get('additional_refs', []):
        tex += r'\item ' + latex_escape(extra) + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('topic_coherence_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('coherence_cv') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('cluster_validation_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('silhouette') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(_strip_md_heading(t('topic_modeling_refs'))) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_escape('Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.') + '\n'
    tex += r'\item ' + latex_escape('Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.') + '\n'
    tex += r'\item ' + latex_escape('Reinert, M. (1983). Une méthode de classification descendante hiérarchique : application à l\'analyse lexicale par contexte. Les Cahiers de l\'Analyse des Données, 8(2), 187-198.') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    # Appendix with distance formulas
    tex += r'\appendix' + '\n'
    appendix_title = _re.sub(r'^[A-Z]\.\s*', '', _strip_md_heading(t('appendix_distance_title')))
    tex += r'\section{' + latex_escape(appendix_title) + r'}' + '\n\n'
    tex += _generate_latex_distance_appendix(lang)

    # End document
    tex += LATEX_END

    return tex


def _generate_latex_corpus_section(results: dict, figures_dir: Path, lang: str) -> str:
    """Generate LaTeX content for corpus description section."""
    t = lambda key: get_text(key, lang)
    df = results['bertopic']['doc_assignments']

    n_docs = len(df)
    n_artists = df['artist'].nunique()
    year_col = df['year'].dropna()

    if len(year_col) > 0:
        min_year = int(year_col.min())
        max_year = int(year_col.max())
        mean_year = f"{year_col.mean():.1f}"
        median_year = int(year_col.median())
    else:
        min_year, max_year = "N/A", "N/A"
        mean_year, median_year = "N/A", "N/A"

    artist_counts = df['artist'].value_counts()
    mean_docs = f"{artist_counts.mean():.1f}"
    median_docs = int(artist_counts.median())
    max_artist = latex_escape(artist_counts.idxmax())
    max_count = artist_counts.max()

    tex = markdown_to_latex(t('corpus_intro')) + '\n\n'

    # Dataset overview table
    tex += r'\subsection{' + latex_escape(t('dataset_overview')) + r'}' + '\n\n'
    headers = [t('metric'), t('value')]
    rows = [
        [r'\textbf{' + latex_escape(t('total_documents')) + r'}', f'{n_docs:,}'],
        [r'\textbf{' + latex_escape(t('year_range')) + r'}', f'{min_year} -- {max_year}'],
        [r'\textbf{' + latex_escape(t('unique_artists')) + r'}', f'{n_artists:,}'],
        [r'\textbf{' + latex_escape(t('mean_year')) + r'}', mean_year],
        [r'\textbf{' + latex_escape(t('median_year')) + r'}', str(median_year)],
        [r'\textbf{' + latex_escape(t('mean_docs_per_artist')) + r'}', mean_docs],
        [r'\textbf{' + latex_escape(t('median_docs_per_artist')) + r'}', str(median_docs)],
        [r'\textbf{' + latex_escape(t('most_prolific_artist')) + r'}', f'{max_artist} ({max_count:,} docs)'],
    ]
    tex += generate_latex_table(headers, rows, caption=t('dataset_overview'))

    # Include figures if they exist
    corpus_fig = figures_dir / 'corpus_year_distribution.png'
    if corpus_fig.exists():
        caption = t('fig_year_dist').strip('*').replace('*', '')
        tex += generate_latex_figure(str(corpus_fig), caption=caption, label='fig:year_dist')

    return tex


def _generate_latex_model_section(run_data: dict, model_type: str, figures_dir: Path, lang: str) -> str:
    """Generate LaTeX content for a single model section."""
    t = lambda key: get_text(key, lang)

    metrics = run_data.get('metrics', {})
    run_dir = run_data.get('run_dir', '')
    topics = run_data.get('topics', {})

    tex = r'\textbf{' + latex_escape(t('run_folder')) + r':} \texttt{' + latex_escape(run_dir) + r'}' + '\n\n'

    # Model description
    if model_type == 'bertopic':
        tex += markdown_to_latex(t('bertopic_desc')) + '\n\n'
    elif model_type == 'lda':
        tex += markdown_to_latex(t('lda_desc')) + '\n\n'
    elif model_type == 'iramuteq':
        tex += markdown_to_latex(t('iramuteq_desc')) + '\n\n'

    # Parameters table
    params = metrics.get('parameters', {})
    if params:
        tex += r'\subsubsection{' + latex_escape(t('parameters')) + r'}' + '\n\n'
        headers = [t('parameter'), t('value')]
        rows = []
        skip_params = ['run_dir', 'timestamp', 'num_documents', 'vocabulary_size']
        for key, value in params.items():
            if key not in skip_params:
                if isinstance(value, dict):
                    value_str = ', '.join(f'{k}={v}' for k, v in value.items())
                else:
                    value_str = str(value)
                rows.append([latex_escape(key), r'\texttt{' + latex_escape(value_str) + r'}'])
        if rows:
            tex += generate_latex_table(headers, rows)

    # Topic distribution metrics
    cluster_metrics = metrics.get('cluster_metrics', {})
    doc_assignments = run_data.get('doc_assignments')

    if not cluster_metrics or 'imbalance_ratio' not in cluster_metrics:
        computed = compute_topic_distribution_metrics(doc_assignments)
        n_topics = computed['n_topics'] if computed['n_topics'] > 0 else len(topics)
        imbalance = computed['imbalance_ratio']
        entropy = computed['distribution_entropy']
    else:
        n_topics = cluster_metrics.get('n_topics', len(topics))
        imbalance = cluster_metrics.get('imbalance_ratio', 'N/A')
        entropy = cluster_metrics.get('distribution_entropy', 'N/A')

    tex += r'\subsubsection{' + latex_escape(t('topic_distribution')) + r'}' + '\n\n'
    headers = [t('metric'), t('value'), t('interpretation')]
    rows = [
        [latex_escape(t('n_topics_label')), str(n_topics), '--'],
        [latex_escape(t('imbalance_ratio')), latex_safe_number(imbalance, '.2f'),
         _get_imbalance_interp(imbalance, t)],
        [latex_escape(t('distribution_entropy')), latex_safe_number(entropy, '.3f'),
         _get_entropy_interp(entropy, t)],
    ]
    tex += generate_latex_table(headers, rows)

    # LDA coherence
    if model_type == 'lda':
        coherence = metrics.get('coherence_metrics', {})
        if coherence:
            tex += r'\subsubsection{' + latex_escape(t('coherence_scores')) + r'}' + '\n\n'
            cv = coherence.get('cv', 'N/A')
            umass = coherence.get('umass', 'N/A')
            headers = [t('metric'), t('value'), t('interpretation')]
            rows = [
                [r'$C_v$ Coherence', latex_safe_number(cv),
                 t('good') if isinstance(cv, float) and cv > 0.5 else t('moderate_interp')],
                [r'$C_{UMass}$ Coherence', latex_safe_number(umass),
                 t('good_closer_to_0') if isinstance(umass, float) and umass > -2 else t('moderate_interp')],
            ]
            tex += generate_latex_table(headers, rows)

    # BERTopic silhouette
    if model_type == 'bertopic':
        silhouette = metrics.get('silhouette_metrics', {})
        if silhouette:
            tex += r'\subsubsection{' + latex_escape(t('clustering_quality')) + r'}' + '\n\n'
            sil = silhouette.get('silhouette_umap', 'N/A')
            headers = [t('metric'), t('value'), t('interpretation')]
            rows = [
                [latex_escape(t('silhouette_umap')), latex_safe_number(sil),
                 t('good') if isinstance(sil, float) and sil > 0.3 else t('moderate_interp')],
            ]
            tex += generate_latex_table(headers, rows)

    # Include model-specific figures
    fig_prefix = model_type
    for fig_name, fig_caption_key in [
        ('topic_distribution.png', 'topic_dist_caption'),
        ('year_topic_heatmap.png', 'year_heatmap_caption'),
        ('artist_topics_heatmap.png', 'artist_heatmap_caption'),
    ]:
        fig_path = figures_dir / f'{fig_prefix}_{fig_name}'
        if fig_path.exists():
            # Strip *italic* markers from caption text before passing to generate_latex_figure
            caption = t(fig_caption_key).strip('*').replace('*', '')
            tex += generate_latex_figure(str(fig_path), caption=caption)

    return tex


def _get_imbalance_interp(imbalance, t) -> str:
    """Get interpretation string for imbalance ratio."""
    if not isinstance(imbalance, (int, float)) or imbalance == float('inf'):
        return 'N/A'
    if imbalance < 2:
        return latex_escape(t('well_balanced'))
    elif imbalance < 5:
        return latex_escape(t('moderately_balanced'))
    elif imbalance < 10:
        return latex_escape(t('imbalanced'))
    else:
        return latex_escape(t('highly_imbalanced'))


def _get_entropy_interp(entropy, t) -> str:
    """Get interpretation string for distribution entropy."""
    if not isinstance(entropy, (int, float)):
        return 'N/A'
    if entropy > 0.9:
        return latex_escape(t('near_uniform'))
    elif entropy > 0.7:
        return latex_escape(t('well_distributed'))
    elif entropy > 0.5:
        return latex_escape(t('moderately_concentrated'))
    else:
        return latex_escape(t('highly_concentrated'))


def _generate_latex_comparison_section(results: dict, lang: str) -> str:
    """Generate LaTeX content for comparative analysis section."""
    import re as _re
    t = lambda key: get_text(key, lang)

    def _strip_md_heading(s):
        return _re.sub(r'^#{1,5}\s+', '', s)

    tex = ''

    # =========================================================================
    # Q1: Model Agreement
    # =========================================================================
    tex += r'\subsection{' + latex_escape(_strip_md_heading(t('q1_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q1_research')) + '\n\n'
    tex += markdown_to_latex(t('q1_method_intro')) + '\n\n'

    # ARI formula
    tex += r'\textbf{Adjusted Rand Index (ARI)}' + '\n\n'
    tex += r'''The ARI measures agreement between two clusterings, adjusted for chance:
\begin{equation}
\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \frac{\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}}{\binom{n}{2}}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \frac{\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}}{\binom{n}{2}}}
\end{equation}

where $n_{ij}$ is the number of objects in both cluster $i$ of the first clustering and cluster $j$ of the second, $a_i = \sum_j n_{ij}$ and $b_j = \sum_i n_{ij}$.

'''

    # NMI formula
    tex += r'\textbf{Normalized Mutual Information (NMI)}' + '\n\n'
    tex += r'''NMI quantifies the information shared between clusterings, normalized by entropy:
\begin{equation}
\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}
\end{equation}

where $I(U; V) = \sum_{i,j} P(i,j) \log \frac{P(i,j)}{P(i)P(j)}$ is the mutual information and $H(\cdot)$ is entropy.

'''

    # Agreement results table
    agreement = results.get('agreement', {})
    if agreement:
        tex += markdown_to_latex(t('q1_results')) + '\n\n'
        headers = [t('pair'), 'ARI', 'NMI', t('interpretation')]
        rows = []
        best_pair, best_nmi = None, -1
        worst_pair, worst_nmi = None, 2

        for pair_name, value in agreement.items():
            if isinstance(value, dict):
                if 'agreement' in value:
                    ari = value['agreement'].get('ari', 0)
                    nmi = value['agreement'].get('nmi', 0)
                else:
                    ari = value.get('ari', 0)
                    nmi = value.get('nmi', 0)
            else:
                ari = 0
                nmi = float(value) if value is not None else 0

            if nmi > best_nmi:
                best_nmi = nmi
                best_pair = pair_name
            if nmi < worst_nmi:
                worst_nmi = nmi
                worst_pair = pair_name

            if nmi > 0.5:
                interp = t('strong_agreement')
            elif nmi > 0.3:
                interp = t('moderate_agreement')
            elif nmi > 0.1:
                interp = t('weak_agreement')
            else:
                interp = t('near_random')

            rows.append([latex_escape(pair_name), f'{ari:.4f}', f'{nmi:.4f}', latex_escape(interp)])

        tex += generate_latex_table(headers, rows, caption='Model Agreement Metrics')

        # Key observations for Q1
        tex += markdown_to_latex(t('key_observations')) + '\n\n'
        tex += r'\begin{enumerate}' + '\n'
        tex += r'\item ' + markdown_to_latex(t('q1_obs1').format(pair=best_pair, nmi=best_nmi)) + '\n'
        tex += r'\item ' + markdown_to_latex(t('q1_obs2').format(pair=worst_pair, nmi=worst_nmi)) + '\n'
        tex += r'\item ' + markdown_to_latex(t('q1_obs3')) + '\n'
        tex += r'\end{enumerate}' + '\n\n'

    # =========================================================================
    # Q2: Artist Separation
    # =========================================================================
    tex += r'\subsection{' + latex_escape(_strip_md_heading(t('q2_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q2_research')) + '\n\n'
    tex += markdown_to_latex(t('q2_method_intro')) + '\n\n'

    # Cramér's V formula
    tex += r'''\textbf{Cram\'er's V}

Measures the strength of association between two categorical variables:
\begin{equation}
V = \sqrt{\frac{\chi^2 / n}{\min(k-1, r-1)}}
\end{equation}

where $\chi^2$ is the chi-squared statistic, $n$ is sample size, $k$ and $r$ are the number of categories.

'''

    artist_sep = results.get('artist_separation', {})
    if artist_sep:
        tex += markdown_to_latex(t('q2_results')) + '\n\n'
        headers = [t('model'), "Cramér's V", t('interpretation')]
        rows = []
        best_model, best_v = None, -1

        for model in ['bertopic', 'lda', 'iramuteq']:
            v = artist_sep.get(f'{model}_cramers_v')
            if v is None:
                model_data = artist_sep.get(model, {})
                if isinstance(model_data, dict):
                    v = model_data.get('cramers_v', 0)
                else:
                    v = 0
            v = float(v) if v is not None else 0

            if v > best_v:
                best_v = v
                best_model = model

            if v > 0.3:
                interp = t('strong_association')
            elif v > 0.2:
                interp = t('moderate_association')
            elif v > 0.1:
                interp = t('weak_association')
            else:
                interp = t('very_weak')

            rows.append([model.upper(), f'{v:.4f}', latex_escape(interp)])

        tex += generate_latex_table(headers, rows, caption='Artist-Topic Association')

        # Key observations for Q2
        if best_model:
            tex += markdown_to_latex(t('key_observations')) + '\n\n'
            tex += r'\begin{enumerate}' + '\n'
            tex += r'\item ' + markdown_to_latex(t('q2_obs1').format(model=best_model.upper(), v=best_v)) + '\n'
            tex += r'\item ' + markdown_to_latex(t('q2_obs2')) + '\n'
            tex += r'\item ' + markdown_to_latex(t('q2_obs3')) + '\n'
            tex += r'\end{enumerate}' + '\n\n'

    # =========================================================================
    # Q3: Temporal Dynamics
    # =========================================================================
    tex += r'\subsection{' + latex_escape(_strip_md_heading(t('q3_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q3_research')) + '\n\n'
    tex += markdown_to_latex(t('q3_method_intro')) + '\n\n'
    tex += markdown_to_latex(t('temporal_variance_desc')) + '\n\n'
    tex += markdown_to_latex(t('q3_results')) + '\n\n'

    temporal = results.get('temporal', {})
    if temporal:
        headers = [t('model'), t('temporal_variance'), t('most_variable_topic'),
                   t('max_variance'), t('interpretation')]
        rows = []
        best_model, best_var = None, -1

        for model in ['bertopic', 'lda', 'iramuteq']:
            mean_var = temporal.get(f'{model}_mean_variance', 0)
            most_var_topic = temporal.get(f'{model}_most_variable_topic', '-')
            max_var = temporal.get(f'{model}_max_variance', 0)

            if mean_var > best_var:
                best_var = mean_var
                best_model = model

            if mean_var > 0.01:
                interp = t('high_dynamics')
            elif mean_var > 0.001:
                interp = t('moderate_dynamics')
            else:
                interp = t('stable')

            rows.append([model.upper(), f'{mean_var:.6f}',
                         latex_escape(str(most_var_topic)), f'{max_var:.6f}',
                         latex_escape(interp)])

        tex += generate_latex_table(headers, rows, caption='Temporal Dynamics')

        # Key observations for Q3
        if best_model:
            tex += markdown_to_latex(t('key_observations')) + '\n\n'
            tex += r'\begin{enumerate}' + '\n'
            tex += r'\item ' + markdown_to_latex(t('q3_obs1').format(model=best_model.upper())) + '\n'
            tex += r'\item ' + markdown_to_latex(t('q3_obs2')) + '\n'
            tex += r'\item ' + markdown_to_latex(t('q3_obs3')) + '\n'
            tex += r'\end{enumerate}' + '\n\n'

    # =========================================================================
    # Q4: Vocabulary Distinctiveness
    # =========================================================================
    tex += r'\subsection{' + latex_escape(_strip_md_heading(t('q4_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q4_research')) + '\n\n'
    tex += markdown_to_latex(t('q4_method_intro')) + '\n\n'
    tex += markdown_to_latex(t('jaccard_desc')) + '\n\n'
    tex += markdown_to_latex(t('distinctiveness_desc')) + '\n\n'
    tex += markdown_to_latex(t('q4_results')) + '\n\n'

    vocabulary = results.get('vocabulary', {})
    if vocabulary:
        headers = [t('model'), t('mean_jaccard_distance'), t('interpretation')]
        rows = []
        for model in ['bertopic', 'lda', 'iramuteq']:
            dist = vocabulary.get(f'{model}_distinctiveness', 0)

            if dist > 0.9:
                interp = t('highly_distinct')
            elif dist > 0.7:
                interp = t('distinct')
            elif dist > 0.5:
                interp = t('moderate_overlap')
            else:
                interp = t('significant_overlap')

            rows.append([model.upper(), f'{dist:.4f}', latex_escape(interp)])

        tex += generate_latex_table(headers, rows, caption='Vocabulary Distinctiveness')

        # Key observations for Q4
        tex += markdown_to_latex(t('key_observations')) + '\n\n'
        tex += r'\begin{enumerate}' + '\n'
        tex += r'\item ' + markdown_to_latex(t('q4_obs1')) + '\n'
        tex += r'\item ' + markdown_to_latex(t('q4_obs2')) + '\n'
        tex += r'\end{enumerate}' + '\n\n'

        # Cross-model full vocabulary Jaccard at multiple thresholds
        cross_model_jaccard = results.get('cross_model_jaccard', {})
        cross_bl = cross_model_jaccard.get('bertopic_vs_lda', {})
        per_threshold = cross_bl.get('per_threshold', {})
        if per_threshold:
            tex += r'\subsubsection{' + latex_escape(_strip_md_heading(t('cross_model_full_jaccard_title'))) + r'}' + '\n\n'
            tex += markdown_to_latex(t('cross_model_full_jaccard_intro')) + '\n\n'
            if lang == 'fr':
                caption_text = 'Jaccard vocabulaire complet (BERTopic vs LDA par seuil de fréquence)'
                h = ['Seuil min.', 'Jaccard moyen', 'Paires']
            else:
                caption_text = 'Full vocabulary Jaccard (BERTopic vs LDA by frequency threshold)'
                h = ['Min freq.', 'Mean Jaccard', 'Pairs']
            r_rows = []
            for thresh in sorted(per_threshold.keys()):
                data = per_threshold[thresh]
                r_rows.append([str(thresh), f"{data['mean_jaccard']:.4f}", str(data['n_pairs'])])
            tex += generate_latex_table(h, r_rows, caption=caption_text)

    return tex


def _generate_latex_distance_section(distance_results: dict, lang: str) -> str:
    """Generate LaTeX content for intra-topic distance analysis (Q5)."""
    import re as _re
    t = lambda key: get_text(key, lang)

    def _strip_md_heading(s):
        return _re.sub(r'^#{1,5}\s+', '', s)

    tex = r'\subsection{' + latex_escape(_strip_md_heading(t('q5_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q5_research')) + '\n\n'
    tex += markdown_to_latex(t('q5_intro')) + '\n\n'

    # Jensen-Shannon formula
    tex += r'''\textbf{Jensen-Shannon Divergence}

Measures the distributional divergence between topic word distributions:
\begin{equation}
D_{\text{JS}}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)
\end{equation}

where $M = \frac{1}{2}(P + Q)$ is the mean distribution and $D_{\text{KL}}$ is the Kullback-Leibler divergence:
\begin{equation}
D_{\text{KL}}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
\end{equation}

'''

    # Labbé distance formula
    tex += r'''\textbf{Labb\'e Distance}

Measures lexical homogeneity using L1 (Manhattan) distance on word frequencies:
\begin{equation}
D_{\text{Labb\'e}}(A, B) = \frac{1}{2} \sum_{i=1}^{V} |f_i(A) - f_i(B)|
\end{equation}

where $f_i(X)$ is the relative frequency of word $i$ in text $X$, and $V$ is the vocabulary size.

'''

    # Results tables
    tex += markdown_to_latex(t('q5_results')) + '\n\n'

    # Jensen-Shannon table
    tex += r'\textbf{Jensen-Shannon Distances}' + '\n\n'
    headers = [t('model'), t('mean_distance'), t('std_dev'), t('n_topics'), t('interpretation')]
    rows = []
    for model in ['bertopic', 'lda', 'iramuteq']:
        js_results = distance_results.get(f'{model}_js', {})
        mean_dist = js_results.get('mean', 0)
        std_dist = js_results.get('std', 0)
        n_topics = js_results.get('n_topics', 0)

        if mean_dist < 0.3:
            interp = t('high_homogeneity')
        elif mean_dist < 0.5:
            interp = t('moderate_homogeneity')
        elif mean_dist < 0.7:
            interp = t('low_homogeneity')
        else:
            interp = t('very_heterogeneous')

        rows.append([model.upper(), f'{mean_dist:.4f}', f'{std_dist:.4f}', str(n_topics), latex_escape(interp)])

    tex += generate_latex_table(headers, rows, caption='Jensen-Shannon Intra-topic Distances')

    # Labbé table
    tex += r'\textbf{Labb\'e Distances}' + '\n\n'
    rows = []
    for model in ['bertopic', 'lda', 'iramuteq']:
        labbe_results = distance_results.get(f'{model}_labbe', {})
        mean_dist = labbe_results.get('mean', 0)
        std_dist = labbe_results.get('std', 0)
        n_topics = labbe_results.get('n_topics', 0)

        if mean_dist < 0.4:
            interp = t('high_homogeneity')
        elif mean_dist < 0.6:
            interp = t('moderate_homogeneity')
        elif mean_dist < 0.8:
            interp = t('low_homogeneity')
        else:
            interp = t('very_heterogeneous')

        rows.append([model.upper(), f'{mean_dist:.4f}', f'{std_dist:.4f}', str(n_topics), latex_escape(interp)])

    tex += generate_latex_table(headers, rows, caption='Labbé Intra-topic Distances')

    return tex


def _generate_latex_distance_4configs_section(
    topic_distance_results: dict,
    aggregation_size: int,
    lang: str
) -> str:
    """Generate LaTeX section for all 4 distance configurations (Q5 extended)."""
    import re as _re
    t = lambda key: get_text(key, lang)

    def _strip_md_heading(s):
        return _re.sub(r'^#{1,5}\s+', '', s)

    tex = r'\subsubsection{' + latex_escape(_strip_md_heading(t('q5_4configs_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q5_4configs_intro')) + '\n\n'

    # Configuration explanation table
    headers = [t('config_table_header'), t('what_it_measures'), t('interpretation_guide')]
    rows = [
        [r'\textbf{' + latex_escape(t('config_intra_all_paired')) + r'}',
         latex_escape(t('homogeneity')), latex_escape(t('lower_better'))],
        [r'\textbf{' + latex_escape(t('config_inter_all_paired')) + r'}',
         latex_escape(t('separation')), latex_escape(t('higher_better'))],
        [r'\textbf{' + latex_escape(t('config_intra_aggregated')) + r'}' + f' (n={aggregation_size})',
         latex_escape(t('homogeneity')), latex_escape(t('lower_better'))],
        [r'\textbf{' + latex_escape(t('config_inter_aggregated')) + r'}' + f' (n={aggregation_size})',
         latex_escape(t('separation')), latex_escape(t('higher_better'))],
    ]
    tex += generate_latex_table(headers, rows, caption='Distance Configurations')
    tex += markdown_to_latex(t('aggregation_note').format(n=aggregation_size)) + '\n\n'

    # Build config list
    configs = [
        ('intra_all_paired', t('config_intra_all_paired'),
         t('config_intra_all_paired_desc'), 'homogeneity'),
        ('inter_all_paired', t('config_inter_all_paired'),
         t('config_inter_all_paired_desc'), 'separation'),
        (f'intra_aggregated_{aggregation_size}',
         t('config_intra_aggregated') + f' (n={aggregation_size})',
         t('config_intra_aggregated_desc'), 'homogeneity'),
        (f'inter_aggregated_{aggregation_size}',
         t('config_inter_aggregated') + f' (n={aggregation_size})',
         t('config_inter_aggregated_desc'), 'separation'),
    ]

    # Per-config results tables
    for config_key, config_label, config_desc, config_type in configs:
        tex += r'\textbf{' + latex_escape(config_label) + r'}' + '\n\n'
        tex += r'\textit{' + markdown_to_latex(config_desc).replace(r'\textbf{', r'\textbf{') + r'}' + '\n\n'

        headers = [t('model'), 'JS', r"Labb\'e"]
        rows = []
        for model in ['bertopic', 'lda', 'iramuteq']:
            model_results = topic_distance_results.get(model, {})
            config_results = model_results.get(config_key, {})
            js_mean = config_results.get('js', {}).get('mean', 0)
            labbe_mean = config_results.get('labbe', {}).get('mean', 0)
            rows.append([model.upper(), f'{js_mean:.4f}', f'{labbe_mean:.4f}'])

        tex += generate_latex_table(headers, rows, caption=config_label)

    # Summary: best models
    tex += markdown_to_latex(t('q5_summary_4configs')) + '\n\n'

    best_homo_js = best_homo_labbe = best_sep_js = best_sep_labbe = None
    best_homo_js_val = best_homo_labbe_val = float('inf')
    best_sep_js_val = best_sep_labbe_val = 0

    for model in ['bertopic', 'lda', 'iramuteq']:
        model_results = topic_distance_results.get(model, {})
        intra = model_results.get('intra_all_paired', {})
        inter = model_results.get('inter_all_paired', {})

        js_intra = intra.get('js', {}).get('mean', float('inf'))
        labbe_intra = intra.get('labbe', {}).get('mean', float('inf'))
        js_inter = inter.get('js', {}).get('mean', 0)
        labbe_inter = inter.get('labbe', {}).get('mean', 0)

        if js_intra < best_homo_js_val:
            best_homo_js_val = js_intra
            best_homo_js = model
        if labbe_intra < best_homo_labbe_val:
            best_homo_labbe_val = labbe_intra
            best_homo_labbe = model
        if js_inter > best_sep_js_val:
            best_sep_js_val = js_inter
            best_sep_js = model
        if labbe_inter > best_sep_labbe_val:
            best_sep_labbe_val = labbe_inter
            best_sep_labbe = model

    tex += r'\begin{itemize}' + '\n'
    if best_homo_js:
        tex += r'\item \textbf{' + latex_escape(t('q5_best_homogeneity')) + r' (JS):} '
        tex += f'{best_homo_js.upper()} ({best_homo_js_val:.4f})\n'
    if best_homo_labbe:
        tex += r"\item \textbf{" + latex_escape(t('q5_best_homogeneity')) + r" (Labb\'e):} "
        tex += f'{best_homo_labbe.upper()} ({best_homo_labbe_val:.4f})\n'
    if best_sep_js:
        tex += r'\item \textbf{' + latex_escape(t('q5_best_separation')) + r' (JS):} '
        tex += f'{best_sep_js.upper()} ({best_sep_js_val:.4f})\n'
    if best_sep_labbe:
        tex += r"\item \textbf{" + latex_escape(t('q5_best_separation')) + r" (Labb\'e):} "
        tex += f'{best_sep_labbe.upper()} ({best_sep_labbe_val:.4f})\n'
    tex += r'\end{itemize}' + '\n\n'

    return tex


def _generate_latex_aggregation_curve_section(
    multi_agg_results: dict,
    agg_metadata: dict,
    figures_dir: str,
    lang: str
) -> str:
    """Generate LaTeX section for aggregation stabilization curve."""
    import re as _re
    t = lambda key: get_text(key, lang)

    def _strip_md(s):
        return _re.sub(r'^#{1,5}\s+', '', s).replace('**', '').replace('*', '')

    tex = r'\subsubsection{' + latex_escape(_strip_md(t('q5_agg_curve_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q5_agg_curve_intro')) + '\n\n'

    if agg_metadata:
        range_text = t('q5_agg_curve_range').format(
            min_agg=agg_metadata.get('agg_min', '?'),
            max_agg=agg_metadata.get('agg_max', '?'),
            min_words=agg_metadata.get('min_words_per_unit', 1000),
            min_units=agg_metadata.get('min_units_per_topic', 10),
            min_topic_size=agg_metadata.get('min_topic_size', '?'),
            n_points=agg_metadata.get('n_points', '?'),
        )
        tex += markdown_to_latex(range_text) + '\n\n'

    # Figure
    fig_path = Path(figures_dir) / 'aggregation_curve.png'
    if fig_path.exists():
        tex += generate_latex_figure(
            str(fig_path),
            _strip_md(t('q5_agg_curve_fig_caption')),
            width=0.95
        )

    return tex


def _generate_latex_inter_topic_ranking_section(
    centroid_results: dict,
    figures_dir: str,
    topic_labels_per_model: dict,
    lang: str
) -> str:
    """Generate LaTeX section for inter-topic separation ranking (centroid distances)."""
    import re as _re
    t = lambda key: get_text(key, lang)

    def _strip_md(s):
        return _re.sub(r'^#{1,5}\s+', '', s).replace('**', '').replace('*', '')

    tex = r'\subsubsection{' + latex_escape(_strip_md(t('q5_inter_ranking_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q5_inter_ranking_intro')) + '\n\n'

    for model in ['bertopic', 'lda', 'iramuteq']:
        centroid_data = centroid_results.get(model, {})
        labbe_per_topic = centroid_data.get('labbe', {}).get('per_topic', {})

        if not labbe_per_topic:
            continue

        js_per_topic = centroid_data.get('js', {}).get('per_topic', {})
        labels = topic_labels_per_model.get(model, {})

        def _label(tid):
            return labels.get(str(tid), labels.get(tid, f"Topic {tid}"))

        # Figure
        fig_path = Path(figures_dir) / f'inter_topic_ranking_{model}.png'
        if fig_path.exists():
            caption = _strip_md(t('q5_inter_ranking_fig_caption').format(model=model.upper()))
            tex += generate_latex_figure(str(fig_path), caption, width=0.95)

        # Summary table: top 3 most/least distinct
        sorted_topics = sorted(
            labbe_per_topic.items(),
            key=lambda x: x[1].get('mean_distance', 0),
            reverse=True
        )
        n_show = min(3, len(sorted_topics))

        # Most distinct
        tex += markdown_to_latex(t('q5_inter_ranking_most_distinct')) + '\n\n'
        headers = ['Topic', latex_escape(_strip_md(t('mean_labbe_distance'))),
                    latex_escape(_strip_md(t('mean_js_inter_distance')))]
        rows = []
        for tid, stats in sorted_topics[:n_show]:
            labbe_val = stats.get('mean_distance', 0)
            js_val = js_per_topic.get(tid, {}).get('mean_distance', 0)
            rows.append([latex_escape(_label(tid)), f'{labbe_val:.4f}', f'{js_val:.4f}'])
        tex += generate_latex_table(headers, rows, col_widths=['5cm', None, None])

        # Least distinct
        tex += markdown_to_latex(t('q5_inter_ranking_least_distinct')) + '\n\n'
        rows = []
        for tid, stats in sorted_topics[-n_show:]:
            labbe_val = stats.get('mean_distance', 0)
            js_val = js_per_topic.get(tid, {}).get('mean_distance', 0)
            rows.append([latex_escape(_label(tid)), f'{labbe_val:.4f}', f'{js_val:.4f}'])
        tex += generate_latex_table(headers, rows, col_widths=['5cm', None, None])

    return tex


def _generate_latex_chi2_section(chi2_results: dict, topic_labels_per_model: dict,
                                 lang: str) -> str:
    """Generate LaTeX section for χ²/n word × topic independence test."""
    import re as _re
    t = lambda key: get_text(key, lang)

    def _strip_md(s):
        return _re.sub(r'^#{1,5}\s+', '', s).replace('**', '').replace('*', '')

    tex = r'\subsubsection{' + latex_escape(_strip_md(t('q5_chi2_title'))) + r'}' + '\n\n'
    tex += markdown_to_latex(t('q5_chi2_intro')) + '\n\n'
    tex += markdown_to_latex(t('q5_chi2_explanation')) + '\n\n'

    for variant_key in ['non_lemmatized', 'lemmatized']:
        variant_data = chi2_results.get(variant_key, {})
        if not variant_data:
            continue

        tex += r'\textbf{' + latex_escape(t(variant_key)) + r'}' + '\n\n'

        # Global table
        headers = ['Model', r'$\chi^2$', latex_escape(t('total_tokens_label')),
                    r'$\chi^2/n$', latex_escape(t('vocab_size_label'))]
        rows = []
        for model in ['bertopic', 'lda', 'iramuteq']:
            r = variant_data.get(model, {})
            if not r:
                continue
            rows.append([
                model.upper(),
                f"{r.get('chi2', 0):,.0f}",
                f"{r.get('n', 0):,}",
                f"{r.get('chi2_over_n', 0):.4f}",
                f"{r.get('vocab_size', 0):,}",
            ])
        tex += generate_latex_table(headers, rows)

    return tex


def _generate_latex_distance_appendix(lang: str) -> str:
    """Generate LaTeX appendix with distance metric comparisons."""
    t = lambda key: get_text(key, lang)

    if lang == 'fr':
        tex = r'''
\subsection{Labb\'e vs Jensen-Shannon : deux regards sur les fr\'equences}

\textbf{Point commun}

Les deux m\'etriques partent de la m\^eme repr\'esentation : chaque texte est une distribution de probabilit\'e sur le vocabulaire.

\begin{equation}
P_A = (f_1(A), f_2(A), \ldots, f_V(A)) \quad \text{o\`u} \quad \sum_{i=1}^{V} f_i(A) = 1
\end{equation}

\subsubsection{Distance de Labb\'e}

Mesure la diff\'erence absolue des fr\'equences :

\begin{equation}
D_{\text{Labb\'e}}(A, B) = \frac{1}{2} \sum_{i=1}^{V} |f_i(A) - f_i(B)|
\end{equation}

C'est une \textbf{distance L1 (Manhattan) normalis\'ee}.

\subsubsection{Divergence de Jensen-Shannon}

Mesure la divergence informationnelle entre les distributions :

\begin{equation}
D_{\text{JS}}(A, B) = \frac{1}{2} D_{\text{KL}}(P_A \| M) + \frac{1}{2} D_{\text{KL}}(P_B \| M)
\end{equation}

o\`u $M = (P_A + P_B) / 2$ est la distribution moyenne.

\subsubsection{Diff\'erence fondamentale}

\begin{table}[H]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Aspect} & \textbf{Labb\'e} & \textbf{Jensen-Shannon} \\
\hline
Sensibilit\'e & Lin\'eaire & Logarithmique \\
Mots rares & Faible impact & \textbf{Fort impact} \\
Fondement & G\'eom\'etrique (L1) & Th\'eorie de l'information \\
\hline
\end{tabular}
\caption{Comparaison des m\'etriques de distance}
\end{table}

\subsubsection{Application au rap fran\c{c}ais}

\textbf{JS est plus sensible aux mots d'argot sp\'ecifiques} \`a certains artistes/th\`emes.

\textbf{Labb\'e capture mieux l'homog\'en\'eit\'e globale} du vocabulaire courant.

\textbf{Recommandation :} Utiliser les deux m\'etriques en compl\'ement :
\begin{itemize}
\item \textbf{Labb\'e} pour l'homog\'en\'eit\'e lexicale g\'en\'erale
\item \textbf{JS} pour d\'etecter les vocabulaires distinctifs
\end{itemize}

'''
    else:
        tex = r'''
\subsection{Labb\'e vs Jensen-Shannon: Two Perspectives on Frequencies}

\textbf{Common Ground}

Both metrics start from the same representation: each text is a probability distribution over vocabulary.

\begin{equation}
P_A = (f_1(A), f_2(A), \ldots, f_V(A)) \quad \text{where} \quad \sum_{i=1}^{V} f_i(A) = 1
\end{equation}

\subsubsection{Labb\'e Distance}

Measures the absolute difference in frequencies:

\begin{equation}
D_{\text{Labb\'e}}(A, B) = \frac{1}{2} \sum_{i=1}^{V} |f_i(A) - f_i(B)|
\end{equation}

This is a \textbf{normalized L1 (Manhattan) distance}.

\subsubsection{Jensen-Shannon Divergence}

Measures the informational divergence between distributions:

\begin{equation}
D_{\text{JS}}(A, B) = \frac{1}{2} D_{\text{KL}}(P_A \| M) + \frac{1}{2} D_{\text{KL}}(P_B \| M)
\end{equation}

where $M = (P_A + P_B) / 2$ is the mean distribution.

\subsubsection{Fundamental Difference}

\begin{table}[H]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Aspect} & \textbf{Labb\'e} & \textbf{Jensen-Shannon} \\
\hline
Sensitivity & Linear & Logarithmic \\
Rare words & Low impact & \textbf{High impact} \\
Foundation & Geometric (L1) & Information theory \\
\hline
\end{tabular}
\caption{Distance metrics comparison}
\end{table}

\subsubsection{Application to French Rap}

\textbf{JS is more sensitive to slang words specific} to certain artists/themes.

\textbf{Labb\'e better captures the global homogeneity} of common vocabulary.

\textbf{Recommendation:} Use both metrics as complements:
\begin{itemize}
\item \textbf{Labb\'e} for general lexical homogeneity
\item \textbf{JS} to detect distinctive vocabularies
\end{itemize}

'''
    return tex
