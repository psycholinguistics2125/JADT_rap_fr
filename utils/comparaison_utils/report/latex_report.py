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
    tex += latex_escape(t('abstract_text')) + '\n\n'
    tex += r'\newpage' + '\n\n'

    # Section 1: Corpus Description
    tex += r'\section{' + latex_escape(t('corpus_description')) + r'}' + '\n\n'
    tex += _generate_latex_corpus_section(results, figures_dir_path, lang)

    # Section 2: Individual Models
    tex += r'\section{' + latex_escape(t('individual_models')) + r'}' + '\n\n'
    tex += latex_escape(t('individual_models_intro')) + '\n\n'

    tex += r'\subsection{BERTopic}' + '\n'
    tex += _generate_latex_model_section(results['bertopic'], 'bertopic', figures_dir_path, lang)

    tex += r'\subsection{LDA}' + '\n'
    tex += _generate_latex_model_section(results['lda'], 'lda', figures_dir_path, lang)

    tex += r'\subsection{IRAMUTEQ}' + '\n'
    tex += _generate_latex_model_section(results['iramuteq'], 'iramuteq', figures_dir_path, lang)

    # Section 3: Comparative Analysis
    tex += r'\section{' + latex_escape(t('comparative_analysis')) + r'}' + '\n\n'
    tex += _generate_latex_comparison_section(results, lang)

    # Section 4: Intra-topic Distance (Q5)
    distance_results = results.get('intra_topic_distances', {})
    if distance_results:
        tex += _generate_latex_distance_section(distance_results, lang)

    # Section 5: Summary
    tex += r'\section{' + latex_escape(t('summary_title').replace('## ', '')) + r'}' + '\n\n'
    tex += latex_escape(t('key_findings')) + '\n\n'
    tex += latex_escape(t('interpretation_section')) + '\n\n'
    tex += latex_escape(t('interpretation_1')) + '\n\n'
    tex += latex_escape(t('interpretation_2')) + '\n\n'
    tex += latex_escape(t('recommendations')) + '\n\n'
    tex += latex_escape(t('rec_semantic')) + '\n'
    tex += latex_escape(t('rec_lexical')) + '\n\n'

    # References (full citations: authors, title, journal)
    def latex_full_ref(key):
        ref = METRIC_REFERENCES[key]
        return latex_escape(f"{ref['citation']}. {ref['paper']}")

    tex += r'\section{' + latex_escape(t('references_title').replace('## ', '')) + r'}' + '\n\n'
    tex += r'\subsection*{' + latex_escape(t('clustering_agreement_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('ari') + '\n'
    tex += r'\item ' + latex_full_ref('nmi') + '\n'
    tex += r'\item ' + latex_full_ref('ami') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(t('association_measures_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('cramers_v') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(t('info_theory_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('js_divergence') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(t('intertextual_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('labbe_distance') + '\n'
    for extra in METRIC_REFERENCES['labbe_distance'].get('additional_refs', []):
        tex += r'\item ' + latex_escape(extra) + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(t('topic_coherence_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('coherence_cv') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(t('cluster_validation_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_full_ref('silhouette') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    tex += r'\subsection*{' + latex_escape(t('topic_modeling_refs')) + r'}' + '\n'
    tex += r'\begin{itemize}' + '\n'
    tex += r'\item ' + latex_escape('Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.') + '\n'
    tex += r'\item ' + latex_escape('Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.') + '\n'
    tex += r'\item ' + latex_escape('Reinert, M. (1983). Une méthode de classification descendante hiérarchique : application à l\'analyse lexicale par contexte. Les Cahiers de l\'Analyse des Données, 8(2), 187-198.') + '\n'
    tex += r'\end{itemize}' + '\n\n'

    # Appendix with distance formulas
    tex += r'\appendix' + '\n'
    tex += r'\section{' + latex_escape(t('appendix_distance_title')) + r'}' + '\n\n'
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

    tex = latex_escape(t('corpus_intro')) + '\n\n'

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
        tex += generate_latex_figure(str(corpus_fig), caption=t('fig_year_dist'), label='fig:year_dist')

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
        tex += latex_escape(t('bertopic_desc')) + '\n\n'
    elif model_type == 'lda':
        tex += latex_escape(t('lda_desc')) + '\n\n'
    elif model_type == 'iramuteq':
        tex += latex_escape(t('iramuteq_desc')) + '\n\n'

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
            tex += generate_latex_figure(str(fig_path), caption=t(fig_caption_key))

    tex += r'\newpage' + '\n\n'
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
    t = lambda key: get_text(key, lang)
    tex = ''

    # Q1: Model Agreement
    tex += r'\subsection{' + latex_escape(t('q1_title').replace('### ', '')) + r'}' + '\n\n'
    tex += latex_escape(t('q1_research')) + '\n\n'
    tex += latex_escape(t('q1_method_intro')) + '\n\n'

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
        tex += latex_escape(t('q1_results')) + '\n\n'
        headers = [t('pair'), 'ARI', 'NMI', t('interpretation')]
        rows = []
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

    # Q2: Artist Separation
    tex += r'\subsection{' + latex_escape(t('q2_title').replace('### ', '')) + r'}' + '\n\n'
    tex += latex_escape(t('q2_research')) + '\n\n'

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
        tex += latex_escape(t('q2_results')) + '\n\n'
        headers = [t('model'), r"Cram\'er's V", t('interpretation')]
        rows = []
        for model in ['bertopic', 'lda', 'iramuteq']:
            v = artist_sep.get(f'{model}_cramers_v')
            if v is None:
                model_data = artist_sep.get(model, {})
                if isinstance(model_data, dict):
                    v = model_data.get('cramers_v', 0)
                else:
                    v = 0
            v = float(v) if v is not None else 0

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

    # Q3: Temporal Dynamics
    tex += r'\subsection{' + latex_escape(t('q3_title').replace('### ', '')) + r'}' + '\n\n'
    tex += latex_escape(t('q3_research')) + '\n\n'
    tex += latex_escape(t('q3_method_intro')) + '\n\n'

    # Q4: Vocabulary
    tex += r'\subsection{' + latex_escape(t('q4_title').replace('### ', '')) + r'}' + '\n\n'
    tex += latex_escape(t('q4_research')) + '\n\n'
    tex += latex_escape(t('q4_method_intro')) + '\n\n'

    return tex


def _generate_latex_distance_section(distance_results: dict, lang: str) -> str:
    """Generate LaTeX content for intra-topic distance analysis (Q5)."""
    t = lambda key: get_text(key, lang)

    tex = r'\subsection{' + latex_escape(t('q5_title').replace('### ', '')) + r'}' + '\n\n'
    tex += latex_escape(t('q5_research')) + '\n\n'
    tex += latex_escape(t('q5_intro')) + '\n\n'

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
    tex += latex_escape(t('q5_results')) + '\n\n'

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
