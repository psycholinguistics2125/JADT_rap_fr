#!/usr/bin/env python3
"""
Markdown comparison report generator.
"""

from datetime import datetime
from pathlib import Path

from ..constants import METRIC_REFERENCES

from .translations import get_text, get_metric_description
from .sections import (
    generate_corpus_description,
    generate_run_description,
    generate_intra_topic_distance_section,
    generate_distance_appendix,
)


def generate_comparison_report(results: dict, output_dir: str, figures_dir: str = None,
                                lang: str = 'fr') -> str:
    """
    Generate the full markdown comparison report.

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
        Full markdown report content.
    """
    t = lambda key: get_text(key, lang)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    figures_dir_path = Path(output_dir) / 'figures' if not figures_dir else Path(figures_dir)

    md = f"""# {t('title')}

**{t('generated')}:** {timestamp}

**{t('output_dir')}:** `{output_dir}`

---

## {t('abstract_title')}

{t('abstract_text')}

---

"""

    # Section 1: Corpus description
    sample_df = results['bertopic']['doc_assignments']
    md += generate_corpus_description(sample_df, str(figures_dir_path), lang=lang)
    md += "\n---\n\n"

    # Section 2: Individual model descriptions
    md += f"""## {t('individual_models')}

{t('individual_models_intro')}

"""

    md += generate_run_description(
        results['bertopic'], "2.1 BERTopic", "bertopic",
        figures_dir=str(figures_dir_path), comparison_figures_dir=str(figures_dir_path),
        lang=lang
    )
    md += "\n"
    md += generate_run_description(
        results['lda'], "2.2 LDA", "lda",
        figures_dir=str(figures_dir_path), comparison_figures_dir=str(figures_dir_path),
        lang=lang
    )
    md += "\n"
    md += generate_run_description(
        results['iramuteq'], "2.3 IRAMUTEQ", "iramuteq",
        figures_dir=str(figures_dir_path), comparison_figures_dir=str(figures_dir_path),
        lang=lang
    )
    md += "\n---\n\n"

    # Section 3: Comparative analysis
    md += f"""## {t('comparative_analysis')}

"""

    # Q1: Model Agreement
    md += f"""{t('q1_title')}

{t('q1_research')}

{t('q1_method_intro')}

**Adjusted Rand Index (ARI)** — {METRIC_REFERENCES['ari']['citation']}

{get_metric_description('ari', lang)}

**Normalized Mutual Information (NMI)** — {METRIC_REFERENCES['nmi']['citation']}

{get_metric_description('nmi', lang)}

{t('q1_results')}

"""

    # Add agreement metrics if available
    agreement = results.get('agreement', {})
    if agreement:
        md += f"| {t('pair')} | ARI | NMI | {t('interpretation')} |\n"
        md += "|------|-----|-----|----------------|\n"

        best_pair, best_nmi = None, -1
        worst_pair, worst_nmi = None, 2

        for pair_name, value in agreement.items():
            # Handle nested structure {'agreement': {'ari': x, 'nmi': y}, 'contingency': {...}}
            if isinstance(value, dict):
                if 'agreement' in value:
                    # Nested structure
                    ari = value['agreement'].get('ari', 0)
                    nmi = value['agreement'].get('nmi', 0)
                else:
                    # Direct structure {'ari': x, 'nmi': y}
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

            md += f"| {pair_name} | {ari:.4f} | {nmi:.4f} | {interp} |\n"

        md += f"""
{t('key_observations')}

1. {t('q1_obs1').format(pair=best_pair, nmi=best_nmi)}

2. {t('q1_obs2').format(pair=worst_pair, nmi=worst_nmi)}

3. {t('q1_obs3')}

"""

    # Q2, Q3, Q4 sections follow similar pattern...
    md += f"""{t('q2_title')}

{t('q2_research')}

{t('q2_method_intro')}

**{t('cramers_v')}** — {METRIC_REFERENCES['cramers_v']['citation']}

{get_metric_description('cramers_v', lang)}

{t('q2_results')}

"""

    # Artist separation results
    artist_sep = results.get('artist_separation', {})
    if artist_sep:
        md += f"| {t('model')} | {t('cramers_v')} | {t('interpretation')} |\n"
        md += "|-------|----------|----------------|\n"

        best_model, best_v = None, -1

        # Handle the flat dict structure with keys like 'bertopic_cramers_v'
        for model in ['bertopic', 'lda', 'iramuteq']:
            # Try different key formats
            v = artist_sep.get(f'{model}_cramers_v')
            if v is None:
                # Try nested dict format
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

            md += f"| {model.upper()} | {v:.4f} | {interp} |\n"

        if best_model:
            md += f"""
{t('key_observations')}

1. {t('q2_obs1').format(model=best_model.upper(), v=best_v)}

2. {t('q2_obs2')}

3. {t('q2_obs3')}

"""

    # Q3: Temporal dynamics
    md += f"""{t('q3_title')}

{t('q3_research')}

{t('q3_method_intro')}

{t('temporal_variance_desc')}

{t('q3_results')}

"""

    temporal = results.get('temporal', {})
    if temporal:
        md += f"| {t('model')} | {t('temporal_variance')} | {t('most_variable_topic')} | {t('max_variance')} | {t('interpretation')} |\n"
        md += "|--------|-----|-----|-----|----------------|\n"

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

            md += f"| {model.upper()} | {mean_var:.6f} | {most_var_topic} | {max_var:.6f} | {interp} |\n"

        # Decade JS divergence table
        has_decade_js = any(f'{m}_decade_js' in temporal for m in ['bertopic', 'lda', 'iramuteq'])
        if has_decade_js:
            # Collect all decade transitions across models
            all_transitions = set()
            for model in ['bertopic', 'lda', 'iramuteq']:
                decade_js = temporal.get(f'{model}_decade_js', {})
                all_transitions.update(decade_js.keys())

            if all_transitions:
                sorted_transitions = sorted(all_transitions)
                md += f"\n**JS Distance entre décennies :**\n\n" if lang == 'fr' else f"\n**JS Distance between decades:**\n\n"
                md += f"| {'Transition' if lang == 'en' else 'Transition'} | BERTopic | LDA | IRAMUTEQ |\n"
                md += "|------------|----------|-----|----------|\n"
                for trans in sorted_transitions:
                    vals = []
                    for model in ['bertopic', 'lda', 'iramuteq']:
                        v = temporal.get(f'{model}_decade_js', {}).get(trans, None)
                        vals.append(f"{v:.4f}" if v is not None else '-')
                    md += f"| {trans} | {vals[0]} | {vals[1]} | {vals[2]} |\n"

        if best_model:
            md += f"""
{t('key_observations')}

1. {t('q3_obs1').format(model=best_model.upper())}

2. {t('q3_obs2')}

3. {t('q3_obs3')}

"""

    # Q4: Vocabulary
    md += f"""{t('q4_title')}

{t('q4_research')}

{t('q4_method_intro')}

{t('jaccard_desc')}

{t('distinctiveness_desc')}

{t('q4_results')}

"""

    vocabulary = results.get('vocabulary', {})
    if vocabulary:
        md += f"| {t('model')} | {t('mean_jaccard_distance')} | {t('interpretation')} |\n"
        md += "|--------|-----|----------------|\n"

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

            md += f"| {model.upper()} | {dist:.4f} | {interp} |\n"

        md += f"""
{t('key_observations')}

1. {t('q4_obs1')}

2. {t('q4_obs2')}

"""

        # Cross-model vocabulary comparison
        cross_vocab = vocabulary.get('bertopic_vs_lda', {})
        if cross_vocab:
            mean_jacc = cross_vocab.get('mean_jaccard', 0)
            mean_overlap = cross_vocab.get('mean_overlap_coef', 0)

            md += f"""{t('cross_model_vocab')}

{t('cross_vocab_intro')}

- **{t('mean_jaccard_sim')}** : {mean_jacc:.4f}
- **{t('mean_overlap_coef')}** : {mean_overlap:.4f}

{t('low_cross_vocab').format(pct=mean_jacc)}

"""
            # Per-topic comparison table
            comparisons = cross_vocab.get('topic_comparisons', [])
            if comparisons:
                md += f"| {t('bertopic_topic')} | {t('lda_topic')} | {t('jaccard')} | {t('common_words')} |\n"
                md += "|------|------|---------|------|\n"
                for comp in comparisons:
                    common = ', '.join(comp.get('common_words', [])[:5])
                    if not common:
                        common = t('none_common')
                    md += f"| {comp.get('bertopic_topic', '-')} | {comp.get('lda_topic', '-')} | {comp.get('jaccard', 0):.4f} | {common} |\n"
                md += "\n"

    # Q5: Intra-topic distance
    distance_results = results.get('intra_topic_distances', {})
    if distance_results:
        md += generate_intra_topic_distance_section(distance_results, lang=lang)

    # Section 4: Summary
    md += f"""
{t('summary_title')}

{t('key_findings')}

{t('interpretation_section')}

{t('interpretation_1')}

{t('interpretation_2')}

{t('recommendations')}

{t('rec_semantic')}
{t('rec_lexical')}

"""

    # Section 5: References (full citations with authors, title, journal)
    def full_ref(key):
        ref = METRIC_REFERENCES[key]
        return f"{ref['citation']}. {ref['paper']}"

    def full_refs_with_additional(key):
        lines = [f"- {full_ref(key)}"]
        for extra in METRIC_REFERENCES[key].get('additional_refs', []):
            lines.append(f"- {extra}")
        return '\n'.join(lines)

    md += f"""
{t('references_title')}

{t('clustering_agreement_refs')}

- {full_ref('ari')}
- {full_ref('nmi')}
- {full_ref('ami')}

{t('association_measures_refs')}

- {full_ref('cramers_v')}

{t('info_theory_refs')}

- {full_ref('js_divergence')}

{t('intertextual_refs')}

{full_refs_with_additional('labbe_distance')}

{t('topic_coherence_refs')}

- {full_ref('coherence_cv')}

{t('cluster_validation_refs')}

- {full_ref('silhouette')}

{t('topic_modeling_refs')}

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.
- Reinert, M. (1983). Une méthode de classification descendante hiérarchique : application à l'analyse lexicale par contexte. Les Cahiers de l'Analyse des Données, 8(2), 187-198.

"""

    # Appendix
    md += f"""
{t('appendix_title')}

{t('run_details')}

- **{t('comparison_timestamp')}:** {timestamp}
- **{t('bertopic_folder')}:** {results.get('bertopic', {}).get('run_dir', 'N/A')}
- **{t('lda_folder')}:** {results.get('lda', {}).get('run_dir', 'N/A')}
- **{t('iramuteq_folder')}:** {results.get('iramuteq', {}).get('run_dir', 'N/A')}

"""

    # Add distance appendix
    md += generate_distance_appendix(lang=lang)

    return md
