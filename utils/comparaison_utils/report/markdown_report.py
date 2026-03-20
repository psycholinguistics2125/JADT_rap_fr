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
    generate_topic_distance_4configs_section,
    generate_distance_appendix,
    generate_aggregation_curve_section,
    generate_inter_topic_ranking_section,
    generate_word_topic_chi2_section,
)


def _build_dynamic_conclusion(results: dict, lang: str) -> str:
    """Build conclusion section from actual computed results."""
    t = lambda key: get_text(key, lang)
    md = ""

    # Q1: Model agreement
    agreement = results.get('agreement', {})
    if agreement:
        pairs = []
        for pair_name, value in agreement.items():
            if isinstance(value, dict) and 'agreement' in value:
                nmi = value['agreement'].get('nmi', 0)
            elif isinstance(value, dict):
                nmi = value.get('nmi', 0)
            else:
                nmi = 0
            pairs.append((pair_name, nmi))
        if pairs:
            best = max(pairs, key=lambda x: x[1])
            worst = min(pairs, key=lambda x: x[1])
            if lang == 'fr':
                md += f"**Q1 — Accord entre modèles :** L'accord le plus fort est observé entre {best[0].replace('_', ' ')} (NMI = {best[1]:.4f}), tandis que {worst[0].replace('_', ' ')} montrent l'accord le plus faible (NMI = {worst[1]:.4f}). Ces valeurs modérées à faibles confirment que chaque modèle capture des aspects distincts du corpus.\n\n"
            else:
                md += f"**Q1 — Model agreement:** Strongest agreement between {best[0].replace('_', ' ')} (NMI = {best[1]:.4f}), weakest between {worst[0].replace('_', ' ')} (NMI = {worst[1]:.4f}). These moderate-to-low values confirm each model captures distinct aspects.\n\n"

    # Q2: Artist separation
    artist_sep = results.get('artist_separation', {})
    if artist_sep:
        best_model, best_v = None, -1
        for model in ['bertopic', 'lda', 'iramuteq']:
            v = artist_sep.get(f'{model}_cramers_v', 0)
            if v and v > best_v:
                best_v = v
                best_model = model
        if best_model:
            spec = artist_sep.get(f'{best_model}_pct_specialists', 0)
            if lang == 'fr':
                md += f"**Q2 — Séparation des artistes :** {best_model.upper()} capture le mieux les signatures artistiques (V de Cramér = {best_v:.4f}, {spec:.1f}% de spécialistes).\n\n"
            else:
                md += f"**Q2 — Artist separation:** {best_model.upper()} best captures artist signatures (Cramér's V = {best_v:.4f}, {spec:.1f}% specialists).\n\n"

    # Q3: Temporal dynamics
    temporal = results.get('temporal', {})
    if temporal:
        best_model, best_var = None, -1
        for model in ['bertopic', 'lda', 'iramuteq']:
            var = temporal.get(f'{model}_mean_variance', 0)
            if var > best_var:
                best_var = var
                best_model = model
        if best_model:
            if lang == 'fr':
                md += f"**Q3 — Dynamique temporelle :** {best_model.upper()} montre la variance temporelle la plus élevée ({best_var:.6f}), le rendant plus sensible à l'évolution du genre.\n\n"
            else:
                md += f"**Q3 — Temporal dynamics:** {best_model.upper()} shows highest temporal variance ({best_var:.6f}), making it most sensitive to genre evolution.\n\n"

    # Q4: Vocabulary — full-vocab Jaccard
    cross_model_jaccard = results.get('cross_model_jaccard', {})
    cross_bl = cross_model_jaccard.get('bertopic_vs_lda', {}).get('per_threshold', {})
    if cross_bl:
        jacc_5 = cross_bl.get(5, {}).get('mean_jaccard', 0)
        jacc_1 = cross_bl.get(1, {}).get('mean_jaccard', 0)
        if lang == 'fr':
            md += f"**Q4 — Chevauchement lexical :** Le Jaccard vocabulaire complet entre BERTopic et LDA varie de {jacc_1:.4f} (seuil=1, vocabulaire fonctionnel partagé) à des valeurs plus faibles aux seuils supérieurs (seuil=5 : {jacc_5:.4f}), confirmant la divergence sur le vocabulaire spécialisé.\n\n"
        else:
            md += f"**Q4 — Lexical overlap:** Full-vocab Jaccard between BERTopic and LDA ranges from {jacc_1:.4f} (threshold=1, shared function words) to lower values at higher thresholds (threshold=5: {jacc_5:.4f}), confirming divergence on specialized vocabulary.\n\n"

    # Q5: Intra-topic homogeneity
    intra = results.get('intra_topic_distances', {})
    if intra:
        best_model_js, best_js = None, 1.0
        best_model_labbe, best_labbe = None, 1.0
        for model in ['bertopic', 'lda', 'iramuteq']:
            js_mean = intra.get(f'{model}_js', {}).get('mean', 1.0)
            labbe_mean = intra.get(f'{model}_labbe', {}).get('mean', 1.0)
            if js_mean < best_js:
                best_js = js_mean
                best_model_js = model
            if labbe_mean < best_labbe:
                best_labbe = labbe_mean
                best_model_labbe = model
        if best_model_js:
            if lang == 'fr':
                md += f"**Q5 — Homogénéité intra-topic :** {best_model_labbe.upper()} présente la meilleure homogénéité lexicale (Labbé = {best_labbe:.4f}), {best_model_js.upper()} la meilleure homogénéité distributionnelle (JS = {best_js:.4f}).\n\n"
            else:
                md += f"**Q5 — Intra-topic homogeneity:** {best_model_labbe.upper()} shows best lexical homogeneity (Labbé = {best_labbe:.4f}), {best_model_js.upper()} best distributional homogeneity (JS = {best_js:.4f}).\n\n"

    # χ²/n
    chi2_results = results.get('chi2_results', {})
    surface = chi2_results.get('non_lemmatized', {})
    if surface:
        best_model_chi2, best_chi2n = None, -1
        for model in ['bertopic', 'lda', 'iramuteq']:
            chi2n = surface.get(model, {}).get('chi2_over_n', 0)
            if chi2n > best_chi2n:
                best_chi2n = chi2n
                best_model_chi2 = model
        if best_model_chi2:
            if lang == 'fr':
                md += f"**χ²/n — Dépendance mot-topic :** {best_model_chi2.upper()} montre la plus forte association mot-topic (χ²/n = {best_chi2n:.4f}), indiquant des topics lexicalement plus distinctifs.\n\n"
            else:
                md += f"**χ²/n — Word-topic dependency:** {best_model_chi2.upper()} shows strongest word-topic association (χ²/n = {best_chi2n:.4f}), indicating more lexically distinctive topics.\n\n"

    # Complementarity statement
    if lang == 'fr':
        md += "**Complémentarité des approches :** Les trois modèles capturent des aspects distincts du corpus :\n"
        md += "- **BERTopic** : similarité sémantique via embeddings neuronaux\n"
        md += "- **LDA** : co-occurrences de mots via modèle génératif probabiliste\n"
        md += "- **IRAMUTEQ** : mondes lexicaux via classification hiérarchique descendante (ALCESTE)\n\n"
        md += "L'utilisation conjointe de ces trois approches fournit une caractérisation multi-dimensionnelle du corpus, chaque modèle éclairant des facettes complémentaires de la structure thématique.\n\n"
    else:
        md += "**Complementary approaches:** The three models capture distinct aspects of the corpus:\n"
        md += "- **BERTopic**: semantic similarity via neural embeddings\n"
        md += "- **LDA**: word co-occurrences via probabilistic generative model\n"
        md += "- **IRAMUTEQ**: lexical worlds via descending hierarchical classification (ALCESTE)\n\n"
        md += "Using all three approaches provides a multi-dimensional characterization, each model illuminating complementary facets of the thematic structure.\n\n"

    return md


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
    md += f"""### {t('q1_title')}

{t('q1_research')}

#### {t('q1_method_intro')}

**Adjusted Rand Index (ARI)** — {METRIC_REFERENCES['ari']['citation']}

{get_metric_description('ari', lang)}

**Normalized Mutual Information (NMI)** — {METRIC_REFERENCES['nmi']['citation']}

{get_metric_description('nmi', lang)}

#### {t('q1_results')}

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
    md += f"""### {t('q2_title')}

{t('q2_research')}

#### {t('q2_method_intro')}

**{t('cramers_v')}** — {METRIC_REFERENCES['cramers_v']['citation']}

{get_metric_description('cramers_v', lang)}

#### {t('q2_results')}

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
    md += f"""### {t('q3_title')}

{t('q3_research')}

#### {t('q3_method_intro')}

{t('temporal_variance_desc')}

#### {t('q3_results')}

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

        if best_model:
            md += f"""
{t('key_observations')}

1. {t('q3_obs1').format(model=best_model.upper())}

2. {t('q3_obs2')}

3. {t('q3_obs3')}

"""

    # Q4: Vocabulary
    md += f"""### {t('q4_title')}

{t('q4_research')}

#### {t('q4_method_intro')}

{t('jaccard_desc')}

{t('distinctiveness_desc')}

#### {t('q4_results')}

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

        # Cross-model full vocabulary Jaccard at multiple thresholds
        cross_model_jaccard = results.get('cross_model_jaccard', {})
        cross_bl = cross_model_jaccard.get('bertopic_vs_lda', {})
        per_threshold = cross_bl.get('per_threshold', {})
        if per_threshold:
            md += f"#### {t('cross_model_full_jaccard_title')}\n\n"
            md += f"{t('cross_model_full_jaccard_intro')}\n\n"
            if lang == 'fr':
                md += "| Seuil min. freq. | Jaccard moyen | Paires |\n"
            else:
                md += "| Min freq. threshold | Mean Jaccard | Pairs |\n"
            md += "|-----|------|------|\n"
            for thresh in sorted(per_threshold.keys()):
                data = per_threshold[thresh]
                md += f"| {thresh} | {data['mean_jaccard']:.4f} | {data['n_pairs']} |\n"
            md += "\n"

    # Q5: Intra-topic distance (legacy section for backward compatibility)
    distance_results = results.get('intra_topic_distances', {})
    if distance_results:
        md += generate_intra_topic_distance_section(distance_results, lang=lang)

    # Q5 Extended: All 4 distance configurations (if available)
    topic_distance_results = results.get('topic_distance_results', {})
    aggregation_size = results.get('aggregation_size', 20)
    if topic_distance_results:
        md += generate_topic_distance_4configs_section(
            topic_distance_results,
            aggregation_size=aggregation_size,
            lang=lang
        )

    # Q5 Feature: Aggregation stabilization curve
    multi_agg_results = results.get('multi_agg_results', {})
    if multi_agg_results:
        md += generate_aggregation_curve_section(
            multi_agg_results,
            agg_metadata=results.get('agg_metadata'),
            lang=lang
        )

    # Q5 Feature: Inter-topic separation ranking (centroid distances)
    centroid_results = results.get('centroid_results', {})
    topic_labels_per_model = results.get('topic_labels_per_model', {})
    if centroid_results:
        md += generate_inter_topic_ranking_section(
            centroid_results,
            topic_labels_per_model=topic_labels_per_model,
            lang=lang
        )

    # Q5 Feature: χ²/n word × topic independence test
    chi2_results = results.get('chi2_results', {})
    if chi2_results:
        md += generate_word_topic_chi2_section(
            chi2_results,
            topic_labels_per_model=topic_labels_per_model,
            lang=lang
        )

    # Section 4: Summary — built from actual computed results
    md += f"\n## {t('summary_title')}\n\n"
    md += f"### {t('key_findings')}\n\n"
    md += _build_dynamic_conclusion(results, lang)
    md += "\n"

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
## {t('references_title')}

### {t('clustering_agreement_refs')}

- {full_ref('ari')}
- {full_ref('nmi')}
- {full_ref('ami')}

### {t('association_measures_refs')}

- {full_ref('cramers_v')}

### {t('info_theory_refs')}

- {full_ref('js_divergence')}

### {t('intertextual_refs')}

{full_refs_with_additional('labbe_distance')}

### {t('topic_coherence_refs')}

- {full_ref('coherence_cv')}

### {t('cluster_validation_refs')}

- {full_ref('silhouette')}

### {t('topic_modeling_refs')}

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.
- Reinert, M. (1983). Une méthode de classification descendante hiérarchique : application à l'analyse lexicale par contexte. Les Cahiers de l'Analyse des Données, 8(2), 187-198.

"""

    # Appendix
    md += f"""
## {t('appendix_title')}

### {t('run_details')}

- **{t('comparison_timestamp')}:** {timestamp}
- **{t('bertopic_folder')}:** {results.get('bertopic', {}).get('run_dir', 'N/A')}
- **{t('lda_folder')}:** {results.get('lda', {}).get('run_dir', 'N/A')}
- **{t('iramuteq_folder')}:** {results.get('iramuteq', {}).get('run_dir', 'N/A')}

"""

    # Add distance appendix
    md += generate_distance_appendix(lang=lang)

    return md
