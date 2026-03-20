# Rapport de Comparaison des Modèles de Topics

**Généré le:** 2026-03-18 18:20:44

**Répertoire de sortie:** `/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/results/comparisons/comparison_20260318_175850`

---

## Résumé

Ce rapport présente une comparaison approfondie de trois approches de modélisation de topics appliquées à un corpus de paroles de rap français : **BERTopic** (embeddings neuronaux), **LDA** (modèle génératif probabiliste), et **IRAMUTEQ** (classification lexicale par la méthode ALCESTE de Reinert). L'analyse se décompose comme suit :

1. **Description du corpus** : caractérisation statistique du corpus (nombre de documents, artistes, couverture temporelle, distribution par décennie).
2. **Description des modèles individuels** : paramètres, métriques de qualité (cohérence C_v, silhouette), distribution des topics et séparation des artistes pour chaque modèle.
3. **Q1 — Accord entre modèles** : évaluation de la similarité des clusterings à l'aide de l'ARI (Hubert & Arabie, 1985), la NMI (Strehl & Ghosh, 2002) et l'AMI (Vinh et al., 2010), avec analyse des correspondances inter-topics.
4. **Q2 — Séparation des artistes** : mesure de l'association artiste-topic par le V de Cramér (Cramér, 1946) et analyse des résidus de Pearson standardisés.
5. **Q3 — Dynamique temporelle** : analyse de la variance des distributions de topics dans le temps et de la divergence de Jensen-Shannon entre périodes biannuelles.
6. **Q4 — Distinctivité lexicale** : évaluation du chevauchement du vocabulaire entre topics par la distance de Jaccard et analyse inter-modèles.
7. **Q5 — Homogénéité intra-topic** : mesure de la cohérence lexicale des clusters par les distances de Labbé (Labbé & Labbé, 2001) et de Jensen-Shannon (Lin, 1991), calculées sur des documents tokenisés avec spaCy.
8. **Synthèse et recommandations** : bilan comparatif des trois modèles et recommandations d'usage.

---

## 1. Description du Corpus

Cette section présente une vue d'ensemble du corpus de paroles de rap français utilisé pour la modélisation de topics. Le corpus est constitué de couplets individuels extraits de chansons, avec des métadonnées incluant l'artiste, le titre et l'année.

### 1.1 Vue d'ensemble du jeu de données

| Métrique | Valeur |
|--------|-------|
| **Documents totaux (couplets)** | 115,805 |
| **Période couverte** | 1992 - 2023 |
| **Artistes uniques** | 605 |
| **Année moyenne** | 2014.7 |
| **Année médiane** | 2017 |
| **Docs moyens par artiste** | 191.4 |
| **Docs médians par artiste** | 137 |
| **Artiste le plus prolifique** | JuL (2,878 docs) |

### 1.2 Couverture Temporelle

Le corpus couvre plus de trois décennies de rap français, de l'ère pionnière des années 1990 à la scène contemporaine.

![Distribution](figures/corpus_year_distribution.png)

*Figure 1.1 : Gauche - Nombre de documents par année. Droite - Distribution de la productivité des artistes (échelle log).*

![Decades](figures/corpus_decade_breakdown.png)

*Figure 1.2 : Distribution du corpus par décennie.*

| Décennie | Documents | % du Corpus |
|--------|-----------|-------------|
| 1990s | 4,517 | 3.9% |
| 2000s | 19,605 | 16.9% |
| 2010s | 60,895 | 52.6% |
| 2020s | 30,788 | 26.6% |

### 1.3 Top 10 Artistes par Nombre de Documents

| Rang | Artiste | Documents | % du Corpus | % Cumulatif |
|------|--------|-----------|-------------|--------------|
| 1 | JuL | 2,878 | 2.5% | 2.5% |
| 2 | La Fouine | 1,191 | 1.0% | 3.5% |
| 3 | Rohff | 1,095 | 0.9% | 4.5% |
| 4 | Sexion d’Assaut | 1,023 | 0.9% | 5.3% |
| 5 | Alkpote | 889 | 0.8% | 6.1% |
| 6 | Naps | 854 | 0.7% | 6.8% |
| 7 | Disiz | 849 | 0.7% | 7.6% |
| 8 | Swift Guad | 797 | 0.7% | 8.3% |
| 9 | IAM | 745 | 0.6% | 8.9% |
| 10 | Sinik | 744 | 0.6% | 9.6% |

**Concentration du corpus** : Les 10 premiers artistes représentent 9.6% du corpus, indiquant une distribution d'artistes diverse.


---

## 2. Description des Modèles Individuels

Cette section présente la configuration, les métriques de qualité et les topics découverts par chaque modèle. Nous incluons les visualisations spécifiques à chaque modèle pour contextualiser l'analyse comparative.

### 2.1 BERTopic

**Dossier du run:** `results/BERTopic/run_20260126_141647_mpnet`

BERTopic (Grootendorst, 2022) est un modèle de topics neuronal utilisant des embeddings de transformers pour la représentation des documents, UMAP pour la réduction de dimensionnalité, et le clustering. Les topics sont représentés via pondération c-TF-IDF, avec labellisation optionnelle par OpenAI et KeyBERT.

#### Paramètres

| Paramètre | Valeur |
|-----------|-------|
| embedding_model | `sentence-transformers/all-mpnet-base-v2` |
| embedding_key | `mpnet` |
| clustering_algorithm | `kmeans` |
| n_clusters | `20` |
| hdbscan_params | `None` |
| agglomerative_params | `None` |
| umap_params | n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine, random_state=42 |
| num_words_per_topic | `30` |
| use_openai | `True` |
| include_keybert | `True` |
| interactive_html | `True` |

#### Qualité du Clustering

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Silhouette (UMAP) | 0.1717 | Modéré |

*Le score de silhouette (Rousseeuw, 1987) mesure la séparation des clusters.*

![Silhouette](figures/bertopic_silhouette_plot.png)


#### Distribution des Topics

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Nombre de topics | 20 | - |
| Ratio d'imbalance | 2.34 | Modérément équilibré |
| Entropie de distribution | 0.992 | Quasi uniforme |

**Définitions des métriques :**

- **Ratio d'imbalance** = max(compte_topic) / min(compte_topic). Mesure l'inégalité des tailles de topics.

- **Entropie de distribution** (normalisée) = -Σ(p_i × log(p_i)) / log(n_topics). Intervalle [0,1] : 1 = uniforme.

![Topic Distribution](figures/bertopic_topic_distribution.png)

*Distribution des documents par topic.*


#### Séparation des Artistes

| Métrique | Valeur | Description |
|--------|-------|-------------|
| % Spécialistes | 2.2% | Artistes avec >50% dans un topic |
| % Modérés | 11.5% | Artistes avec 25-50% dans le topic dominant |
| % Généralistes | 86.3% | Artistes répartis sur plusieurs topics |
| Indice de spécialisation | 0.168 | Concentration moyenne |
| Divergence JS | 0.451 | Divergence des profils d'artistes |

![Artist-Topic Heatmap](figures/bertopic_artist_topics_heatmap.png)

*Heatmap de distribution des artistes par topic.*

![Artist Specialization](figures/bertopic_artist_specialization.png)

*Profils de spécialisation des artistes.*


#### Dynamique Temporelle

| Métrique | Valeur |
|--------|-------|
| Variance moyenne des topics | 0.000655 |
| JS annuel moyen | 0.0803 |

![Annual JS](figures/bertopic_annual_js_divergence.png)

*Divergence JS entre années consécutives.*

![Year-Topic Heatmap](figures/bertopic_year_topic_heatmap.png)

*Évolution des topics dans le temps.*


#### Vue d'ensemble des Topics

**Topic 0** — *Révolte et Identité dans le Rap*

- **c-TF-IDF:** de, le, les, est, la, pas, et, rap, un, on
- **KeyBERT (15 terms):** pas, ceux, ouais, rap, rappe, fait, comme, avec, monde, musique, vous, jamais, une, veux, que

**Topic 1** — *Pression Sociale et Ambitions Contradictoires*

- **c-TF-IDF:** ouais, la, oh, est, hey, yeah, pas, dans, le, on
- **KeyBERT (15 terms):** ouais, oui, pas, vous, fait, comme, pour, vois, jamais, avec, vie, tu, veux, une, sur

**Topic 2** — *Rébellion et Résilience dans la Rue*

- **c-TF-IDF:** les, la, le, de, on, pas, est, des, et, en
- **KeyBERT (15 terms):** pas, ouais, ceux, fait, vous, avec, comme, veux, monde, sur, jamais, vie, leur, une, pour

**Topic 3** — *Résistances et Réalités Urbaines*

- **c-TF-IDF:** de, la, les, le, est, et, on, des, un, pas
- **KeyBERT (15 terms):** pas, entre, leurs, fait, monde, ceux, avec, comme, une, sur, vous, veux, vie, france, leur

**Topic 4** — *Interrogations Existentielles et Solitude*

- **c-TF-IDF:** la, est, pas, on, dans, le, qu, tu, je, les
- **KeyBERT (15 terms):** pourquoi, que, pas, fait, ouais, avec, monde, mais, vie, sur, quoi, qui, peux, oui, jamais

**Topic 5** — *Réalité Crue et Résilience Urbaine*

- **c-TF-IDF:** les, la, le, de, est, pas, on, des, dans, et
- **KeyBERT (15 terms):** pas, ouais, vous, avec, fait, comme, monde, que, jamais, vie, une, sur, vois, veux, tu

**Topic 6** — *Résilience et Lutte pour la Vie*

- **c-TF-IDF:** la, pas, le, les, de, est, on, je, ai, tu
- **KeyBERT (15 terms):** pas, ouais, vous, fait, peux, avec, comme, vie, passe, veux, pour, monde, tu, une, vois

**Topic 7** — *Réflexions sur la Solitude et l'Injustice*

- **c-TF-IDF:** de, la, les, le, et, des, un, est, dans, en
- **KeyBERT (15 terms):** pas, fait, ceux, leurs, avec, entre, vie, comme, vous, monde, sur, une, veux, jour, homme

**Topic 8** — *Amour, Perte et Résilience Urbaine*

- **c-TF-IDF:** oh, la, pas, est, moi, je, ai, tu, ma, on
- **KeyBERT (15 terms):** ouais, pas, cœur, oui, peux, vie, avec, fait, laisse, vois, sur, comme, veux, mais, pour

**Topic 9** — *Desespoir et Lutte pour la Survivance*

- **c-TF-IDF:** de, la, le, les, est, et, je, que, un, on
- **KeyBERT (15 terms):** pas, cœur, fait, ceux, vie, comme, avec, monde, vous, sur, jour, une, pour, leur, veux

**Topic 10** — *Amours Perdue et Solitude Urbaine*

- **c-TF-IDF:** je, tu, pas, moi, est, que, ai, de, la, toi
- **KeyBERT (15 terms):** vie, comme, pas, laisse, cœur, fait, voir, ouais, pourquoi, avec, mais, moi, pour, peux, jour

**Topic 11** — *Chagrin et Résilience en Milieu Urbain*

- **c-TF-IDF:** la, pas, ai, est, je, de, tu, elle, le, qu
- **KeyBERT (15 terms):** pas, cœur, ouais, vie, vous, fait, peux, pour, avec, sur, tu, veux, mais, comme, jamais

**Topic 12** — *Réalité du Ghetto et Résilience*

- **c-TF-IDF:** la, de, les, le, est, dans, un, des, pas, on
- **KeyBERT (15 terms):** pas, ouais, avec, fait, comme, vie, vous, sur, veux, que, tu, monde, jamais, une, chez

**Topic 13** — *Rêves de Réussite et Réalité de la Rue*

- **c-TF-IDF:** la, pas, est, le, les, dans, suis, ai, on, de
- **KeyBERT (15 terms):** pas, vous, ouais, bon, peux, avec, vie, fait, comme, tu, veux, pour, que, connais, sur

**Topic 14** — *Amour, Trahison et Désillusion*

- **c-TF-IDF:** elle, je, de, est, que, la, amour, pas, et, qu
- **KeyBERT (15 terms):** pas, fait, vie, comme, avec, cœur, amour, pour, mais, fais, vois, jamais, tu, sur, jour


... et 5 autres topics

![UMAP](figures/bertopic_umap_topics.png)

*Projection UMAP des embeddings colorés par topic.*


### 2.2 LDA

**Dossier du run:** `results/LDA/run_20260224_094852_bigram_only`

Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003) est un modèle génératif probabiliste représentant les documents comme mélanges de topics, où chaque topic est une distribution sur les mots. Cette implémentation utilise Gensim avec prétraitement n-gram.

#### Paramètres

| Paramètre | Valeur |
|-----------|-------|
| num_topics | `20` |
| alpha | `symmetric` |
| eta | `auto` |
| passes | `15` |
| iterations | `400` |
| min_word_len | `2` |
| min_doc_freq | `5` |
| max_doc_freq_ratio | `0.3` |
| use_ngrams | `bigram_only` |
| ngram_min_count | `10` |
| ngram_threshold | `50` |
| num_words_per_topic | `30` |
| keep_all_the_document | `True` |

#### Scores de Cohérence

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Cohérence C_v | 0.5897 | Bon |
| Cohérence UMass | -10.2532 | Modéré |

*La cohérence C_v (Röder et al., 2015) mesure la cohérence sémantique des mots des topics.*

![Coherence](figures/lda_coherence_plot.png)


#### Distribution des Topics

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Nombre de topics | 20 | - |
| Ratio d'imbalance | 17.06 | Très déséquilibré |
| Entropie de distribution | 0.887 | Bien distribué |

**Définitions des métriques :**

- **Ratio d'imbalance** = max(compte_topic) / min(compte_topic). Mesure l'inégalité des tailles de topics.

- **Entropie de distribution** (normalisée) = -Σ(p_i × log(p_i)) / log(n_topics). Intervalle [0,1] : 1 = uniforme.

![Topic Distribution](figures/lda_topic_distribution.png)

*Distribution des documents par topic.*


#### Séparation des Artistes

| Métrique | Valeur | Description |
|--------|-------|-------------|
| % Spécialistes | 0.5% | Artistes avec >50% dans un topic |
| % Modérés | 22.4% | Artistes avec 25-50% dans le topic dominant |
| % Généralistes | 77.0% | Artistes répartis sur plusieurs topics |
| Indice de spécialisation | 0.186 | Concentration moyenne |
| Divergence JS | 0.140 | Divergence des profils d'artistes |

![Artist-Topic Heatmap](figures/lda_artist_topics_heatmap.png)

*Heatmap de distribution des artistes par topic.*

![Artist Specialization](figures/lda_artist_specialization.png)

*Profils de spécialisation des artistes.*


#### Dynamique Temporelle

| Métrique | Valeur |
|--------|-------|
| Variance moyenne des topics | 0.000032 |
| JS annuel moyen | 0.0272 |

![Annual JS](figures/lda_annual_js_divergence.png)

*Divergence JS entre années consécutives.*

![Year-Topic Heatmap](figures/lda_year_topic_heatmap.png)

*Évolution des topics dans le temps.*


#### Vue d'ensemble des Topics

| Topic | Mots clés |
|-------|----------|
| 0 | coeur, bye_bye, savais, comptes, yeux_rouges, madame, voudrais, full, douce_france, satan |
| 1 | d'la, rue, merde, putain, parle, gars, connais, j'vais, négro, j'veux |
| 2 | fils_pute, j'm'en_fous, j'en_marre, chaud_chaud, fils_putes, yeux_fermés, bord_mer, gucci, i'm, canon_scié |
| 3 | xxx, bat_couilles, semblant, quatre, briller, air_max, fumé, tient, grammes, taff |
| 4 | j'y_pense, ailleurs, pense_qu'à, d'ma_mère, aurait_pu, rend_fou, prie, j'vous, po_po, sac_dos |
| 5 | m'a, vu, j'veux, j'étais, j'me, j'vais, j'peux, c'était, j'avais, pris |
| 6 | belle, sale, j'connais, danser, étais, vies, petite, bonne, meilleur, dieu_merci |
| 7 | comment, ouh, ciel, hip_hop, j'te_jure, étoiles, souvenirs, nulle_part, barre, montre |
| 8 | deux_trois, d'en_bas, ferme_yeux, jusqu'au_bout, tirer, l'impression_d'être, m'a_rendu, roue_tourne, j'rentre, s'il_plait |
| 9 | j'pense, m'as, bon, dix_ans, donné, t'aime, voulais, trouvé, rempli, n'a_changé |
| 10 | bang_bang, p'tit, j'me_rappelle, tess, beuh, bienvenue, dalle, solo, équipe, paw_paw |
| 11 | j'me_sens, paroles_rédigées, j'aurais_pu, expliquées_communauté, qu'j'ai, terrain, rapgenius_france, disque_d'or, s'il_plaît, doigt |
| 12 | demain, zone, ferme_gueule, mama, fera, jours, marseille, ira, bloc, obligé |
| 13 | vas, t'inquiète, tourne_rond, années_passent, qu'tu_sois, souffrir, faits_divers, nord_sud, l'être_humain, compte_banque |
| 14 | j'aime, bébé, noir, lune, rentre, j'prends, aura, qu'j'suis, paye, danse |

... et 5 autres topics

![PCA](figures/lda_topic_pca.png)

*Projection PCA des distributions topic-mot.*


### 2.3 IRAMUTEQ

**Dossier du run:** `results/IRAMUTEQ/evaluation_20260126_124001`

IRAMUTEQ implémente la méthode ALCESTE de Reinert (Reinert, 1983), qui effectue une classification hiérarchique descendante sur les segments de texte, identifiant les mondes lexicaux.

#### Paramètres

| Paramètre | Valeur |
|-----------|-------|
| method | `IRAMUTEQ` |
| n_classes | `20` |
| n_documents | `115805` |
| min_docs_per_artist | `10` |
| top_artists_per_topic | `20` |

#### Distribution des Topics

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Nombre de topics | 20 | - |
| Ratio d'imbalance | 31.70 | Très déséquilibré |
| Entropie de distribution | 2.709 | Quasi uniforme |

**Définitions des métriques :**

- **Ratio d'imbalance** = max(compte_topic) / min(compte_topic). Mesure l'inégalité des tailles de topics.

- **Entropie de distribution** (normalisée) = -Σ(p_i × log(p_i)) / log(n_topics). Intervalle [0,1] : 1 = uniforme.

![Topic Distribution](figures/iramuteq_topic_distribution.png)

*Distribution des documents par topic.*


#### Séparation des Artistes

| Métrique | Valeur | Description |
|--------|-------|-------------|
| % Spécialistes | 12.2% | Artistes avec >50% dans un topic |
| % Modérés | 44.3% | Artistes avec 25-50% dans le topic dominant |
| % Généralistes | 43.4% | Artistes répartis sur plusieurs topics |
| Indice de spécialisation | 0.364 | Concentration moyenne |
| Divergence JS | 0.541 | Divergence des profils d'artistes |

![Artist-Topic Heatmap](figures/iramuteq_artist_topics_heatmap.png)

*Heatmap de distribution des artistes par topic.*

![Artist Specialization](figures/iramuteq_artist_specialization.png)

*Profils de spécialisation des artistes.*


#### Dynamique Temporelle

| Métrique | Valeur |
|--------|-------|
| Variance moyenne des topics | 0.001857 |
| JS annuel moyen | 0.1019 |

![Annual JS](figures/iramuteq_annual_js_divergence.png)

*Divergence JS entre années consécutives.*

![Year-Topic Heatmap](figures/iramuteq_year_topic_heatmap.png)

*Évolution des topics dans le temps.*


#### Vue d'ensemble des Topics

| Topic | Mots clés |
|-------|----------|
| 1 | with, and, that, when, they, you, it, can, the, like |
| 2 | chen, ekip, ldo, zuukou, etho, goddamn, nrm, digi, mms, lin |
| 3 | jul, marseille, gadji, moto, poto, fumette, bdh, zone, dégun, miss |
| 4 | hey, brr, yah, gucci, grr, gang, bébé, mmh, woh, fendi |
| 5 | luni, slimes, geeked, shawty, sacki, voidd, drip, majdon, ola, slime |
| 6 | sexion, wati, assaut, 9ème, gims, jeryzoos, akhi, maska, llefa, 3ème |
| 7 | bitch, négro, flow, meuf, club, weed, flex, boy, yo, dj |
| 8 | france, peuple, pays, politique, afrique, communauté, rédiger, justice, état, système |
| 9 | cli, ients, binks, détailler, midi, minuit, visser, gue, terrain, pe |
| 10 | mic, rime, rap, style, mc, hip_hop, rimer, beat, micro, texte |
| 11 | art, swift, acide, artère, guad, tekk, carcasse, delleck, corps, nikkfurie |
| 12 | chose, impression, penser, fois, temps, gens, question, moment, envie, vraiment |
| 13 | amour, aimer, sentiment, mentir, couple, amoureux, relation, femme, coeur, défaut |
| 14 | 2mz, qlf, sourou, igd, pnl, adios, amigo, benab, igo, rio |
| 15 | billet, violet, monnaie, vert, liasse, euro, poche, payer, charbonner, bleu |

... et 5 autres topics

---

## 3. Analyse Comparative

### 3.1 Q1 : Les modèles capturent-ils la même structure ?

**Question de recherche :** Les différentes approches découvrent-elles des structures similaires ?

#### Contexte Méthodologique

Nous utilisons trois métriques d'accord de clustering :

**Adjusted Rand Index (ARI)** — Hubert, L., & Arabie, P. (1985)

L'indice de Rand ajusté mesure la similarité entre deux clusterings,
corrigé par le hasard. Il calcule le nombre d'accords de paires (toutes deux dans le même cluster
ou toutes deux dans des clusters différents), normalisé par la valeur attendue sous un modèle aléatoire.
ARI = (RI - RI_attendu) / (RI_max - RI_attendu).
Intervalle : [-1, 1], où 1 = accord parfait, 0 = aléatoire, <0 = inférieur au hasard.

**Normalized Mutual Information (NMI)** — Strehl, A., & Ghosh, J. (2002)

La NMI mesure la dépendance mutuelle entre deux clusterings en utilisant la
théorie de l'information. Elle quantifie dans quelle mesure la connaissance d'un clustering réduit
l'incertitude sur l'autre. NMI = 2 × I(X;Y) / (H(X) + H(Y)), où I est l'information mutuelle et H l'entropie.
Intervalle : [0, 1], où 1 = clusterings identiques, 0 = indépendants.

#### Résultats

| Paire | ARI | NMI | Interprétation |
|------|-----|-----|----------------|
| bertopic_vs_lda | 0.0076 | 0.0150 | Quasi aléatoire |
| bertopic_vs_iramuteq | 0.0442 | 0.1024 | Accord faible |
| lda_vs_iramuteq | 0.0049 | 0.0151 | Quasi aléatoire |

**Observations clés :**

1. **Meilleur accord :** bertopic_vs_iramuteq (NMI = 0.1024)

2. **Accord le plus faible :** bertopic_vs_lda (NMI = 0.0150)

3. **Pattern général :** Les scores d'accord relativement faibles (NMI < 0.5) suggèrent que chaque modèle capture des aspects distincts :
- BERTopic : similarité sémantique (sens)
- LDA : co-occurrences de mots (distribution)
- IRAMUTEQ : classification lexicale (vocabulaire)

### 3.2 Q2 : Les modèles séparent-ils les artistes ?

**Question de recherche :** Les topics capturent-ils des signatures stylistiques propres aux artistes ?

#### Contexte Méthodologique

**V de Cramér** — Cramér, H. (1946)

Le V de Cramér mesure la force d'association entre deux variables catégorielles.
Il est dérivé de la statistique du chi-deux : V = √(χ² / (n × min(k-1, r-1))),
où k et r sont le nombre de catégories. V normalise le chi-deux par la taille de l'échantillon
et la dimensionnalité, permettant la comparaison entre tables de tailles différentes.
Intervalle : [0, 1], où 0 = aucune association, 1 = association parfaite.

#### Résultats

| Modèle | V de Cramér | Interprétation |
|-------|----------|----------------|
| BERTOPIC | 0.2079 | Association modérée |
| LDA | 0.1143 | Association faible |
| IRAMUTEQ | 0.3854 | Association forte |

**Observations clés :**

1. **Séparation la plus forte :** IRAMUTEQ (V = 0.3854)

2. **Spécialistes :** La proportion varie selon les modèles.

3. **Généralistes :** Artistes répartis sur plusieurs topics = thèmes divers.

### 3.3 Q3 : Les modèles capturent-ils l'évolution temporelle ?

**Question de recherche :** Les distributions de topics changent-elles dans le temps ?

#### Contexte Méthodologique

**Variance Temporelle** : Mesure la fluctuation des topics au fil du temps.

#### Résultats

| Modèle | Variance Temporelle | Topic le plus variable | Variance max | Interprétation |
|--------|-----|-----|-----|----------------|
| BERTOPIC | 0.000677 | 3 | 0.003566 | Stable |
| LDA | 0.000034 | 16 | 0.000094 | Stable |
| IRAMUTEQ | 0.001917 | 9 | 0.015929 | Dynamique modérée |

**Observations clés :**

1. **Le plus dynamique :** IRAMUTEQ montre la variance la plus élevée.

2. **Transitions majeures :** Une divergence JS élevée entre décennies indique des changements.

3. **Topics stables vs évolutifs :** Faible variance = thèmes pérennes, haute variance = tendances.

### 3.4 Q4 : Quelle est la distinctivité lexicale des topics ?

**Question de recherche :** Les topics représentent-ils des vocabulaires distincts ?

#### Contexte Méthodologique

**Distance de Jaccard** : Mesure la distinctivité du vocabulaire entre topics.

**Distinctivité** : Distance de Jaccard moyenne entre vocabulaires de topics.

#### Résultats

| Modèle | Distance de Jaccard Moyenne | Interprétation |
|--------|-----|----------------|
| BERTOPIC | 0.5062 | Chevauchement modéré |
| LDA | 0.9977 | Topics très distincts |
| IRAMUTEQ | 0.9964 | Topics très distincts |

**Observations clés :**

1. **LDA et IRAMUTEQ** montrent une haute distinctivité (>0.9).

2. **BERTopic** peut montrer une distinctivité plus faible (embeddings sémantiques).

#### Chevauchement Lexical Inter-Modèles (Vocabulaire Complet)

Pour évaluer le recouvrement lexical entre les topics correspondants de BERTopic et LDA, nous calculons l'indice de Jaccard sur le **vocabulaire complet** des documents assignés à chaque topic (et non sur les seuls mots représentatifs extraits par c-TF-IDF ou probabilité). Nous faisons varier le seuil de fréquence minimale pour distinguer le vocabulaire fonctionnel partagé (seuil bas, Jaccard élevé) du vocabulaire thématique spécifique (seuil élevé, Jaccard plus faible). Un Jaccard décroissant avec le seuil indique que les modèles divergent sur les termes spécialisés tout en partageant le socle lexical commun.

| Seuil min. freq. | Jaccard moyen | Paires |
|-----|------|------|
| 1 | 0.3639 | 20 |
| 5 | 0.3904 | 20 |
| 20 | 0.3391 | 20 |

### 3.5 Q5 : Quelle est l'homogénéité lexicale des topics ?

**Question de recherche :** Les documents d'un même topic sont-ils lexicalement similaires ?

Des distances intra-topic plus faibles indiquent des clusters plus cohérents.

#### Contexte Méthodologique

Nous calculons les distances par paires entre documents du même topic. Deux métriques complémentaires :

| Distance | Ce qu'elle capture | Justification scientifique |
|----------|------------------|--------------------------|
| **Jensen-Shannon** | Divergence distributionnelle | Largement utilisée en NLP. Fondée sur la théorie de l'information. Bornée [0,1]. |
| **Labbé** | Homogénéité lexicale | Standard JADT pour la stylométrie française. |

**Jensen-Shannon** — Lin, J. (1991)

La distance JS est la racine carrée de la divergence JS, une mesure
symétrique de similarité entre distributions de probabilité. scipy.spatial.distance.jensenshannon()
retourne directement cette valeur de distance. La distance JS est une métrique propre satisfaisant
l'inégalité triangulaire.
Intervalle : [0, 1], où 0 = distributions identiques, 1 = maximalement différentes.

**Labbé** — Labbé, D., & Labbé, C. (2001)

La distance de Labbé mesure la similarité lexicale entre deux textes, implémentée
selon l'algorithme original d'IRAMUTEQ. Elle gère explicitement l'asymétrie de longueur entre textes :
1) Identifier le texte plus petit (N_small) et plus grand (N_large)
2) Normaliser les comptages du texte plus grand : n'_i = n_i × U où U = N_small/N_large
3) Calculer la somme des différences absolues sur les comptages normalisés
4) Normaliser : D = Σ|n_small - n'_large| / (N_small + Σ(n' où n'≥1))
Cette métrique est le standard en stylométrie française et dans la communauté JADT pour l'attribution d'auteur.
Intervalle : [0, 1], où 0 = vocabulaires identiques, 1 = aucun chevauchement.

#### Résultats

**Distance Jensen-Shannon (Distributionnelle)**

| Modèle | Distance Moyenne | Écart-type | Topics | Interprétation |
|-------|---------------|---------|----------|----------------|
| BERTOPIC | 0.8201 | 0.0021 | 20 | Très hétérogène |
| LDA | 0.8203 | 0.0020 | 20 | Très hétérogène |
| IRAMUTEQ | 0.8185 | 0.0069 | 20 | Très hétérogène |

**Distance de Labbé (Lexicale)**

| Modèle | Distance Moyenne | Écart-type | Topics | Interprétation |
|-------|---------------|---------|----------|----------------|
| BERTOPIC | 0.9771 | 0.0040 | 20 | Très hétérogène |
| LDA | 0.9782 | 0.0042 | 20 | Très hétérogène |
| IRAMUTEQ | 0.9748 | 0.0131 | 20 | Très hétérogène |

**Observations clés :**

1. **Meilleure homogénéité distributionnelle (JS) :** IRAMUTEQ montre la distance moyenne la plus faible (0.8185).

2. **Meilleure homogénéité lexicale (Labbé) :** IRAMUTEQ montre la distance moyenne la plus faible (0.9748).

3. **Complémentarité :** JS capture la similarité distributionnelle, Labbé le chevauchement lexical absolu.

#### Analyse par Topic

Top 5 topics les plus et les moins homogènes (distance JS) :

**BERTOPIC**

*Topics les plus homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 14 | 0.8159 | 4959 |
| 0 | 0.8178 | 9049 |
| 10 | 0.8178 | 5437 |
| 17 | 0.8182 | 4736 |
| 9 | 0.8183 | 5470 |

*Topics les moins homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 7 | 0.8215 | 5585 |
| 15 | 0.8216 | 4791 |
| 3 | 0.8216 | 7098 |
| 8 | 0.8230 | 5534 |
| 4 | 0.8250 | 6799 |

**LDA**

*Topics les plus homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 2 | 0.8165 | 5432 |
| 18 | 0.8180 | 5296 |
| 11 | 0.8181 | 7323 |
| 13 | 0.8186 | 2954 |
| 8 | 0.8187 | 5250 |

*Topics les moins homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 15 | 0.8221 | 2276 |
| 1 | 0.8227 | 6637 |
| 5 | 0.8228 | 2988 |
| 19 | 0.8235 | 5748 |
| 0 | 0.8239 | 27139 |

**IRAMUTEQ**

*Topics les plus homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 1 | 0.7899 | 704 |
| 12 | 0.8160 | 7853 |
| 13 | 0.8160 | 5362 |
| 20 | 0.8165 | 11508 |
| 5 | 0.8172 | 363 |

*Topics les moins homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 18 | 0.8219 | 709 |
| 8 | 0.8222 | 7549 |
| 7 | 0.8224 | 10727 |
| 4 | 0.8228 | 6950 |
| 11 | 0.8234 | 4837 |


*Voir Annexe B pour une explication des métriques de distance.*

#### Analyse par Configuration de Distance

Nous calculons les distances selon 4 configurations complémentaires pour évaluer différents aspects de la qualité des topics :

| Configuration | Ce qu'elle mesure | Guide d'interprétation |
|--------------|----------------------|----------------------|
| **Intra-topic (paires)** | Homogénéité | Plus bas = meilleur |
| **Inter-topic (paires)** | Séparation | Plus haut = meilleur |
| **Intra-topic (agrégé)** (n=20) | Homogénéité | Plus bas = meilleur |
| **Inter-topic (agrégé)** (n=20) | Séparation | Plus haut = meilleur |

**Note :** L'agrégation de 20 couplets crée des unités textuelles plus comparables pour la distance de Labbé, qui est sensible à la longueur des textes.

**Intra-topic (paires)**

*Distances entre paires de documents du même topic. Mesure l'**homogénéité interne** : des distances faibles indiquent des topics cohérents.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.8201 | 0.9771 |
| LDA | 0.8203 | 0.9782 |
| IRAMUTEQ | 0.8185 | 0.9748 |

**Inter-topic (paires)**

*Distances entre documents du topic et documents hors du topic. Mesure la **séparation** : des distances élevées indiquent des topics bien distincts.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.8220 | 0.9813 |
| LDA | 0.8216 | 0.9806 |
| IRAMUTEQ | 0.8229 | 0.9832 |

**Intra-topic (agrégé) (n=20)**

*Comme intra-paires, mais en agrégeant n couplets ensemble. Réduit la sensibilité de Labbé aux différences de longueur.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.7111 | 0.7658 |
| LDA | 0.7121 | 0.7659 |
| IRAMUTEQ | 0.7022 | 0.7543 |

**Inter-topic (agrégé) (n=20)**

*Comme inter-paires, mais avec documents agrégés. Plus robuste pour les comparaisons de séparation.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.7261 | 0.7943 |
| LDA | 0.7211 | 0.7840 |
| IRAMUTEQ | 0.7345 | 0.8132 |

**Synthèse des 4 configurations :**

- **Meilleure homogénéité (JS):** IRAMUTEQ (0.8185)
- **Meilleure homogénéité (Labbé):** IRAMUTEQ (0.9748)
- **Meilleure séparation (JS):** IRAMUTEQ (0.8229)
- **Meilleure séparation (Labbé):** IRAMUTEQ (0.9832)


#### Stabilisation de la distance de Labbé par agrégation

Cette analyse montre comment la distance de Labbé évolue en fonction du nombre de couplets agrégés. La distance de Labbé étant sensible à la longueur des textes, l'agrégation de plusieurs couplets produit des unités textuelles plus comparables et des distances plus stables.

**Plage d'agrégation :** de 8 à 72 documents (>500 mots/unité, ≥5 unités/topic), 5 points. Taille minimale de topic (tous modèles) : 363 documents.

![Aggregation Curve](figures/aggregation_curve.png)

*Évolution de la distance de Labbé intra-topic (gauche) et inter-topic (droite) en fonction de la taille d'agrégation.*


#### Classement de la séparation inter-topic

Pour chaque modèle, les topics sont classés par leur distance inter-topic moyenne (un-contre-reste). Des distances plus élevées indiquent des topics lexicalement plus distincts du reste du corpus.

![BERTOPIC Ranking](figures/inter_topic_ranking_bertopic.png)

*Classement des topics par séparation inter-topic (BERTOPIC).*

**Topics les plus distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T8: Amour, Perte et Résilience Urbaine | 0.3417 | 0.3703 |
| T10: Amours Perdue et Solitude Urbaine | 0.3379 | 0.3509 |
| T19: Résilience et Évasion Urbaine | 0.3198 | 0.3469 |

**Topics les moins distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T5: Réalité Crue et Résilience Urbaine | 0.1861 | 0.2331 |
| T2: Rébellion et Résilience dans la Rue | 0.1827 | 0.2379 |
| T16: Récits de Résilience et de Réussite | 0.1716 | 0.2384 |

![LDA Ranking](figures/inter_topic_ranking_lda.png)

*Classement des topics par séparation inter-topic (LDA).*

**Topics les plus distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T5: m'a, vu, j'veux, j'étais, j'me | 0.2502 | 0.3127 |
| T14: j'aime, bébé, noir, lune, rentre | 0.2271 | 0.3085 |
| T6: belle, sale, j'connais, danser, étais | 0.2250 | 0.3109 |

**Topics les moins distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T2: fils_pute, j'm'en_fous, j'en_marre, chaud_chaud, fils_putes | 0.1498 | 0.2189 |
| T8: deux_trois, d'en_bas, ferme_yeux, jusqu'au_bout, tirer | 0.1467 | 0.2198 |
| T11: j'me_sens, paroles_rédigées, j'aurais_pu, expliquées_communauté, qu'j'ai | 0.1359 | 0.2056 |

![IRAMUTEQ Ranking](figures/inter_topic_ranking_iramuteq.png)

*Classement des topics par séparation inter-topic (IRAMUTEQ).*

**Topics les plus distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| C1: with, and, that, when, they | 0.7653 | 0.6591 |
| C5: luni, slimes, geeked, shawty, sacki | 0.5747 | 0.5816 |
| C2: chen, ekip, ldo, zuukou, etho | 0.5377 | 0.5288 |

**Topics les moins distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| C10: mic, rime, rap, style, mc | 0.2616 | 0.2815 |
| C17: rap, baiser, flow, mec, game | 0.2539 | 0.2683 |
| C16: aller, tess, biff, frérot, hess | 0.2443 | 0.2773 |


#### Test d'indépendance χ²/n (mot × topic)

Le test χ² sur la table de contingence mot × topic mesure à quel point les fréquences de mots dépendent de l'assignation thématique. Un χ²/n plus élevé indique que les topics capturent un vocabulaire plus distinctif. Nous comparons les résultats avec et sans lemmatisation pour évaluer l'impact du prétraitement.

La contribution de chaque topic au χ² total indique à quel point ce topic utilise un vocabulaire spécifique par rapport à la distribution générale du corpus. Les topics avec une contribution élevée sont lexicalement plus distinctifs — ils utilisent des mots que les autres topics n'utilisent pas (ou beaucoup moins). Une distribution uniforme des contributions (chaque topic ≈ 1/K du total) suggère que tous les topics sont également distinctifs.

**Formes de surface (sans lemmatisation)**

| Model | χ² | N (tokens) | χ²/n | Taille vocab. |
|-------|-----|--------|--------|----------|
| BERTOPIC | 5,085,957 | 7,716,683 | 0.6591 | 60,127 |
| LDA | 3,078,063 | 7,716,683 | 0.3989 | 60,127 |
| IRAMUTEQ | 10,306,467 | 7,716,683 | 1.3356 | 60,127 |

**Formes lemmatisées**

| Model | χ² | N (tokens) | χ²/n | Taille vocab. |
|-------|-----|--------|--------|----------|
| BERTOPIC | 4,427,959 | 7,748,855 | 0.5714 | 43,675 |
| LDA | 2,459,692 | 7,748,855 | 0.3174 | 43,675 |
| IRAMUTEQ | 9,556,352 | 7,748,855 | 1.2333 | 43,675 |


## 4. Synthèse et Conclusions

### Principales Conclusions

**Q1 — Accord entre modèles :** L'accord le plus fort est observé entre bertopic vs iramuteq (NMI = 0.1024), tandis que bertopic vs lda montrent l'accord le plus faible (NMI = 0.0150). Ces valeurs modérées à faibles confirment que chaque modèle capture des aspects distincts du corpus.

**Q2 — Séparation des artistes :** IRAMUTEQ capture le mieux les signatures artistiques (V de Cramér = 0.3854, 12.2% de spécialistes).

**Q3 — Dynamique temporelle :** IRAMUTEQ montre la variance temporelle la plus élevée (0.001917), le rendant plus sensible à l'évolution du genre.

**Q4 — Chevauchement lexical :** Le Jaccard vocabulaire complet entre BERTopic et LDA varie de 0.3639 (seuil=1, vocabulaire fonctionnel partagé) à des valeurs plus faibles aux seuils supérieurs (seuil=5 : 0.3904), confirmant la divergence sur le vocabulaire spécialisé.

**Q5 — Homogénéité intra-topic :** IRAMUTEQ présente la meilleure homogénéité lexicale (Labbé = 0.9748), IRAMUTEQ la meilleure homogénéité distributionnelle (JS = 0.8185).

**χ²/n — Dépendance mot-topic :** IRAMUTEQ montre la plus forte association mot-topic (χ²/n = 1.3356), indiquant des topics lexicalement plus distinctifs.

**Complémentarité des approches :** Les trois modèles capturent des aspects distincts du corpus :
- **BERTopic** : similarité sémantique via embeddings neuronaux
- **LDA** : co-occurrences de mots via modèle génératif probabiliste
- **IRAMUTEQ** : mondes lexicaux via classification hiérarchique descendante (ALCESTE)

L'utilisation conjointe de ces trois approches fournit une caractérisation multi-dimensionnelle du corpus, chaque modèle éclairant des facettes complémentaires de la structure thématique.



## 5. Références Méthodologiques

### Métriques d'Accord de Clustering

- Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.
- Strehl, A., & Ghosh, J. (2002). Cluster ensembles: A knowledge reuse framework. Journal of Machine Learning Research, 3, 583-617.
- Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. Journal of Machine Learning Research, 11, 2837-2854.

### Mesures d'Association

- Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.

### Théorie de l'Information

- Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.

### Distance Intertextuelle

- Labbé, D., & Labbé, C. (2001). Inter-textual distance and authorship attribution. Corela : cognition, représentation, langage. Journal of Quantitative Linguistics, 8(3), 213-231.
- Labbé, D., & Monière, D. (2003). Le vocabulaire gouvernemental : Canada, Québec, France (1945-2000). Honoré Champion.
- Labbé, C., & Labbé, D. (2007). Experiments on authorship attribution by intertextual distance in English. Journal of Quantitative Linguistics, 14(1), 33-80.
- IRAMUTEQ implementation: gitlab.huma-num.fr/pratinaud/iramuteq (distance-labbe.R)

### Cohérence des Topics

- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM), 399-408.

### Validation de Clusters

- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.

### Modélisation de Topics

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.
- Reinert, M. (1983). Une méthode de classification descendante hiérarchique : application à l'analyse lexicale par contexte. Les Cahiers de l'Analyse des Données, 8(2), 187-198.


## Annexes

### A. Détails des Runs

- **Timestamp de comparaison:** 2026-03-18 18:20:44
- **Dossier BERTopic:** results/BERTopic/run_20260126_141647_mpnet
- **Dossier LDA:** results/LDA/run_20260224_094852_bigram_only
- **Dossier IRAMUTEQ:** results/IRAMUTEQ/evaluation_20260126_124001

### B. Comparaison Mathématique : Labbé vs Jensen-Shannon

#### Labbé vs Jensen-Shannon : deux regards sur les fréquences

**Différences fondamentales**

Les deux métriques mesurent la similarité lexicale entre textes mais avec des approches différentes.

---

#### Distance de Labbé (algorithme IRAMUTEQ)

La distance de Labbé, implémentée selon l'algorithme original d'IRAMUTEQ, gère explicitement
l'asymétrie de longueur entre textes. L'algorithme procède comme suit :

1. **Identifier** le texte le plus petit ($N_{small}$) et le plus grand ($N_{large}$)
2. **Normaliser** les comptages du texte plus grand : $n'_i = n_i \times U$ où $U = N_{small} / N_{large}$
3. **Calculer** la somme des différences absolues sur les comptages normalisés
4. **Normaliser** par le dénominateur ajusté

$$D_{\text{Labbé}}(A, B) = \frac{\sum_{i=1}^{V} |n_{small}(i) - n'_{large}(i)|}{N_{small} + \sum_{n'_i \geq 1} n'_i}$$

Cette approche est particulièrement adaptée à la comparaison de textes de longueurs différentes.

---

#### Divergence de Jensen-Shannon

Mesure la divergence informationnelle entre les distributions de probabilité.

$$D_{\text{JS}}(A, B) = \frac{1}{2} D_{\text{KL}}(P_A \| M) + \frac{1}{2} D_{\text{KL}}(P_B \| M)$$

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

