# Rapport de Comparaison des Modèles de Topics

**Généré le:** 2026-07-06 22:37:13

**Répertoire de sortie:** `/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/results/comparisons/comparison_20260706_222138`

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

**Dossier du run:** `results/BERTopic/run_20260706_220123_solon`

BERTopic (Grootendorst, 2022) est un modèle de topics neuronal utilisant des embeddings de transformers pour la représentation des documents, UMAP pour la réduction de dimensionnalité, et le clustering. Les topics sont représentés via pondération c-TF-IDF, avec labellisation optionnelle par OpenAI et KeyBERT.

#### Paramètres

| Paramètre | Valeur |
|-----------|-------|
| embedding_model | `OrdalieTech/Solon-embeddings-large-0.1` |
| embedding_key | `solon` |
| clustering_algorithm | `kmeans` |
| n_clusters | `20` |
| hdbscan_params | `None` |
| agglomerative_params | `None` |
| umap_params | n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine, random_state=42 |
| num_words_per_topic | `30` |
| use_openai | `True` |
| include_keybert | `False` |
| interactive_html | `True` |

#### Qualité du Clustering

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Silhouette (UMAP) | 0.2303 | Modéré |

*Le score de silhouette (Rousseeuw, 1987) mesure la séparation des clusters.*

![Silhouette](figures/bertopic_silhouette_plot.png)


#### Distribution des Topics

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Nombre de topics | 20 | - |
| Ratio d'imbalance | 147.47 | Très déséquilibré |
| Entropie de distribution | 0.976 | Quasi uniforme |

**Définitions des métriques :**

- **Ratio d'imbalance** = max(compte_topic) / min(compte_topic). Mesure l'inégalité des tailles de topics.

- **Entropie de distribution** (normalisée) = -Σ(p_i × log(p_i)) / log(n_topics). Intervalle [0,1] : 1 = uniforme.

![Topic Distribution](figures/bertopic_topic_distribution.png)

*Distribution des documents par topic.*


#### Séparation des Artistes

| Métrique | Valeur | Description |
|--------|-------|-------------|
| % Spécialistes | 6.6% | Artistes avec >50% dans un topic |
| % Modérés | 20.6% | Artistes avec 25-50% dans le topic dominant |
| % Généralistes | 72.8% | Artistes répartis sur plusieurs topics |
| Indice de spécialisation | 0.240 | Concentration moyenne |
| Divergence JS | 0.513 | Divergence des profils d'artistes |

![Artist-Topic Heatmap](figures/bertopic_artist_topics_heatmap.png)

*Heatmap de distribution des artistes par topic.*

![Artist Specialization](figures/bertopic_artist_specialization.png)

*Profils de spécialisation des artistes.*


#### Dynamique Temporelle

| Métrique | Valeur |
|--------|-------|
| Variance moyenne des topics | 0.000770 |
| JS annuel moyen | N/A |

![Annual JS](figures/bertopic_annual_js_divergence.png)

*Divergence JS entre années consécutives.*

![Year-Topic Heatmap](figures/bertopic_year_topic_heatmap.png)

*Évolution des topics dans le temps.*


#### Vue d'ensemble des Topics

**Topic 0** — *“Solitude et Quête d’Identité”*

- **c-TF-IDF:** je, de, la, le, que, est, ai, et, les, qu
- **KeyBERT (0 terms):** 

**Topic 1** — *Réalité de la Rue et Résistance*

- **c-TF-IDF:** la, les, le, on, est, dans, pas, de, des, tu
- **KeyBERT (0 terms):** 

**Topic 2** — *Vie de Rue et Rébellion Urbaine*

- **c-TF-IDF:** la, dans, le, suis, pas, est, les, on, en, ai
- **KeyBERT (0 terms):** 

**Topic 3** — *Authenticité et Lutte dans le Rap*

- **c-TF-IDF:** le, de, est, les, la, rap, et, pas, un, on
- **KeyBERT (0 terms):** 

**Topic 4** — *Vengeance, Trahison et Rébellion Urbaine*

- **c-TF-IDF:** les, la, de, est, le, pas, un, des, tu, comme
- **KeyBERT (0 terms):** 

**Topic 5** — *Amour, Rupture et Solitude*

- **c-TF-IDF:** je, tu, moi, que, pas, toi, ai, est, qu, me
- **KeyBERT (0 terms):** 

**Topic 6** — *Identité et Lutte Sociale*

- **c-TF-IDF:** de, les, la, on, est, le, et, des, qu, que
- **KeyBERT (0 terms):** 

**Topic 7** — *Résilience face à la Violence et l'Injustice*

- **c-TF-IDF:** de, la, le, les, et, des, un, est, dans, que
- **KeyBERT (0 terms):** 

**Topic 8** — *Quête de Richesse et Détresse Urbaine*

- **c-TF-IDF:** la, le, on, les, pas, est, dans, de, ai, des
- **KeyBERT (0 terms):** 

**Topic 9** — *Titre du thème : "Lutte et Résilience dans la Rue*

- **c-TF-IDF:** de, la, est, pas, les, le, on, ai, qu, et
- **KeyBERT (0 terms):** 

**Topic 10** — *Résilience et Rébellion dans la Rue*

- **c-TF-IDF:** pas, la, les, est, tu, on, ai, de, qu, le
- **KeyBERT (0 terms):** 

**Topic 11** — *Réflexion sur la Vie de Rue*

- **c-TF-IDF:** la, les, de, est, le, on, des, pas, dans, et
- **KeyBERT (0 terms):** 

**Topic 12** — *Résilience et Révolte des Quartiers*

- **c-TF-IDF:** les, la, de, est, on, le, des, pas, et, dans
- **KeyBERT (0 terms):** 

**Topic 13** — *Vie de Rue et Évasion Urbaine*

- **c-TF-IDF:** elle, la, est, pas, tu, le, dans, les, de, je
- **KeyBERT (0 terms):** 

**Topic 14** — *Trahison et Désillusion dans les Relations*

- **c-TF-IDF:** pas, je, ai, que, est, de, la, qu, tu, le
- **KeyBERT (0 terms):** 


... et 5 autres topics

![UMAP](figures/bertopic_umap_topics.png)

*Projection UMAP des embeddings colorés par topic.*


### 2.2 LDA

**Dossier du run:** `results/LDA/run_20260706_171018_both`

Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003) est un modèle génératif probabiliste représentant les documents comme mélanges de topics, où chaque topic est une distribution sur les mots. Cette implémentation utilise Gensim avec prétraitement n-gram.

#### Paramètres

| Paramètre | Valeur |
|-----------|-------|
| backend | `tomotopy` |
| num_topics | `20` |
| alpha | `optimized` |
| eta | `0.01 (fixed)` |
| learned_alpha | `[0.1164388507604599, 0.07838124781847, 0.30533406138420105, 0.08740535378456116, 0.030320167541503906, 0.2100149691104889, 0.2892478108406067, 0.12688910961151123, 0.22516897320747375, 0.3718010187149048, 0.1383122354745865, 0.1755695641040802, 0.2608986794948578, 0.09985540807247162, 0.09110520035028458, 0.2588096261024475, 0.2526785731315613, 0.3634151220321655, 0.1991192102432251, 0.20089060068130493]` |
| passes | `15` |
| iterations | `400` |
| gibbs_iterations | `1000` |
| min_word_len | `2` |
| min_doc_freq | `5` |
| max_doc_freq_ratio | `0.5` |
| use_ngrams | `both` |
| ngram_min_count | `10` |
| ngram_threshold | `50` |
| legacy_stopwords | `False` |
| dedup | `False` |
| dedup_method | `None` |
| num_docs_before_dedup | `115805` |
| num_docs_removed_dedup | `0` |
| num_words_per_topic | `30` |
| keep_all_the_document | `True` |

#### Scores de Cohérence

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Cohérence C_v | 0.4156 | Modéré |
| Cohérence UMass | -2.6848 | Modéré |

*La cohérence C_v (Röder et al., 2015) mesure la cohérence sémantique des mots des topics.*

![Coherence](figures/lda_coherence_plot.png)


#### Distribution des Topics

| Métrique | Valeur | Interprétation |
|--------|-------|----------------|
| Nombre de topics | 20 | - |
| Ratio d'imbalance | 9.85 | Déséquilibré |
| Entropie de distribution | 0.954 | Quasi uniforme |

**Définitions des métriques :**

- **Ratio d'imbalance** = max(compte_topic) / min(compte_topic). Mesure l'inégalité des tailles de topics.

- **Entropie de distribution** (normalisée) = -Σ(p_i × log(p_i)) / log(n_topics). Intervalle [0,1] : 1 = uniforme.

![Topic Distribution](figures/lda_topic_distribution.png)

*Distribution des documents par topic.*


#### Séparation des Artistes

| Métrique | Valeur | Description |
|--------|-------|-------------|
| % Spécialistes | 9.7% | Artistes avec >50% dans un topic |
| % Modérés | 33.4% | Artistes avec 25-50% dans le topic dominant |
| % Généralistes | 56.9% | Artistes répartis sur plusieurs topics |
| Indice de spécialisation | 0.294 | Concentration moyenne |
| Divergence JS | 0.323 | Divergence des profils d'artistes |

![Artist-Topic Heatmap](figures/lda_artist_topics_heatmap.png)

*Heatmap de distribution des artistes par topic.*

![Artist Specialization](figures/lda_artist_specialization.png)

*Profils de spécialisation des artistes.*


#### Dynamique Temporelle

| Métrique | Valeur |
|--------|-------|
| Variance moyenne des topics | 0.000457 |
| JS annuel moyen | 0.0524 |

![Annual JS](figures/lda_annual_js_divergence.png)

*Divergence JS entre années consécutives.*

![Year-Topic Heatmap](figures/lda_year_topic_heatmap.png)

*Évolution des topics dans le temps.*


#### Vue d'ensemble des Topics

| Topic | Mots clés |
|-------|----------|
| 0 | qu'elle, j'aime, belle, soir, bonne, danse, bébé, aime, sais, fille |
| 1 | négro, négros, jeune, j'veux, j'arrive, noir, pétasse, j'fais, black, toujours |
| 2 | qu'on, gens, qu'il, jamais, besoin, vie, sais, toujours, qu'ils, vrai |
| 3 | qu'on, comment, bon, vas-y, qu'est-ce, gars, sais, monde, allez, fou |
| 4 | i'm, your, but, it's, just, don't, night, now, time, when |
| 5 | rue, street, frères, mecs, ghetto, parle, baise, tess, keufs, ici |
| 6 | vite, ici, tête, coup, merde, passe, gars, qu'on, rue, place |
| 7 | deux, d'la, tête, soir, j'fais, j'me, fume, roule, trois, beuh |
| 8 | fin, monde, deux, grand, nom, terre, corps, sens, puis, aucun |
| 9 | d'la, j'me, j'veux, j'fais, j'vais, c'que, j'sais, vie, j'vois, qu'on |
| 10 | vie, nuit, loin, jour, mort, seul, ville, encore, soir, noir |
| 11 | france, monde, qu'on, guerre, frères, ici, pays, jeunes, qu'ils, haine |
| 12 | pute, j'te, putain, gueule, cul, sale, vas, baise, mère, chatte |
| 13 | zone, marseille, j'fais, quartier, nique, j'vais, jaloux, vois, mets, sang |
| 14 | pourquoi, cœur, bébé, j'te, sais, dis-moi, j'veux, jamais, j'vais, mama |

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
| bertopic_vs_lda | 0.0746 | 0.1618 | Accord faible |
| bertopic_vs_iramuteq | 0.0857 | 0.1769 | Accord faible |
| lda_vs_iramuteq | 0.1056 | 0.2050 | Accord faible |

**Observations clés :**

1. **Meilleur accord :** lda_vs_iramuteq (NMI = 0.2050)

2. **Accord le plus faible :** bertopic_vs_lda (NMI = 0.1618)

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
| BERTOPIC | 0.2593 | Association modérée |
| LDA | 0.3138 | Association forte |
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
| BERTOPIC | 0.000794 | 3 | 0.004110 | Stable |
| LDA | 0.000472 | 18 | 0.001878 | Stable |
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
| BERTOPIC | 0.4025 | Chevauchement significatif |
| LDA | 0.9476 | Topics très distincts |
| IRAMUTEQ | 0.9964 | Topics très distincts |

**Observations clés :**

1. **LDA et IRAMUTEQ** montrent une haute distinctivité (>0.9).

2. **BERTopic** peut montrer une distinctivité plus faible (embeddings sémantiques).

#### Chevauchement Lexical Inter-Modèles (Vocabulaire Complet)

Pour évaluer le recouvrement lexical entre les topics correspondants de BERTopic et LDA, nous calculons l'indice de Jaccard sur le **vocabulaire complet** des documents assignés à chaque topic (et non sur les seuls mots représentatifs extraits par c-TF-IDF ou probabilité). Nous faisons varier le seuil de fréquence minimale pour distinguer le vocabulaire fonctionnel partagé (seuil bas, Jaccard élevé) du vocabulaire thématique spécifique (seuil élevé, Jaccard plus faible). Un Jaccard décroissant avec le seuil indique que les modèles divergent sur les termes spécialisés tout en partageant le socle lexical commun.

| Seuil min. freq. | Jaccard moyen | Paires |
|-----|------|------|
| 1 | 0.4412 | 20 |
| 5 | 0.5046 | 20 |
| 20 | 0.5031 | 20 |

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
| BERTOPIC | 0.8171 | 0.0116 | 20 | Très hétérogène |
| LDA | 0.8200 | 0.0028 | 20 | Très hétérogène |
| IRAMUTEQ | 0.8185 | 0.0069 | 20 | Très hétérogène |

**Distance de Labbé (Lexicale)**

| Modèle | Distance Moyenne | Écart-type | Topics | Interprétation |
|-------|---------------|---------|----------|----------------|
| BERTOPIC | 0.9738 | 0.0142 | 20 | Très hétérogène |
| LDA | 0.9776 | 0.0052 | 20 | Très hétérogène |
| IRAMUTEQ | 0.9748 | 0.0131 | 20 | Très hétérogène |

**Observations clés :**

1. **Meilleure homogénéité distributionnelle (JS) :** BERTOPIC montre la distance moyenne la plus faible (0.8171).

2. **Meilleure homogénéité lexicale (Labbé) :** BERTOPIC montre la distance moyenne la plus faible (0.9738).

3. **Complémentarité :** JS capture la similarité distributionnelle, Labbé le chevauchement lexical absolu.

#### Analyse par Topic

Top 5 topics les plus et les moins homogènes (distance JS) :

**BERTOPIC**

*Topics les plus homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 19 | 0.7672 | 55 |
| 9 | 0.8158 | 5885 |
| 5 | 0.8166 | 7417 |
| 14 | 0.8167 | 4800 |
| 3 | 0.8171 | 7832 |

*Topics les moins homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 8 | 0.8215 | 5951 |
| 4 | 0.8219 | 7615 |
| 1 | 0.8221 | 7922 |
| 13 | 0.8222 | 5068 |
| 2 | 0.8232 | 7916 |

**LDA**

*Topics les plus homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 4 | 0.8141 | 1318 |
| 9 | 0.8153 | 12145 |
| 16 | 0.8158 | 7067 |
| 14 | 0.8170 | 3150 |
| 2 | 0.8181 | 7835 |

*Topics les moins homogènes :*

| Topic | Distance JS Moyenne | Documents |
|-------|------------------|-------------|
| 19 | 0.8219 | 9442 |
| 3 | 0.8221 | 2230 |
| 1 | 0.8223 | 2608 |
| 7 | 0.8243 | 3265 |
| 8 | 0.8251 | 4297 |

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
| BERTOPIC | 0.8171 | 0.9738 |
| LDA | 0.8200 | 0.9776 |
| IRAMUTEQ | 0.8185 | 0.9748 |

**Inter-topic (paires)**

*Distances entre documents du topic et documents hors du topic. Mesure la **séparation** : des distances élevées indiquent des topics bien distincts.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.8221 | 0.9816 |
| LDA | 0.8228 | 0.9828 |
| IRAMUTEQ | 0.8229 | 0.9832 |

**Intra-topic (agrégé) (n=20)**

*Comme intra-paires, mais en agrégeant n couplets ensemble. Réduit la sensibilité de Labbé aux différences de longueur.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.6919 | 0.7435 |
| LDA | 0.7085 | 0.7658 |
| IRAMUTEQ | 0.7022 | 0.7543 |

**Inter-topic (agrégé) (n=20)**

*Comme inter-paires, mais avec documents agrégés. Plus robuste pour les comparaisons de séparation.*

| Modèle | JS | Labbé |
|-------|------|-------|
| BERTOPIC | 0.7289 | 0.8018 |
| LDA | 0.7332 | 0.8110 |
| IRAMUTEQ | 0.7345 | 0.8132 |

**Synthèse des 4 configurations :**

- **Meilleure homogénéité (JS):** BERTOPIC (0.8171)
- **Meilleure homogénéité (Labbé):** BERTOPIC (0.9738)
- **Meilleure séparation (JS):** IRAMUTEQ (0.8229)
- **Meilleure séparation (Labbé):** IRAMUTEQ (0.9832)


#### Stabilisation de la distance de Labbé par agrégation

Cette analyse montre comment la distance de Labbé évolue en fonction du nombre de couplets agrégés. La distance de Labbé étant sensible à la longueur des textes, l'agrégation de plusieurs couplets produit des unités textuelles plus comparables et des distances plus stables.

**Plage d'agrégation :** de 8 à 11 documents (>500 mots/unité, ≥5 unités/topic), 2 points. Taille minimale de topic (tous modèles) : 55 documents.

![Aggregation Curve](figures/aggregation_curve.png)

*Évolution de la distance de Labbé intra-topic (gauche) et inter-topic (droite) en fonction de la taille d'agrégation.*


#### Classement de la séparation inter-topic

Pour chaque modèle, les topics sont classés par leur distance inter-topic moyenne (un-contre-reste). Des distances plus élevées indiquent des topics lexicalement plus distincts du reste du corpus.

![BERTOPIC Ranking](figures/inter_topic_ranking_bertopic.png)

*Classement des topics par séparation inter-topic (BERTOPIC).*

**Topics les plus distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T19: Quête d'Amour et Lutte Personnelle | 0.8264 | 0.7349 |
| T5: Amour, Rupture et Solitude | 0.3963 | 0.3860 |
| T15: Relations Complexes et Désillusions Amoureuses | 0.3400 | 0.3574 |

**Topics les moins distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T10: Résilience et Rébellion dans la Rue | 0.2082 | 0.2563 |
| T11: Réflexion sur la Vie de Rue | 0.2073 | 0.2528 |
| T16: Des Cendres à l'Ambition : Luttes et Résilience | 0.2011 | 0.2522 |

![LDA Ranking](figures/inter_topic_ranking_lda.png)

*Classement des topics par séparation inter-topic (LDA).*

**Topics les plus distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T4: i'm, your, but, it's, just | 0.6490 | 0.5769 |
| T14: pourquoi, cœur, bébé, j'te, sais | 0.4290 | 0.4311 |
| T1: négro, négros, jeune, j'veux, j'arrive | 0.4183 | 0.4185 |

**Topics les moins distincts :**

| Topic | Distance Labbé moy. | Distance JS moy. |
|-------|-----------------|----------------|
| T6: vite, ici, tête, coup, merde | 0.2367 | 0.2603 |
| T2: qu'on, gens, qu'il, jamais, besoin | 0.2362 | 0.2611 |
| T9: d'la, j'me, j'veux, j'fais, j'vais | 0.1682 | 0.2086 |

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
| BERTOPIC | 6,311,868 | 7,716,683 | 0.8180 | 60,127 |
| LDA | 8,751,198 | 7,716,683 | 1.1341 | 60,127 |
| IRAMUTEQ | 10,306,467 | 7,716,683 | 1.3356 | 60,127 |

**Formes lemmatisées**

| Model | χ² | N (tokens) | χ²/n | Taille vocab. |
|-------|-----|--------|--------|----------|
| BERTOPIC | 5,457,777 | 7,748,855 | 0.7043 | 43,675 |
| LDA | 7,736,897 | 7,748,855 | 0.9985 | 43,675 |
| IRAMUTEQ | 9,556,352 | 7,748,855 | 1.2333 | 43,675 |


### Q5-sem — Séparation sémantique (espace d'embeddings)

Contrepartie de la Q5 lexicale dans l'espace des embeddings de phrases : mêmes tailles d'agrégation, même échantillonnage de paires (graine 42), la distance cosinus remplaçant la distance de Labbé. Ces métriques sont **alignées par construction** avec les méthodes à base d'embeddings — tout comme le χ²/n l'est avec le critère de Reinert — et sont rapportées par souci de transparence, non comme arbitre de la qualité globale. Non-circularité : chaque partition est évaluée dans des espaces d'embeddings *différents* de celui sur lequel BERTopic a été entraîné (« solon »).

**SR lexical (Labbé)**

| Model | n=8 | n=11 |
|-------|------|------|
| BERTOPIC | 1.0436 | 1.0546 |
| LDA | 1.0318 | 1.0394 |
| IRAMUTEQ | 1.0434 | 1.0536 |


#### Espace : `e5`

![Semantic aggregation curve — e5](figures/aggregation_curve_semantic_e5.png)

*Qualité de clustering (indices internes). Silhouette cosinus sur échantillon stratifié ; Calinski-Harabasz et Davies-Bouldin sur la matrice complète.*

| Model | Silhouette (cos) | Calinski-Harabasz | Davies-Bouldin |
|-------|------------------|-------------------|----------------|
| BERTOPIC | -0.0282 | 277.1 | 11.2324 |
| LDA | -0.0276 | 219.4 | 11.2640 |
| IRAMUTEQ | -0.0307 | 204.3 | 14.2574 |

*Ratio de séparation SR = inter/intra (agrégation n=11). SR > 1 : distances entre classes supérieures aux distances intra-classe.*

| Model | n=8 | n=11 |
|-------|------|------|
| BERTOPIC | 1.4018 | 1.5430 |
| LDA | 1.3052 | 1.4196 |
| IRAMUTEQ | 1.3071 | 1.4313 |


#### Espace : `camembert`

![Semantic aggregation curve — camembert](figures/aggregation_curve_semantic_camembert.png)

*Qualité de clustering (indices internes). Silhouette cosinus sur échantillon stratifié ; Calinski-Harabasz et Davies-Bouldin sur la matrice complète.*

| Model | Silhouette (cos) | Calinski-Harabasz | Davies-Bouldin |
|-------|------------------|-------------------|----------------|
| BERTOPIC | -0.0314 | 421.9 | 10.1078 |
| LDA | -0.0373 | 367.3 | 10.1343 |
| IRAMUTEQ | -0.0295 | 326.8 | 13.5102 |

*Ratio de séparation SR = inter/intra (agrégation n=11). SR > 1 : distances entre classes supérieures aux distances intra-classe.*

| Model | n=8 | n=11 |
|-------|------|------|
| BERTOPIC | 1.5920 | 1.8234 |
| LDA | 1.5057 | 1.6948 |
| IRAMUTEQ | 1.5071 | 1.6958 |


#### Espace : `solon` — **circulaire pour BERTopic**

![Semantic aggregation curve — solon](figures/aggregation_curve_semantic_solon.png)

*Qualité de clustering (indices internes). Silhouette cosinus sur échantillon stratifié ; Calinski-Harabasz et Davies-Bouldin sur la matrice complète.*

| Model | Silhouette (cos) | Calinski-Harabasz | Davies-Bouldin |
|-------|------------------|-------------------|----------------|
| BERTOPIC | -0.0099 | 377.2 | 8.9252 |
| LDA | -0.0219 | 226.7 | 11.6015 |
| IRAMUTEQ | -0.0173 | 219.7 | 13.6701 |

*Ratio de séparation SR = inter/intra (agrégation n=11). SR > 1 : distances entre classes supérieures aux distances intra-classe.*

| Model | n=8 | n=11 |
|-------|------|------|
| BERTOPIC | 1.5365 | 1.7209 |
| LDA | 1.2935 | 1.4068 |
| IRAMUTEQ | 1.3158 | 1.4376 |


## 4. Synthèse et Conclusions

### Principales Conclusions

**Q1 — Accord entre modèles :** L'accord le plus fort est observé entre lda vs iramuteq (NMI = 0.2050), tandis que bertopic vs lda montrent l'accord le plus faible (NMI = 0.1618). Ces valeurs modérées à faibles confirment que chaque modèle capture des aspects distincts du corpus.

**Q2 — Séparation des artistes :** IRAMUTEQ capture le mieux les signatures artistiques (V de Cramér = 0.3854, 12.2% de spécialistes).

**Q3 — Dynamique temporelle :** IRAMUTEQ montre la variance temporelle la plus élevée (0.001917), le rendant plus sensible à l'évolution du genre.

**Q4 — Chevauchement lexical :** Le Jaccard vocabulaire complet entre BERTopic et LDA varie de 0.4412 (seuil=1, vocabulaire fonctionnel partagé) à des valeurs plus faibles aux seuils supérieurs (seuil=5 : 0.5046), confirmant la divergence sur le vocabulaire spécialisé.

**Q5 — Homogénéité intra-topic :** BERTOPIC présente la meilleure homogénéité lexicale (Labbé = 0.9738), BERTOPIC la meilleure homogénéité distributionnelle (JS = 0.8171).

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

- **Timestamp de comparaison:** 2026-07-06 22:37:13
- **Dossier BERTopic:** results/BERTopic/run_20260706_220123_solon
- **Dossier LDA:** results/LDA/run_20260706_171018_both
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

