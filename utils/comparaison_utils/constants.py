#!/usr/bin/env python3
"""
Constants and Scientific References for Topic Model Comparison
==============================================================
"""

METRIC_REFERENCES = {
    'ari': {
        'name': 'Adjusted Rand Index (ARI)',
        'citation': 'Hubert, L., & Arabie, P. (1985)',
        'paper': 'Comparing partitions. Journal of Classification, 2(1), 193-218.',
        'description': '''The Adjusted Rand Index measures the similarity between two clusterings,
adjusted for chance. It computes the number of pair agreements (both in same cluster or both in
different clusters) normalized by the expected value under a random model. ARI = (RI - Expected_RI) / (Max_RI - Expected_RI).
Range: [-1, 1], where 1 = perfect agreement, 0 = random, <0 = less than random.''',
        'description_fr': '''L'indice de Rand ajusté mesure la similarité entre deux clusterings,
corrigé par le hasard. Il calcule le nombre d'accords de paires (toutes deux dans le même cluster
ou toutes deux dans des clusters différents), normalisé par la valeur attendue sous un modèle aléatoire.
ARI = (RI - RI_attendu) / (RI_max - RI_attendu).
Intervalle : [-1, 1], où 1 = accord parfait, 0 = aléatoire, <0 = inférieur au hasard.''',
        'interpretation': {
            'perfect': 1.0,
            'good': 0.7,
            'moderate': 0.4,
            'poor': 0.0
        }
    },
    'nmi': {
        'name': 'Normalized Mutual Information (NMI)',
        'citation': 'Strehl, A., & Ghosh, J. (2002)',
        'paper': 'Cluster ensembles: A knowledge reuse framework. Journal of Machine Learning Research, 3, 583-617.',
        'description': '''NMI measures the mutual dependence between two clusterings using information
theory. It quantifies how much knowing one clustering reduces uncertainty about the other.
NMI = 2 * I(X;Y) / (H(X) + H(Y)), where I is mutual information and H is entropy.
Range: [0, 1], where 1 = identical clusterings, 0 = independent.''',
        'description_fr': '''La NMI mesure la dépendance mutuelle entre deux clusterings en utilisant la
théorie de l'information. Elle quantifie dans quelle mesure la connaissance d'un clustering réduit
l'incertitude sur l'autre. NMI = 2 × I(X;Y) / (H(X) + H(Y)), où I est l'information mutuelle et H l'entropie.
Intervalle : [0, 1], où 1 = clusterings identiques, 0 = indépendants.''',
        'interpretation': {
            'perfect': 1.0,
            'good': 0.5,
            'moderate': 0.3,
            'poor': 0.1
        }
    },
    'ami': {
        'name': 'Adjusted Mutual Information (AMI)',
        'citation': 'Vinh, N. X., Epps, J., & Bailey, J. (2010)',
        'paper': 'Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. Journal of Machine Learning Research, 11, 2837-2854.',
        'description': '''AMI extends NMI by adjusting for chance agreement, similar to ARI.
AMI = (MI - E[MI]) / (max(H(X), H(Y)) - E[MI]). This correction is important when
comparing clusterings with different numbers of clusters.
Range: [-1, 1], with similar interpretation to ARI.''',
        'description_fr': '''L'AMI étend la NMI en corrigeant pour l'accord dû au hasard, de manière
similaire à l'ARI. AMI = (MI - E[MI]) / (max(H(X), H(Y)) - E[MI]). Cette correction est
importante lorsqu'on compare des clusterings avec des nombres de clusters différents.
Intervalle : [-1, 1], avec une interprétation similaire à l'ARI.''',
        'interpretation': {
            'perfect': 1.0,
            'good': 0.5,
            'moderate': 0.3,
            'poor': 0.0
        }
    },
    'cramers_v': {
        'name': "V de Cramér",
        'citation': 'Cramér, H. (1946)',
        'paper': 'Mathematical Methods of Statistics. Princeton University Press.',
        'description': '''Cramér's V measures the strength of association between two categorical
variables. It is derived from the chi-square statistic: V = sqrt(χ² / (n × min(k-1, r-1))),
where k and r are the number of categories. V normalizes the chi-square by sample size
and dimensionality, allowing comparison across tables of different sizes.
Range: [0, 1], where 0 = no association, 1 = perfect association.''',
        'description_fr': '''Le V de Cramér mesure la force d'association entre deux variables catégorielles.
Il est dérivé de la statistique du chi-deux : V = √(χ² / (n × min(k-1, r-1))),
où k et r sont le nombre de catégories. V normalise le chi-deux par la taille de l'échantillon
et la dimensionnalité, permettant la comparaison entre tables de tailles différentes.
Intervalle : [0, 1], où 0 = aucune association, 1 = association parfaite.''',
        'interpretation': {
            'strong': 0.5,
            'moderate': 0.3,
            'weak': 0.1,
            'none': 0.0
        }
    },
    'js_divergence': {
        'name': 'Distance de Jensen-Shannon',
        'citation': 'Lin, J. (1991)',
        'paper': 'Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.',
        'description': '''JS distance is the square root of JS divergence, a symmetric measure of
similarity between probability distributions. scipy.spatial.distance.jensenshannon() returns
this distance value directly. JS distance is a proper metric satisfying the triangle inequality.
Range: [0, 1], where 0 = identical distributions, 1 = maximally different.''',
        'description_fr': '''La distance JS est la racine carrée de la divergence JS, une mesure
symétrique de similarité entre distributions de probabilité. scipy.spatial.distance.jensenshannon()
retourne directement cette valeur de distance. La distance JS est une métrique propre satisfaisant
l'inégalité triangulaire.
Intervalle : [0, 1], où 0 = distributions identiques, 1 = maximalement différentes.''',
        'interpretation': {
            'very_different': 0.5,
            'different': 0.3,
            'similar': 0.1,
            'identical': 0.0
        }
    },
    'labbe_distance': {
        'name': 'Distance intertextuelle de Labbé',
        'citation': 'Labbé, D., & Labbé, C. (2001)',
        'paper': 'Inter-textual distance and authorship attribution. Corela : cognition, représentation, langage. Journal of Quantitative Linguistics, 8(3), 213-231.',
        'description': '''The Labbé distance measures lexical similarity between two texts based on
relative word frequencies. It computes the sum of absolute differences in word frequencies:
D(A, B) = 0.5 × Σ|f_A(w) - f_B(w)|, where f_A(w) and f_B(w) are relative frequencies of word w.
This metric is the standard in French stylometry and JADT community for authorship attribution
and textual analysis. It captures lexical homogeneity independently of semantic content.
Range: [0, 1], where 0 = identical vocabularies, 1 = no overlap.''',
        'description_fr': '''La distance de Labbé mesure la similarité lexicale entre deux textes à partir
des fréquences relatives des mots. Elle calcule la somme des différences absolues des fréquences :
D(A, B) = 0.5 × Σ|f_A(w) - f_B(w)|, où f_A(w) et f_B(w) sont les fréquences relatives du mot w.
Cette métrique est le standard en stylométrie française et dans la communauté JADT pour l'attribution
d'auteur et l'analyse textuelle. Elle capture l'homogénéité lexicale indépendamment du contenu sémantique.
Intervalle : [0, 1], où 0 = vocabulaires identiques, 1 = aucun chevauchement.''',
        'interpretation': {
            'very_similar': 0.2,
            'similar': 0.4,
            'different': 0.6,
            'very_different': 0.8
        },
        'additional_refs': [
            'Labbé, D., & Monière, D. (2003). Le vocabulaire gouvernemental : Canada, Québec, France (1945-2000). Honoré Champion.',
            'Labbé, C., & Labbé, D. (2007). Experiments on authorship attribution by intertextual distance in English. Journal of Quantitative Linguistics, 14(1), 33-80.'
        ]
    },
    'coherence_cv': {
        'name': 'Score de cohérence (C_v)',
        'citation': 'Röder, M., Both, A., & Hinneburg, A. (2015)',
        'paper': 'Exploring the space of topic coherence measures. In Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM), 399-408.',
        'description': '''C_v coherence combines normalized pointwise mutual information (NPMI)
with cosine similarity of word vectors. It correlates well with human judgment of topic
quality. Higher scores indicate more interpretable topics where top words are semantically related.
Range: typically [0, 1], higher is better.''',
        'description_fr': '''La cohérence C_v combine l'information mutuelle ponctuelle normalisée (NPMI)
avec la similarité cosinus des vecteurs de mots. Elle corrèle bien avec le jugement humain de la
qualité des topics. Des scores plus élevés indiquent des topics plus interprétables dont les mots
principaux sont sémantiquement liés.
Intervalle : typiquement [0, 1], plus élevé = meilleur.''',
        'interpretation': {
            'excellent': 0.7,
            'good': 0.5,
            'moderate': 0.4,
            'poor': 0.3
        }
    },
    'silhouette': {
        'name': 'Score de silhouette',
        'citation': 'Rousseeuw, P. J. (1987)',
        'paper': 'Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.',
        'description': '''The silhouette score measures how similar a point is to its own cluster
compared to other clusters. s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is the mean
intra-cluster distance and b(i) is the mean nearest-cluster distance.
Range: [-1, 1], where 1 = well-clustered, 0 = on boundary, -1 = misclassified.''',
        'description_fr': '''Le score de silhouette mesure la similarité d'un point avec son propre cluster
par rapport aux autres clusters. s(i) = (b(i) - a(i)) / max(a(i), b(i)), où a(i) est la distance
moyenne intra-cluster et b(i) la distance moyenne au cluster le plus proche.
Intervalle : [-1, 1], où 1 = bien clusterisé, 0 = sur la frontière, -1 = mal classé.''',
        'interpretation': {
            'strong': 0.7,
            'reasonable': 0.5,
            'weak': 0.25,
            'poor': 0.0
        }
    }
}
