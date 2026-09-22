# Notes pour les deux slides

## Slide 1, environ 45 secondes

On entraîne sur les saisons anciennes, on valide sur 2022 et on teste sur 2023 à 2025. Les trois modèles classiques sont proches, autour de 0,80 d'AUC. La régression logistique atteint 0,805 sur le test complet. Les intervalles viennent de 200 rééchantillonnages par joueur, en gardant ensemble ses différentes saisons. La différence avec XGBoost reste incertaine. Toutes les valorisations datées précèdent le 1er juillet. Les saisons suivantes absentes restent inconnues.

## Slide 2, environ 45 secondes

On ajoute le même petit bruit aux mesures continues pour chaque modèle. Le score logistique varie de 0,10 point de pourcentage en moyenne, contre 0,23 pour la forêt et 0,37 pour XGBoost. Il résiste donc mieux à cette perturbation précise. Les essais sur cinq graines donnent de faibles écarts d'AUC. Le PSI mesure séparément le déplacement de distribution. On a aussi réestimé les coefficients logistiques 200 fois. Cela donne des arguments pour garder un modèle simple, à compléter avec l'interprétabilité et la fairness du groupe.

## Questions à préparer

**Pourquoi 10 buts ?** C'est le seuil principal demandé dans le brief. Le taux mesuré est de 17,3 % en entraînement et 16,5 % en test, pas 20 à 25 %. Les variantes 8 et 15 figurent dans `reports/target_sensitivity.csv`. Le 75e percentile des buts d'entraînement est 8. Le « premier quartile des meilleurs buteurs » demande une définition avant de modifier la cible.

**Comment éviter la fuite de valorisation ?** La jointure choisit une date strictement antérieure au 1er juillet. 8 424 valorisations datées sur 8 424 passent le contrôle, et 4 sont manquantes. Un test ajoute des valeurs futures et vérifie qu'elles ne changent pas la jointure. Cela ne certifie pas la validité historique des métadonnées actuelles.

**Un joueur peut-il apparaître en train et test ?** Oui, sur des saisons différentes. C'est cohérent pour prévoir la prochaine saison d'un joueur connu. Le découpage est chronologique. Les erreurs peuvent rester corrélées, d'où le bootstrap par joueur. Une évaluation porte séparément sur les 1 138 lignes de joueurs absents de l'entraînement.

**Les différences par nationalité viennent-elles des postes ?** C'est possible. Le dépôt fournit des taux d'erreur globaux et par poste, avec effectifs et intervalles. Il faut encore examiner la ligue et les autres facteurs. Ces tableaux ne suffisent pas à conclure à une discrimination.

**Retirer la nationalité suffit-il ?** Elle ne figure déjà pas dans les modèles. D'autres variables peuvent lui rester associées. La comparaison avec et sans valorisation est exploratoire et n'identifie pas à elle seule un effet causal.

**Quel modèle gagne ?** TabICL atteint une AUC de 0,811, la logistique 0,805, XGBoost 0,802 et la forêt 0,792. TabICL utilise 1 500 lignes d'entraînement, contre 4 081 pour les modèles classiques principaux. Des références sur les mêmes 1 500 lignes sont fournies. Les petits écarts ne justifient pas un classement absolu.

**Que mesure le bootstrap des coefficients ?** Il rééchantillonne les joueurs d'entraînement et réestime prétraitement et logistique. Les coefficients numériques sont remis dans les unités d'écart-type du premier entraînement. Avec des variables corrélées et une pénalisation, le signe d'un coefficient n'est pas un effet causal. Les intervalles d'AUC utilisent un autre bootstrap, sur le test, en gardant les modèles fixes.

**Que signifie le PSI ?** Il compare les fréquences par intervalles, avec lissage et catégorie pour les valeurs manquantes. Ce n'est pas un test statistique. Les scores d'entraînement sont ajustés sur ces mêmes données, donc on fournit aussi une référence de validation.

**Quelle est la principale limite ?** 19,0 % des lignes éligibles n'ont aucun résultat observé la saison suivante. Les départs hors des neuf ligues peuvent aussi laisser des totaux partiels. On évalue la performance observée dans ce périmètre. Les sélections internationales historiques sont indisponibles.
