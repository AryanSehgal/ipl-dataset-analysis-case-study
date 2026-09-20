# IPL Data Analysis (2008–2024) 🏏

## Problem Statement

This project analyzes 17 seasons of Indian Premier League (IPL) match data to uncover patterns around toss impact, venue advantage, and team performance, and to build predictive models for match outcomes, player-of-the-match awards, and toss-decision strategy — turning historical match data into data-driven insights for teams and analysts.

## Dataset

**Source:** [IPL Complete Dataset 2008–2020 — Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020?select=matches.csv) (`matches.csv`) — data spans 2008–2024

The dataset contains **1,095 matches** described across **20 features**:

| Feature | Description |
|---|---|
| id | Unique match identifier |
| season | IPL season (year) |
| city / venue | Location of the match |
| date | Match date |
| match_type | League, Playoff, Final, etc. |
| team1 / team2 | Competing teams |
| toss_winner / toss_decision | Toss outcome and decision (bat/field) |
| winner / result / result_margin | Match outcome and margin |
| target_runs / target_overs | Chase target details |
| super_over | Whether the match went to a super over |
| method | Result method (e.g., Duckworth–Lewis) |
| player_of_match | Player of the match award |
| umpire1 / umpire2 | On-field umpires |

### Data Quality Notes
- 5 matches were abandoned mid-game (no winner or player-of-the-match recorded).
- 51 matches (≈4.6%) had missing `city` values — recovered via a venue → city mapping, since venue had no nulls.
- 19 matches have no `result_margin` (5 abandoned + 14 decided by super over).
- 21 matches were decided via the Duckworth–Lewis (D/L) method; 9 additional matches had reduced overs due to rain but completed normally.

## Approach

### 1. Data Cleaning & Preprocessing
- **Team name normalization:** merged historical renamed franchises (e.g., "Delhi Daredevils" → "Delhi Capitals", "Kings XI Punjab" → "Punjab Kings", "Royal Challengers Bangalore" → "Royal Challengers Bengaluru", "Rising Pune Supergiants" → "Rising Pune Supergiant").
- **Venue de-duplication:** consolidated venues referring to the same stadium under different names (e.g., "Feroz Shah Kotla" → "Arun Jaitley Stadium").
- **Missing city imputation:** filled via venue-to-city mapping.
- Dropped `season` as redundant, since it is fully derivable from `date`.

### 2. Exploratory & Bivariate Analysis
- Toss impact on match outcome.
- Team win percentage by venue.
- Win percentage after winning vs. losing the toss (by team).
- Win percentage by batting-first vs. fielding-first choice, per venue.

### 3. Team Performance Clustering (K-Means)
- Clustered teams on performance metrics using K-Means (K=3, selected via elbow method + silhouette score).
- Identified three clusters: consistently strong teams (e.g., Mumbai Indians, Chennai Super Kings), weaker/struggling teams, and discontinued franchises.

### 4. Match Outcome Prediction — Logistic Regression
- Binary target: `team1_win` (1 if team1 won, 0 otherwise).
- Excluded post-match/leakage features (`winner`, `result`, `result_margin`, `player_of_match`) and redundant `city` (captured via venue).
- Compared L1-regularized, L2-regularized, and unregularized logistic regression.

### 5. Performance Trend Analysis
- Modeled each team's win-rate trend over time via slope of performance.
- Positive slope → improving team, negative → declining, near-zero → stable.

### 6. Player-of-the-Match Prediction — KNN
- Engineered match-context features and built a KNN model to predict top-k candidates for player of the match.
- Tuned `k` by validation accuracy.

### 7. Toss Decision Strategy — Decision Tree
- Built an interpretable decision tree to recommend bat/field-first decisions using venue batting bias, weather (humidity, temperature), and relative team strength.

### 8. Advanced Match Prediction — Ensemble Methods
- Pipeline: Feature Engineering → Random Forest → AdaBoost / XGBoost → Voting Ensemble → Model Comparison.
- Compared bagging (Random Forest) vs. boosting (AdaBoost, XGBoost) vs. a Voting Ensemble.

## Results

### Logistic Regression (Match Outcome)
| Variant | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| L1 Regularization | 0.583 | 0.599 | 0.612 |
| L2 Regularization | 0.555 | 0.576 | 0.593 |
| No Regularization | 0.555 | 0.569 | 0.572 |

### Ensemble Methods (Match Outcome)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.532 | 0.542 | 0.523 | 0.532 |
| AdaBoost | 0.518 | 0.528 | 0.514 | 0.521 |
| XGBoost | 0.514 | 0.523 | 0.505 | 0.514 |
| **Voting Ensemble** | **0.546** | **0.554** | **0.559** | **0.556** |

The Voting Ensemble outperformed individual bagging/boosting models, and L1-regularized Logistic Regression gave the best single-model result overall — reflecting how much of the outcome variance in a T20 match is inherently unpredictable (toss, form on the day, etc.).

## Key Insights

- **Toss impact is negligible overall:** the toss winner won only ~50.6% (554/1095) of matches — no meaningful unfair advantage from winning the toss.
- **Team-level toss leverage varies:** teams like Chennai Super Kings convert toss wins into match wins significantly more often; teams like Lucknow Super Giants actually win more after *losing* the toss.
- **Venue matters for the bat/field decision:** at venues like Wankhede Stadium, batting first tends to win more (pitch slows down later); at venues like Subrata Roy Sahara Stadium, fielding first wins more (better for chasing, possible dew factor).
- **Mumbai has hosted the most IPL matches** of any city.
- **IPL is seasonal:** ~91% of matches are played in March–May, with occasional exceptions (Sept–Nov) due to external disruptions (e.g., elections, COVID).
- **Team clustering** cleanly separates consistently strong franchises, weaker/struggling teams, and discontinued teams — useful for benchmarking and strategy discussions.
- **Toss-decision recommendations** (from the decision tree) factor in venue batting bias, humidity, temperature, and relative team strength — e.g., field first when venue batting bias is low and teams are evenly matched; bat first under high humidity in playoff matches.
- **Feature importance (ensemble models):** team identity, venue, toss winner, and target runs are the most influential predictors of match outcome.

## Tech Stack

- **Python**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn (LogisticRegression, KMeans, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier), XGBoost

## Repository Structure

```
├── matches.csv                  # Raw IPL match dataset
├── IPL_Data_Analysis.ipynb      # Full analysis & modeling notebook
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook IPL_Data_Analysis.ipynb
```
