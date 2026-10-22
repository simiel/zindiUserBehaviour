# Zindi User Behaviour Birthday Challenge

This repository is Samuel Mensah's working exploration of the [Zindi User Behaviour Birthday Challenge](https://zindi.africa/competitions/zindi-user-behaviour-birthday-challenge). For each user and month, the task is to predict whether that user will be active on Zindi. `Target` is 1 when the user enrols in a competition, writes a comment, makes a submission, or starts a discussion in that month, and 0 otherwise. Train rows cover masked years 1–3 and include those four activity flags. The test set is masked year 4, months 1–3, and contains only `User_ID`, `month`, and `year`. Countries, public ranks, point bands, and several category fields are masked, and the same mask is used across tables. Month and weekday codes are ordered in time, but January is not necessarily month 1 and Monday is not necessarily day 1.

## Key features

- Competition tables for users, competitions, participation, submissions, discussions, and comments, plus train, test, and a sample submission keyed by `UserMonthYear` (`{User_ID}_{month}_{year}`).
- Monthly aggregation of comments, discussions, and submissions, then left-joins of those counts onto the user-month panel.
- A longer modelling notebook that fills missing competition fields, attaches competition metadata (points reward, daily submission limit, start and end time) to each user-month, and retrains after under-sampling inactive users and over-sampling active users.
- Submission-shaped outputs (`UserMonthYear`, `Target`) written from those experiments.
- Two included reference projects: a full French coursework write-up of the same challenge, and a separate exploratory-data-analysis notebook that builds processed per-table extracts.

## Tech stack

The work is a set of Jupyter notebooks (Python 3). The libraries imported across the notebooks are:

- **pandas** and **NumPy** for loading, grouping, and joining the tables
- **matplotlib** and **seaborn** for distribution and missing-value plots
- **scikit-learn** for the models that are actually fit: `RandomForestClassifier` in the PlayGround notebooks, and, in the included coursework notebook, `KNeighborsClassifier`, `DecisionTreeClassifier`, `LogisticRegression`, `GaussianNB`, `LinearRegression`, and `SVR`
- **LightGBM** (`LGBMClassifier`) is imported alongside the random forest in several notebooks. The PlayGround models that run are random forests. `LGBMClassifier` is fit in `PlayGround/Financial_Resilience_Python_StarterNotebook.ipynb`, which is a starter for a different challenge (see [Project status](#project-status))

There is no dependency file, environment file, or application framework.

## Repository structure

The competition CSVs are stored three times: at the repository root, under `PlayGround/data/`, and under `Projet-Zindi-Zindi-User-Behaviour-Birthday-Challenge-main/`. Row counts below are from the root copies.

| File | Rows | Role |
| --- | ---: | --- |
| `Train.csv` | 259,832 | User-month panel for years 1–3: `User_ID`, `month`, `year`, `CompPart`, `Comment`, `Sub`, `Disc`, `Target`. 20,218 users. 36,306 active rows (14.0%) and 223,526 inactive rows. |
| `Test.csv` | 65,223 | Year 4, months 1–3. 22,402 users, including every train user and 2,184 users who do not appear in train. |
| `SampleSubmission.csv` | 65,223 | `UserMonthYear`, `Target` |
| `Users.csv` | 22,407 | Masked country, two user categories, four point groups, and signup year/month/weekday |
| `CompetitionPartipation.csv` | 48,565 | Enrolments, with public rank and successful-submission count in masked bands |
| `Competitions.csv` | 154 | Hackathon or competition (`Kind`, 77 rows each), plus masked categories, scaled points reward, secret-code flag, and start/end time |
| `Submissions.csv` | 375,763 | Per-submission rows with a masked submission category |
| `Discussions.csv` | 6,211 | Discussion id, masked category, author, and date parts |
| `Comments.csv` | 11,751 | Comment author and date parts |
| `VariableDefinitions.csv` | — | Official field descriptions, including the definition of `Target` |

```text
.
├── Train.csv, Test.csv, SampleSubmission.csv, VariableDefinitions.csv
├── Users.csv, Competitions.csv, CompetitionPartipation.csv
├── Submissions.csv, Discussions.csv, Comments.csv
├── result.csv, result2.csv, result3.csv      # submission-shaped outputs
├── PlayGround/                                 # Mensah's notebooks
│   ├── Data Cleaning.ipynb                     # missing-value inspection
│   ├── DATA_GROUPING.ipynb                     # monthly comment, discussion, and submission counts
│   ├── merge_grouped_data.ipynb                # join those counts and fit a random forest
│   ├── zindiPrediction.ipynb                   # calendar baseline, then counters from the cleaned extracts
│   ├── final/Untitled.ipynb                    # merged feature set, resampling, random forest, Baseline.csv
│   ├── monthly_grouped_data/                   # saved group-by outputs
│   ├── data/                                   # local copy of the competition CSVs
│   └── Financial_Resilience_Python_StarterNotebook.ipynb
├── Projet-Zindi-Zindi-User-Behaviour-Birthday-Challenge-main/
│   ├── rapport.pdf                             # team MAD project report
│   ├── Data_Clean.ipynb, Visualisation .ipynb, Note_Final.ipynb
│   └── Data_Clean/                             # comment, discussion, and submission counters
└── Zindi-User-Behaviour-Birthday-Challenge-master/
    ├── EDA_User_Behaviour_Zindi.ipynb
    └── Timestamp Creation.ipynb
```

`PlayGround/Data Cleaning.ipynb` and `PlayGround/final/Untitled.ipynb` still open with the prompt from the UmojaHack Financial Resilience starter ("predict if an individual will be able to make a payment in an emergency"). The code under that heading loads this challenge's tables.

## Approach

Predicting the test month means the model cannot use that month's `CompPart`, `Comment`, `Sub`, or `Disc` flags: those flags are how `Target` is defined, and they are absent from `Test.csv`. The PlayGround notebooks build features that exist before the label:

1. **Inspection.** `Data Cleaning.ipynb` checks each table for missing values. Participation is missing `PublicRank` and `Successful Submission Count`. Competitions are missing country, `FeatureC`, `SubmissionLimitPerDay`, and parts of the end date.
2. **Monthly counts.** `DATA_GROUPING.ipynb` groups comments by user, year, and month; discussions by user, year, month, and category; and submissions by user, competition, year, month, and category. The aggregates are saved under `PlayGround/monthly_grouped_data/`.
3. **Joined baselines.** `merge_grouped_data.ipynb` left-joins monthly comment and discussion counts onto train and test, fills unmatched months with 0, and fits `RandomForestClassifier` on `month`, `year`, comment count, and discussion count. `zindiPrediction.ipynb` first fits a forest on `month` and `year` alone, then merges the per-user counters produced by the coursework cleaning notebook (`comment_counter`, `discussions_counter`, `submission_counter`, successful-submission counts, and cleaned user and competition tables) and fits a second forest on month, year, discussion count, and submission count.
4. **Richer user-month table.** `PlayGround/final/Untitled.ipynb` renames columns to a shared schema, fills missing ranks and competition fields, drops the four activity flags before combining train and test, and left-joins comment counts, discussion counts, and a submission-to-competition merge (so each user-month can carry points reward, submission limit, and competition start/end). Categorical holes are filled with the column mode and numeric count, reward, and limit holes with 0. It then builds an under-sampled frame (inactive rows sampled down to the active count) and an over-sampled frame (active rows sampled with replacement up to the inactive count), and fits `RandomForestClassifier` on the over-sampled frame using month, year, discussion count, day, submission count, points reward, submission limit, competition end month, and competition start year and month.

The included `Note_Final.ipynb` takes a different route. It fits k-nearest neighbours, a decision tree, logistic regression, Gaussian naive Bayes, linear regression, and support-vector regression, often using the train activity flags (`CompPart`, `Comment`, `Sub`, `Disc`) as inputs and swapping in rank or counter columns on the test side. Training accuracy in that notebook climbs toward 1.0 once those flags are included, which follows from the definition of `Target`.

`Zindi-User-Behaviour-Birthday-Challenge-master/EDA_User_Behaviour_Zindi.ipynb` prepares each side table on its own: competition duration in months, an `isSubmitted` flag, separate counters for each masked discussion and submission category, one-hot competition ids, and user-country and points dummies limited to users who appear in test. `Timestamp Creation.ipynb` starts a per-user assembly of those extracts and stops before finishing it.

## Results recorded in the repo

No leaderboard file is stored for the PlayGround submissions. What the notebooks do record is training-set accuracy from `model.score` on the rows the model was fit on:

| Notebook | Features | Recorded training accuracy |
| --- | --- | --- |
| `PlayGround/zindiPrediction.ipynb` | `month`, `year` | 0.8613 |
| `PlayGround/merge_grouped_data.ipynb` | `month`, `year`, monthly comment count, monthly discussion count | 0.8840 |

An always-inactive prediction scores 223,526 / 259,832 = 0.8603 on train, so the calendar-only forest sits on that majority-class rate. Adding monthly comment and discussion counts is the step that moves training accuracy. The final notebook imports `roc_auc_score` and `classification_report`, and the hold-out split and AUC print are left commented out, so it does not record an AUC.

Saved submission files at the test-set length of 65,223 rows:

| File | Predicted active rows |
| --- | ---: |
| `result.csv` | 0 |
| `PlayGround/wrang.csv` | 0 |
| `PlayGround/final/Baseline.csv` | 0 |
| `result3.csv` | 6,270 |
| `result2.csv` | 1,107 rows labelled active, 6,843 labelled inactive, and the remaining `Target` cells empty |

The included report, `Projet-Zindi-Zindi-User-Behaviour-Birthday-Challenge-main/rapport.pdf`, records that team's own Zindi result: a k-nearest-neighbours submission score of **0.509958266790094**, and a final rank of **91 of 111** under the team name MAD. The first commit message notes that submitting that work scored 0.5.

`PlayGround/Financial_Resilience_Python_StarterNotebook.ipynb` prints a random-forest AUC of about 0.76 and a classification report. Those figures belong to the UmojaHack Africa 2021 Financial Resilience starter, which uses different columns (`country`, `region`, `owns_mobile`, survey `Q*` fields) and a different target.

## Getting started

Prerequisites are Python 3, Jupyter, and the libraries listed under [Tech stack](#tech-stack). The repository has no `requirements.txt`, no environment variables, and no install, build, or test command.

The CSV files are already in the tree. From the repository root:

```bash
jupyter notebook
```

Notebooks expect a particular working directory:

- `PlayGround/Data Cleaning.ipynb`, `PlayGround/DATA_GROUPING.ipynb`, and `PlayGround/merge_grouped_data.ipynb` read `data/` and `monthly_grouped_data/` relative to `PlayGround/`.
- `PlayGround/final/Untitled.ipynb` reads `../data/` relative to `PlayGround/final/`.
- `PlayGround/zindiPrediction.ipynb` reads `Train.csv` and `Test.csv` from the working directory and the cleaned counters from `Projet-Zindi-Zindi-User-Behaviour-Birthday-Challenge-main/Data_Clean/`.

Cells were run interactively. Some cells are empty, and a few prediction lines are commented out, so opening a notebook and running every cell top to bottom will not reproduce every saved CSV without the working directory above.

## Project status

Personal working repository for a completed Zindi competition. Git history starts on 3 January 2022; the included report was generated on 31 December 2021. The notebooks are an exploration of features and baselines, with saved submission files and no packaged training entry point.

`Projet-Zindi-Zindi-User-Behaviour-Birthday-Challenge-main/` is the 2021 integrated project (*projet intégré*) of team MAD — Chawali Rania, Aouini Sabrine, Zalleg Mohamed Adem, and Akrout Yosr — second-year licence students in applied mathematics for data analysis and decision support (*Mathématiques appliquées à l'analyse des données et aide à la décision*) at Esprit School of Business. The report's last page thanks Manai Elyes. That folder arrived in the initial commit alongside the competition data; later commits edit the cleaning notebook while reading those extracts into `zindiPrediction.ipynb`.

## What this demonstrates

- **Relational feature engineering on an event log.** Comments, discussions, and submissions are collapsed to user-month (and user-competition-month) grain and joined back to a panel, which is the right shape for a forecast that cannot see the month it is scoring.
- **Reading a masked schema before modelling.** The variable dictionary is used as written: consistent masks across tables, ordinal months that are not calendar months, and rank and submission counts that arrive as bands rather than raw integers. Missing rank, country, and competition-end fields are filled explicitly before the join.
- **A baseline that respects the base rate.** The first forest, fit only on month and year, lands on the 86% inactive rate. The next experiment adds real activity history and is the one that changes training accuracy. That is a useful check on a label this skewed.
- **Imbalance handling as part of the model, not an afterthought.** The final notebook measures the class counts, builds both an under-sampled and an over-sampled training frame, and fits the forest on the over-sampled frame while keeping the activity flags out of the feature list.
- **End-to-end competition mechanics.** Outputs are written in the `UserMonthYear` / `Target` schema Zindi expects, and another team's full cleaning pipeline, plots, and report are kept in-repo and reused as feature inputs rather than re-derived from scratch.
