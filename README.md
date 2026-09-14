# Spotify-Songs-Genre-Segmentation
Unsupervised & supervised machine learning on Spotify audio features: clustering, genre classification, and a content-based recommendation system.

## Objective

Spotify recommends music by grouping songs with comparable audio characteristics. This project builds that pipeline end-to-end on a public Spotify tracks dataset:

1. Data pre-processing
2. Exploratory data analysis & visualizations
3. Correlation analysis
4. Clustering by audio features, compared against actual playlist genres
5. A supervised classifier that predicts genre from audio features
6. A content-based recommendation system
7. Summary of insights

## Files in this delivery

| File | Description |
|---|---|
| `Spotify_Genre_Segmentation.ipynb` | Full, already-executed Jupyter notebook — code, markdown narrative, and all outputs/plots included. Open and run top-to-bottom to reproduce. |
| `Spotify_Genre_Segmentation.html` | Static HTML export of the executed notebook — view in any browser, no Jupyter install needed. |
| `Spotify_Genre_Segmentation_Report.docx` | Written report (Word) covering the same analysis with embedded charts and narrative findings, for submission/printing. |
| `spotify_dataset.csv` | Source dataset (32,833 playlist-track listings, 23 columns). 

## Dataset

32,833 rows × 23 columns: track/artist/album metadata, `track_popularity`, `playlist_genre` / `playlist_subgenre` labels (6 genres, 24 subgenres), and 9 numeric audio features `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.

## How to reproduce

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter nbconvert --to notebook --execute Spotify_Genre_Segmentation.ipynb --output rerun.ipynb
```
Or simply open `Spotify_Genre_Segmentation.ipynb` in Jupyter/JupyterLab and run all cells.
The notebook expects `spotify_dataset.csv` in the same working directory.

## Methodology summary

- **Pre-processing:** dropped 5 rows with missing metadata; kept the full playlist-level table for EDA and built a de-duplicated one-row-per-track table (28,352 tracks) for clustering / modeling so multi-playlist tracks aren't over-weighted; z-scored the 9 audio features before any distance-based method.
- **Clustering:** K-Means on standardized features; k=6 chosen (via elbow + silhouette, matched to the number of playlist genres) for interpretability; visualized with PCA (2 components, ~41% variance explained).
- **Classification:** Random Forest (300 trees) predicting `playlist_genre` from the 9 audio features, 80/20 stratified split.
- **Recommendation:** content-based — restrict candidates to the seed track's K-Means cluster, then rank by cosine similarity of standardized audio-feature vectors.

## Key results

| Finding | Detail |
|---|---|
| Genre balance | 6 genres, 4,951–6,043 tracks each; 24 subgenres |
| Strongest correlations | energy↔loudness (+0.68), energy↔acousticness (−0.54), danceability↔valence (+0.33) |
| Popularity vs. audio features | No meaningful correlation (\|r\| ≤ 0.15) — popularity is driven by non-acoustic factors |
| Clustering (k=6) | Silhouette ≈ 0.14–0.20 — partial, not perfect, genre recovery; some clusters map cleanly to a dominant genre, others mix several |
| Genre classifier | 54.5% accuracy vs. 16.7% random baseline (6 classes); `tempo`, `speechiness`, `danceability` most predictive; edm/rock easiest, pop/r&b hardest |
| Recommender | Cluster filter + cosine similarity returns sonically coherent recommendations (e.g., 0.89–0.96 similarity within genre-consistent neighbors) |

## Tools

Python — `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (`KMeans`, `PCA`, `StandardScaler`, `RandomForestClassifier`, `cosine_similarity`), `nbformat`/`nbconvert` for notebook assembly and export.
