<!-- C01 -->

## Project Overview

This project analyzes Wikipedia article quality to identify which features (content characteristics, categories, languages) predict article ratings (GA, B, C, START, STUB). The goal is to determine **what aspects of an article to focus on to improve its rating**.

### Dataset
- Wikipedia articles with quality ratings
- Features: content length, links, categories, available languages
- Target: article quality rating (5 classes)

### Approach
1. **Preprocessing**: Expand multi-value features (categories, languages) into binary columns for ML modeling
2. **Unsupervised Learning**: Explore patterns and clusters in article characteristics
3. **Supervised Learning**: Build predictive models for article quality

---

**Note**: Code execution requires explicit authorization. Default behavior is to propose modifications only.

---

<!-- C02 -->

## Preprocessing Strategy

### Initial Exploration

Before one-hot encoding categories and languages, we analyzed the dataset structure:

**Dataset composition:**
- 16,472 total articles
- 97 unique languages, 112 unique categories (in current sample)
- 65% of articles exist only in English (no translation data)

**Coverage analysis:**
- **Categories**: Top 20 cover 98.7% of articles, with only 212 articles lacking any top-20 category
  - Remaining 1.3% scattered across 130 niche categories (not worth including)
- **Languages**: Top 20 cover 34.5% of articles with language data
  - More dispersed distribution than categories

**Preprocessing pipeline:**
1. **Data cleaning**: Add +1 to all language counts (original data incorrectly labeled English-only articles as 0 languages)
2. **Feature selection**:
   - Use **top 20 languages** for language features
   - Use **top 100 meta categories** (maintenance flags separated from subject matter)
3. **Feature encoding**: Binary encode selected categories and languages
4. **Standardization**: Scale all features to mean=0, std=1 for clustering and modeling
5. **Train/validation/test split**: 70%/15%/15% split for supervised learning

This approach reduces feature dimensionality from tens of thousands to ~130 meaningful features while maintaining signal quality.

---

<!-- C03 -->

## Unsupervised Learning

### K-Means Clustering

**Finding optimal k:**
- Used elbow method testing k=1 through k=10
- Identified k=**8** as optimal (elbow point where inertia gain diminishes)

**Cluster findings (k=8):**
- Clusters strongly align with article quality ratings
- Key differentiators: **content length**, **number of references**, and **number of languages**
- High-quality clusters (B/FA/GA ratings) have significantly longer content, more references, and broader language availability
- Low-quality clusters (START/STUB ratings) show below-average metrics across all features

**Key insight:** Content depth (length + references) is a stronger quality signal than breadth (links + languages). Articles with many links/languages but shorter content and fewer references tend toward medium quality (C/START) rather than high quality (B/GA).

### Re-clustering without Language Features

Given that language flags dominated the correlation analysis without adding much interpretive value, we removed all 20 `lang_*` features and re-ran K-means clustering.

**Results:**
- Optimal k increased from 8 to **14 clusters** (elbow method)
- Without language noise, the algorithm finds more granular structure in content/metadata features

**Cluster quality hierarchy (k=14):**
| Rank | Cluster | % Total | Avg Rating | Characterization |
|------|---------|---------|------------|------------------|
| 1 | 10 | 1.6% | 4.00 | Highest quality (B-level avg) |
| 2 | 9 | 0.8% | 3.63 | Near B-level |
| 3 | 2 | 1.3% | 3.58 | High quality |
| 4 | 8 | 16.2% | 3.44 | Large high-quality cluster |
| 5 | 13 | 1.2% | 3.23 | Above average |
| ... | ... | ... | ... | ... |
| 13 | 6 | 5.8% | 2.14 | Low quality |
| 14 | 1 | 2.4% | 1.57 | Lowest (58% stubs) |

The largest clusters (7, 0) sit near the middle (~2.4-2.5 avg), representing typical C/START articles.

### Feature-Rating Correlations

Analyzed correlations between all features and article quality ratings (excluding leaky columns like `meta_Good articles`, `meta_All stub articles`, `meta_Featured articles`).

**Top correlated features:**
1. **num_sections** (0.57) -- strongest predictor
2. **content_length** (0.57)
3. **num_references** (0.49)
4. **num_categories** (0.39)
5. **summary_length** (0.38)
6. **last_revision_date** (0.37)
7. **num_languages** (0.35)

After the top 7 structural features, correlations drop to ~0.30 and are dominated by individual language flags (lang_ca, lang_cs, lang_fi, etc.) -- suggesting that *having translations* matters more than *which specific languages*.

**Wayback links as a quality signal:**
The `meta_Webarchive template wayback links` feature (corr 0.31) marks articles citing archived sources. Two competing hypotheses:
- **Lindy/link rot:** Older articles accumulate dead links -> wayback citations (passive decay)
- **Editorial rigor:** Quality editors seek historical sources -> cite archived originals (intentional)

**Initial analysis** used `last_revision_date` as a proxy for article age:
- Raw correlation (wayback vs rating): 0.31
- Partial correlation (controlling for last_revision_date): 0.26
- Drop: 16%

However, `last_revision_date` is a poor proxy -- nearly all articles were last revised near the scrape date (late 2025/early 2026), reflecting Wikipedia's high edit frequency rather than actual article age.

**Updated analysis** using real article creation dates (fetched via the MediaWiki API for 8,449 articles):
- Raw correlation (wayback vs rating): **0.290**
- Partial correlation (controlling for article age): **0.224**
- Drop: **22.7%** -- article age is a meaningful confound
- Correlation (article age vs wayback): 0.237
- Correlation (article age vs rating): 0.371

The larger drop confirms that article age *is* a real confound -- older articles naturally accumulate both wayback links and higher ratings. But the residual partial correlation of 0.224 still holds firmly.

**Conclusion:** Even after properly controlling for article age, wayback links predict quality. This means that even for newer articles, the presence of archived sources signals editorial rigor -- quality editors seek older, primary sources regardless of when the article was created.

### Principal Component Analysis

PCA on the 110 features (excluding `lang_*`) reveals interpretable structure:

**Variance explained:**
- PC1: 22.1%
- PC2: 9.3%
- Total: 31.4% (modest, but sufficient to reveal cluster/quality structure)

**PC1 = Article Quality Dimension**
Top loadings: `num_categories`, `content_length`, `num_references`, `num_sections`
- Higher PC1 -> higher quality articles
- Clear gradient from STUB (low PC1) to GA (high PC1) visible in scatter plot

**PC2 = Chemistry/Science Domain**
Top loadings: `meta_Articles containing unverified chemical infoboxes`, `meta_Articles without KEGG source`, chemistry-related metadata flags
- Separates chemistry articles from the general population
- Negative loadings on `num_sections` -- chemistry articles have different structure

**Key finding:** Rating correlates strongly with PC1 (positive) and inversely with PC2. The first principal component essentially captures a "content quality" axis that aligns with Wikipedia's rating system, while PC2 captures domain-specific variance in chemistry/science articles.

### Silhouette Analysis

**Overall silhouette score: 0.017** -- very weak cluster separation.

| Cluster | Silhouette | Interpretation |
|---------|------------|----------------|
| 11, 13, 4, 0 | >0.10 | Reasonably tight |
| 5, 2, 1, 9 | 0.00-0.10 | Borderline |
| 3, 6, 7, 8, 10, 12, 14 | <0.00 | Negative (closer to other clusters) |

**Interpretation:** K-means is carving up a continuous space rather than finding natural clusters. The PCA scatter confirms this -- articles form a gradient along PC1, not discrete blobs. Clusters correlate with quality because K-means essentially bins a continuum. This suggests **article quality exists on a spectrum rather than in discrete tiers**.

---

## Supervised Learning

### Baseline Models

Trained Logistic Regression, Decision Tree, and Random Forest on three feature sets:
- **All features** (106): Full dataset minus `lang_*` and leaky columns
- **Numeric only** (9): Structural features (`content_length`, `num_sections`, etc.)
- **Top 30 correlated** (30): Features with highest rating correlation

**Results (Validation Accuracy):**

| Model | All (106) | Numeric (9) | Top 30 |
|-------|-----------|-------------|--------|
| Random Forest | **64.1%** | 62.7% | 63.2% |
| Logistic Regression | **59.2%** | 58.8% | 58.8% |
| Decision Tree | **53.1%** | 52.7% | 51.2% |

**Key findings:**
- All models perform best with the full feature set -- more features help, not hurt
- 9 numeric features capture ~98% of predictive signal
- 100 `meta_*` flags add only 1-2% accuracy over numeric-only
- FA/GA classes easy to classify; B/C/START form a confused middle ground

### Exploring Causality

The numeric features (`content_length`, `num_references`, `num_sections`) are **effects**, not causes. They don't explain *why* an article receives editor attention -- they're downstream outputs of that attention.

**Hypothesized causal chain:**
Notability/Interest -> Editor attention -> Content depth -> High rating

**Causal candidate features** (things that might *cause* editor attention):
- `meta_Wikipedia indefinitely move-protected pages` -> high-traffic/controversial topics attract dedicated editors
- `meta_Webarchive template wayback links` -> signals editorial rigor culture
- `meta_CS1 *-language sources` -> international sourcing effort
- `num_languages` -> proxy for global interest/importance

**Next steps:** Test whether "upstream" indicators (notability, editor culture, protection status) can predict quality independently of "downstream" outputs (word count, reference count).

---

<!-- C26 -->

Languages correlate decently with ratings, but they don't mean much on their own. Focusing on maximizing them would violate Goodhart's law. It could be that better rated articles usually have their first version in a foreign language, and English builds on top of it, but it's not clear, and verifying it right now would be intractable. We would have to compare the earliest edit of each article in each language to see.

---

<!-- C28 -->

Apparently, sentence length is the least predictive of the numeric features, by an 8% margin.

Now, the category "Webarchive template wayback links" is interesting. Articles under that category have references to Wayback Machine or Archive.is pages i.e. pages that no longer exist. Probably reflective of the Lindy effect. These articles must be older, with their references frequently checked. If they no longer exist, then they point to the Wayback Machine. Could also be a function of the editors looking for earlier sources about the topic, to avoid revisionism.

---

<!-- C31 -->

It seems to me that, controlling for age, wayback links are predictive of quality, indicated by the small drop in correlation, from .31 to .26, when I do the partial correlation.

---

<!-- C81 -->

Next: I need to understand these features more, then we can go local, break up this notebook etc.

---

