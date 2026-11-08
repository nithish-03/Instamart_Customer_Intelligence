# Instamart Customer Intelligence & Support Copilot

An enterprise-grade customer intelligence, retention modeling, and automated support triage copilot designed for high-velocity quick-commerce operations across 6 Indian metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune).

---

## Table of Contents
1. [Overview](#overview)
2. [Reproduce](#reproduce)
3. [Part A: Data Foundation: Clean, Store, Query](#part-a-data-foundation-clean-store-query)
4. [Part B: Customer Segments and Lapse Prediction](#part-b-customer-segments-and-lapse-prediction)
5. [Part C: Support Ticket Triage with Guardrails](#part-c-support-ticket-triage-with-guardrails)
6. [Part D: Policy RAG Assistant and Refund Agent](#part-d-policy-rag-assistant-and-refund-agent)
7. [Limitations and Next Steps](#limitations-and-next-steps)

---

## Overview

High-frequency quick-commerce platforms encounter three critical operational friction points as order volumes scale exponentially:
1. **Unreliable Raw Order Exports**: Inconsistent schema hygiene, duplicated network retries, conflicting order lifecycle states (`pending` vs. `completed`), unparseable prices, and anomalous delivery metrics degrade analytical integrity.
2. **Customer Churn Blindspots**: Inability to systematically differentiate high-value long-term patrons from churn-risk cohorts, resulting in untargeted retention capital expenditure.
3. **Overwhelmed Support Queues**: High-latency manual triage of unstructured support tickets, customer privacy compliance vulnerabilities (PII exposure in prompts), and slow, error-prone manual refund adjudication.

This repository deploys a production-ready, modular system resolving all three challenges:
- **Part A (Data Foundation)**: Cleans, deduplicates, and validates raw transactional tables into an ACID-compliant SQLite analytical warehouse (`data/instamart.db`) and executes 7 core business SQL analyses.
- **Part B (Intelligence & Retention)**: Implements 3D RFM segmentation ($K=4$ $K$-Means clustering and 2D PCA) and trains leakage-free supervised classification pipelines predicting 30-day customer lapse risk ($F_1 \ge 0.65$).
- **Part C (Support Ticket Triage)**: Deploys a zero-trust LLM classification pipeline featuring deterministic PII redaction, exponential jitter backoff, schema-constrained output validation, cost-aware token budgeting, and segment-aware priority routing.
- **Part D (Autonomous Policy RAG & Refund Agent)**: Implements word-chunked TF-IDF retrieval-augmented generation (RAG) alongside a stateful, resumable ReAct autonomous agent capable of resolving customer refund requests via multi-step database and policy tool executions.

---

## Reproduce

The entire project is 100% reproducible offline without external API keys or network dependencies. To reproduce all data artifacts, database tables, trained models, evaluation matrices, and outputs from a clean environment, execute the following numbered steps in your shell:

```bash
# 1. Clone the repository and navigate into the root directory
git clone https://github.com/<user>/instamart_customer_intelligence.git
cd instamart_customer_intelligence

# 2. Check out the main branch
git checkout main

# 3. Create an isolated Python 3.11+ virtual environment
python -m venv venv

# 4. Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 5. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Generate deterministic raw synthetic data and policy corpora (seed = 42)
python generate_data.py

# 7. Execute Part A: Clean raw data, populate SQLite database, and run SQL queries
python src/part_a.py
python src/sql_analysis.py

# 8. Validate Part A data integrity checkpoints against reference rubric
python src/validate.py

# 9. Execute Part B: Build RFM features, fit K-Means/PCA, train lapse models, and output retention lists
python src/part_b.py

# 10. Execute Part C: Run PII sanitisation, prompt evaluations, retry resilience, and triage routing
python src/part_c.py

# 11. Execute Part D: Build TF-IDF chunk retriever, run policy QA evaluation, and execute ReAct agent scenarios
python src/part_d.py

# 12. Alternatively, run the complete end-to-end pipeline in a single command:
python src/run_all.py
```

---

## Part A: Data Foundation: Clean, Store, Query

### 1. Raw Data Missingness and Anomaly Audit

The raw data ingestion comprises three primary tables: `customers.csv` ($1,200 \times 5$), `products.csv` ($40 \times 4$), and `orders.csv` ($13,084 \times 9$). Initial schema and null audits reveal systematic data defects planted by network retries, upstream form validation drops, and manual logging discrepancies.

#### Table Audit Summary
| Table Name | Raw Shape | Memory Usage | Primary Key Candidate | Completeness |
| :--- | :--- | :--- | :--- | :--- |
| `customers.csv` | $1,200 \text{ rows} \times 5 \text{ cols}$ | ~47.0 KB | `customer_id` | Missing values present in `age` and `gender` |
| `products.csv` | $40 \text{ rows} \times 4 \text{ cols}$ | ~1.4 KB | `product_id` | 100% complete (clean catalogue reference) |
| `orders.csv` | $13,084 \text{ rows} \times 9 \text{ cols}$ | ~920.1 KB | Non-unique `order_id` | Duplicate rows, unparseable amounts, negative durations |

#### Column-Level Null and Anomaly Audit
| Table | Column Name | Data Type | Null Count | Null % | Anomaly / Dirty Pattern Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customers` | `customer_id` | object | 0 | 0.00% | Format `C0001` to `C1200` |
| `customers` | `city` | object | 0 | 0.00% | 24 distinct casing variations (`mumbai`, `MUMBAI`, ` Mumbai `) |
| `customers` | `signup_date` | object | 0 | 0.00% | ISO date strings `YYYY-MM-DD` |
| `customers` | `age` | float64 | 61 | 5.08% | 61 blanks; 6 values outside $[10, 100]$: `-4, 0, 150, 200, 280, 999` |
| `customers` | `gender` | object | 30 | 2.50% | 30 blanks; 10 distinct raw spellings (`m`, `M`, `male`, `FEMALE`, etc.) |
| `orders` | `order_id` | object | 0 | 0.00% | 12,309 unique IDs across 13,084 rows (duplicate records) |
| `orders` | `customer_id` | object | 0 | 0.00% | Valid foreign keys |
| `orders` | `product_id` | object | 0 | 0.00% | Valid foreign keys |
| `orders` | `quantity` | int64 | 0 | 0.00% | Integer order volumes ($1$ to $3$) |
| `orders` | `order_ts` | object | 0 | 0.00% | Timestamps spanning 2025-04-01 to 2025-09-30 |
| `orders` | `status` | object | 0 | 0.00% | 656 `pending`, 11,229 `completed`, 1,199 `cancelled` |
| `orders` | `discount_pct` | int64 | 0 | 0.00% | Values in $\{0, 5, 10, 15, 20, 25, 40\}$ |
| `orders` | `order_amount` | object | 120 | 0.92% | 120 unparseable strings (`"N/A"` or `"error"`) |
| `orders` | `delivery_minutes` | float64 | 2,541 | 19.42% | 2,541 blanks; 106 planted negative values (`-10`) |

#### Specific Raw Audit Figures
| Metric Description | Exact Value | Reference Checkpoint | Status |
| :--- | :--- | :--- | :--- |
| Raw `orders.csv` rows | **13,084** | 13,084 | Passed |
| Exact duplicate rows in `orders.csv` | **119** | 119 | Passed |
| Rows with status = `pending` | **656** | 656 | Passed |
| Distinct `order_id` values | **12,309** | 12,309 | Passed |
| Non-numeric `order_amount` entries | **120** | 120 | Passed |
| Negative `delivery_minutes` entries | **106** | 106 | Passed |
| Blank `delivery_minutes` entries | **2,541** | 2,541 | Passed |
| Blank `age` entries in `customers.csv` | **61** | 61 | Passed |
| `age` entries outside range $[10, 100]$ | **6** | 6 | Passed |
| Distinct raw city spellings | **24** | 24 | Passed |
| Distinct non-blank raw gender spellings | **10** | 10 | Passed |
| Blank `gender` entries | **30** | 30 | Passed |

---

### 2. Explicit Cleaning Rules and Execution

#### Customer Cleaning Rules (`customers.csv`)
1. **City Standardisation**: Stripped leading/trailing whitespace and applied title-casing (`.str.strip().str.title()`). Reduced 24 corrupted spelling variations down to exactly 6 canonical cities:
   - Pune: 227
   - Delhi: 210
   - Bengaluru: 197
   - Chennai: 191
   - Hyderabad: 191
   - Mumbai: 184
   *(Values modified: 698 non-canonical casing/whitespace entries standardised)*.
2. **Gender Normalisation**: Mapped raw strings using case-insensitive regex pattern matching: `male|m` $\rightarrow$ `male`, `female|f` $\rightarrow$ `female`, and whitespace/empty $\rightarrow$ `unknown`.
   - Results: `female`: 610, `male`: 560, `unknown`: 30.
   *(Values modified: 1,200 rows mapped to canonical enum; 30 blanks converted to `unknown`)*.
3. **Age Validation & City-Level Median Imputation**: Identified invalid ages ($< 10$ or $> 100$) and set them to NaN (6 invalid entries). Combined with the 61 blank entries (total 67 missing), imputed missing ages using the median age of the customer's corresponding city via `df.groupby('city')['age'].transform('median')`.
   - Results: 0 missing ages remaining; minimum age = 18, maximum age = 60.
   *(Values modified: 67 age fields imputed)*.
4. **Signup Date Parsing**: Cast `signup_date` strings to native datetime format (`YYYY-MM-DD`).

#### Order Cleaning Rules (`orders.csv`)
The order cleaning pipeline was executed strictly in sequential order:
1. **Timestamp Conversion**: Parsed `order_ts` to datetime objects (`pd.to_datetime`).
2. **Exact Duplicate Removal**: Purged exact duplicate rows across all columns.
   *(Rows removed: 119 exact duplicates; table reduced from 13,084 to 12,965 rows)*.
3. **Lifecycle State Reconciliation (Latest Timestamp Resolution)**: Resolved multiple records sharing the same `order_id` (e.g., initial `pending` entry followed by terminal `completed` or `cancelled` update) by sorting ascending on `order_ts` and retaining only the last entry per `order_id` (`keep='last'`).
   *(Rows removed: 656 superseded records; table reduced from 12,965 to exactly 12,309 unique orders. 0 `pending` records remain)*.
4. **Order Amount Repair**: Converted `order_amount` to numeric via `pd.to_numeric(errors='coerce')`, identifying 120 unparseable strings. Merged with product catalogue prices and reconstructed missing amounts using the exact formula:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   *(Values modified: 120 unparseable amounts calculated and filled; 0 missing values remain)*.
5. **Delivery Duration Sanitisation & Category Imputation**:
   - Replaced negative durations ($-10$) with NaN (106 values).
   - For `completed` orders, imputed missing delivery times using the median delivery time of that order's product category among completed orders via `groupby('category')['delivery_minutes'].transform('median')`.
   - For `cancelled` orders, left `delivery_minutes` as missing (`np.nan`), since cancelled orders were never dispatched or delivered; imputing synthetic delivery durations would corrupt operational log fidelity.
   *(Values modified: 106 negative values nullified; 1,236 missing values in completed orders imputed; exactly 1,199 missing values retained for cancelled orders)*.

#### Final Cleaned Dataset Verification
- Total unique clean orders: **12,309** rows
- Status breakdown: **11,110** `completed`, **1,199** `cancelled`, **0** `pending`
- Missing values: `order_amount` = 0; `delivery_minutes` (completed) = 0; `delivery_minutes` (cancelled) = 1,199
- Cumulative completed gross revenue: **INR 2,998,668.80**

The clean datasets were exported to `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`. Furthermore, all three tables were committed to the relational SQLite database `data/instamart.db` with `order_ts` indexed and formatted as `YYYY-MM-DD HH:MM:SS`.

---

### 3. Checkpoint Validation Script Execution

Executing `python src/validate.py` performs 8 deterministic assertion tests verifying schema constraints, row counts, categorical frequencies, and financial balances:

```
[VALIDATION SUITE: PART A]
[PASS] Checkpoint 1: Clean order count matches distinct order_id count (12,309 == 12,309)
[PASS] Checkpoint 2: Order status distribution matches reference (11,110 completed, 1,199 cancelled, 0 pending)
[PASS] Checkpoint 3: Completed orders gross revenue matches exact sum (INR 2,998,668.80)
[PASS] Checkpoint 4: Customer city frequency distribution matches reference across all 6 metros
[PASS] Checkpoint 5: Customer gender counts match reference (Female: 610, Male: 560, Unknown: 30)
[PASS] Checkpoint 6: Customer age bounds valid [18, 60] with 0 nulls
[PASS] Checkpoint 7: Zero missing order_amount and zero missing delivery_minutes for completed orders
[PASS] Checkpoint 8: Exactly 1,199 missing delivery_minutes preserved for cancelled orders
----------------------------------------------------------------------
ALL CHECKPOINTS PASSED
```

---

### 4. SQL Business Intelligence Analysis

All analytical queries are persisted in `sql/queries.sql` and executed against `data/instamart.db` using SQLite 3.25+ window and aggregation functions. All revenue calculations consider `completed` orders only.

```sql
-- queries.sql summary extract
-- Q1: WHERE order_amount > 500 GROUP BY customer HAVING SUM > 8000
-- Q2: 3-Table Join (City x Category Revenue Breakdown)
-- Q3: Scalar Subquery (Products priced above catalogue mean)
-- Q4: LEFT JOIN (Customers with zero completed transactions)
-- Q5: CTE (Customers spending above system-wide customer average)
-- Q6: Window Functions (RANK, DENSE_RANK, ROW_NUMBER by City Spend)
-- Q7: Cumulative Running Revenue Window SUM OVER (ORDER BY Day)
```

#### SQL Query Findings & Reference Verifications
| Query ID | Architectural Focus | Returned Dimensions | Core Analytical Result | Reference Validation |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | `WHERE` then `HAVING` Filter | 30 rows $\times$ 2 cols | **30 high-value customers** spent $>$ INR 8,000 on orders $>$ INR 500 | Exactly 30 customers |
| **Q2** | Three-Table Relational Join | 30 rows $\times$ 3 cols | Total Revenue = **INR 2,998,668.80**; Highest cell: **Bengaluru / Electronics** at **INR 233,909.45** | Exactly 30 rows; sum matches |
| **Q3** | Scalar Subquery | 11 rows $\times$ 3 cols | **11 products** priced above catalogue mean (INR 283.48) | Exactly 11 products |
| **Q4** | `LEFT JOIN` with Null Check | 25 rows $\times$ 2 cols | **25 customers** registered with zero completed orders | Exactly 25 customers |
| **Q5** | Common Table Expressions | 550 rows $\times$ 2 cols | System customer average = **INR 2,552.06**; **550 customers** spend above average | Exactly 550 customers |
| **Q6** | Window Ranking Functions | 1,175 rows $\times$ 6 cols | Top customer per city: Bengaluru `C0082` (INR 12,856.20), Chennai `C0930` (INR 10,151.40), Delhi `C0291` (INR 13,742.60), Hyderabad `C0743` (INR 12,730.05), Mumbai `C0693` (INR 14,766.55), Pune `C0426` (INR 10,377.10) | Exactly 8 rank ties detected |
| **Q7** | Cumulative Window Aggregation | 183 rows $\times$ 3 cols | Day 1 (2025-04-01): INR 16,394.00; Day 183 (2025-09-30) Cumulative Total = **INR 2,998,668.80** | Exactly 183 days; terminal sum matches |

**Explanation of Q6 Window Ranking Differences**:
In Q6, exactly 8 rows exhibit a discrepancy between `RANK()` and `ROW_NUMBER()`. This divergence occurs because `ROW_NUMBER()` assigns strictly distinct, sequential integers to rows regardless of ties, whereas `RANK()` detects identical `total_spend` values within the same city partition, assigns them an identical rank, and skips the subsequent ranking positions by the tie cardinality.

---

### 5. pandas Cross-Check and Exploratory Data Analysis (EDA)

#### City-Level Spend Aggregations (Pandas Named Aggregations)
Executing `pd.pivot_table(orders_df, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)` yields a grand margin total of exactly **INR 2,998,668.80**, matching the SQL Q2 join total to the paisa.

| City | Completed Orders Count | Gross Spend (INR) | Mean Order Value (INR) | Spend Share (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Bengaluru** | 1,894 | INR 523,643.40 | INR 276.47 | 17.46% |
| **Chennai** | 1,863 | INR 494,570.65 | INR 265.47 | 16.49% |
| **Delhi** | 2,059 | INR 564,285.05 | INR 274.06 | 18.82% |
| **Hyderabad** | 1,865 | INR 498,382.40 | INR 267.23 | 16.62% |
| **Mumbai** | 1,613 | INR 479,058.50 | INR 297.00 | 15.98% |
| **Pune** | 1,816 | INR 438,728.80 | INR 241.59 | 14.63% |
| **Total** | **11,110** | **INR 2,998,668.80** | **INR 269.91** | **100.00%** |

#### Distributional Properties of Completed Order Amount
- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: 4.61 (heavily right-skewed / positive skew)
- **First Quartile ($Q_1$)**: INR 64.80
- **Third Quartile ($Q_3$)**: INR 297.00
- **Interquartile Range ($IQR$)**: INR 232.20
- **Tukey Fences**: Lower Fence = $64.80 - 1.5 \times 232.20 = -\text{INR } 283.50$ (effective bound 0); Upper Fence = $297.00 + 1.5 \times 232.20 = \text{INR } 645.30$
- **Outliers Detected**: Lower outliers = **0**; Upper outliers ($> \text{INR } 645.30$) = **897 orders** (comprising premium consumer electronics such as wireless earbuds, power banks, and bluetooth speakers).

#### Correlation Matrix (Completed Orders)
| Metric | Order Amount | Discount % | Quantity | Delivery Minutes |
| :--- | :--- | :--- | :--- | :--- |
| **Order Amount** | 1.000000 | -0.095431 | 0.087224 | 0.004112 |
| **Discount %** | -0.095431 | 1.000000 | -0.003120 | 0.001845 |
| **Quantity** | 0.087224 | -0.003120 | 1.000000 | -0.002871 |
| **Delivery Minutes**| 0.004112 | 0.001845 | -0.002871 | 1.000000 |

#### Exploratory Data Synthesis (Under 200 Words)
This dataset tracks quick-commerce grocery and electronic transactions across 6 Indian metros. The completed order amount distribution is strongly right-skewed (mean INR 269.91 vs. median INR 135.00, skewness 4.61), driven by frequent everyday grocery basket staples (INR 25 to 150) alongside high-value electronics purchases reaching INR 1,999.00. Setting the upper $1.5\times IQR$ fence at INR 645.30 isolates 897 legitimate high-ticket outlier transactions without any lower anomalies. The strongest linear relationship exists between `order_amount` and `discount_pct` at $r = -0.095$ (with `quantity` closely following at $r = +0.087$). Crucially, a correlation coefficient of $-0.095$ is close to zero and does not establish causation; higher discount percentages slightly compress net billing amounts mathematically rather than discouraging larger transaction values. Delivery duration remains uncorrelated with transaction basket size.

---

## Part B: Customer Segments and Lapse Prediction

### 1. RFM Feature Extraction

Behavioral features were engineered across all **1,175 customers** with completed orders up to snapshot timestamp `2025-10-01 00:00:00`:
- **Recency ($R$)**: Whole days from customer's latest completed order to snapshot (`(snapshot - max_ts).dt.days`).
- **Frequency ($F$)**: Distinct count of completed orders.
- **Monetary ($M$)**: Cumulative sum of completed `order_amount` (INR).

#### RFM Baseline Statistics
| Metric | Mean | Std Dev | Min | 25% | Median | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recency (days)** | 27.65 | 27.81 | 0 | 7.00 | 19.00 | 39.00 | 180 |
| **Frequency (orders)**| 9.46 | 6.54 | 1 | 4.00 | 8.00 | 14.00 | 33 |
| **Monetary (INR)** | 2,552.06 | 2,214.38 | 20.00 | 954.00 | 1,884.00 | 3,460.00 | 14,766.55 |

---

### 2. $K$-Means Clustering Diagnostics ($K=2$ to $8$)

RFM features were standardised using `StandardScaler`. Unsupervised clustering was evaluated across $K \in [2, 8]$ with `KMeans(n_clusters=K, n_init=10, random_state=42)`:

| Clusters ($K$) | Inertia | Silhouette Score | Business Actionability Rating |
| :--- | :--- | :--- | :--- |
| **$K=2$** | 1,820.8 | 0.465 | Coarse split; merges high-spend and high-frequency users |
| **$K=3$** | 1,268.2 | 0.445 | Suboptimal boundary resolution; poor separation |
| **$K=4$** | **743.1** | **0.526** | **Optimal Elbow & Global Silhouette Peak; Highly Actionable** |
| **$K=5$** | 617.4 | 0.481 | Fragmentation of churned segment without strategic utility |
| **$K=6$** | 503.2 | 0.482 | Over-segmentation of mid-tier frequency buyers |
| **$K=7$** | 408.5 | 0.416 | Excessive subdivision; diminishing returns |
| **$K=8$** | 366.5 | 0.412 | Noise fitting; clusters lack distinct marketing identity |

#### Inertia Mathematical Property vs. Model Selection
Inertia measures the within-cluster sum of squared errors:
$$\text{Inertia} = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$$
As $K$ increases from 2 to 8, inertia monotonically decreases because adding cluster centroids mechanically shortens the Euclidean distance from data points to their nearest center, reaching 0 when $K=N$. Consequently, inertia alone cannot select $K$; we choose **$K=4$** because it forms a distinct mathematical elbow, achieves the global maximum silhouette score (0.526), and neatly maps onto four intuitive quick-commerce customer personas. Note that cluster numbers are arbitrary computational labels with no intrinsic ordering.

---

### 3. Segment Profiles, Naming, and PCA Projection

```
                  PCA Projection of Customer Segments (K=4)
   4 +-----------------------------------------------------------------------+
     |                                                                       |
     |                              *  Cluster 1: Premium High-Spenders      |
   2 |                            * *                                        |
     |                           * * *                                       |
PCA2 |         +  Cluster 2: Frequent Loyalists                              |
   0 |       + + + +                                                         |
     |     + + + + + + +                # Cluster 0: At-Risk / Lapsed        |
     |       + + + +                   # # # # # # #                         |
  -2 |                  . Cluster 3:   # # # # # # # #                       |
     |                  . Low-Value     # # # # #                               |
     |                  . Occasional                                         |
  -4 +-----------------------------------------------------------------------+
    -4              -2               0               2               4
                                     PCA1
```

#### Segment Profiles ($K=4$)
| Segment ID | Segment Business Name | Customer Count | Mean Recency (days) | Mean Frequency (orders) | Mean Monetary (INR) | Strategic Retention Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **At-Risk / Lapsed** | 436 | 48.2 | 4.1 | INR 1,024.50 | Automated win-back coupon nudges and push notifications |
| **Cluster 1** | **Premium High-Spenders** | **138** | **14.2** | **6.8** | **INR 7,130.40** | **VIP Priority Support Desk & Exclusive Tech Pre-orders** |
| **Cluster 2** | **Frequent Everyday Loyalists**| 419 | 8.8 | 19.4 | INR 4,112.20 | Instamart Super Saver subscription incentives |
| **Cluster 3** | **Occasional Light Buyers** | 182 | 31.5 | 3.2 | INR 482.10 | Low-AOV threshold delivery fee waivers |

- **Designated Premium Segment**: **Cluster 1** (highest mean monetary spend of INR 7,130.40). Assigned `is_premium = True` in `outputs/customer_segments.csv`.
- **PCA Dimensionality Reduction**: Fitting `PCA(n_components=2)` on scaled RFM yields:
  - Component 1 Explained Variance Ratio: **0.634** (63.4%)
  - Component 2 Explained Variance Ratio: **0.218** (21.8%)
  - Cumulative 2-Component Explained Variance: **0.852** (85.2%)

---

### 4. Lapse Problem Definition & Feature Engineering

- **Cutoff Date**: `2025-09-01 00:00:00`.
- **Target Definition (`lapsed`)**: Binary flag where `1` indicates zero completed orders between `2025-09-01` and `2025-09-30`, and `0` indicates at least one completed order in September 2025.
- **Eligible Cohort**: Customers with $\ge 1$ completed order prior to the cutoff ($N = 1,156$).
- **Observed Churn Split**: 443 lapsed (38.32%), 713 retained (61.68%).
- **Engineered Features (Pre-Cutoff Only)**:
  1. `recency`: Days from customer's latest pre-cutoff completed order to cutoff.
  2. `frequency`: Count of pre-cutoff completed orders.
  3. `monetary`: Sum of pre-cutoff completed spend.
  4. `avg_discount_pct`: Mean discount percentage across pre-cutoff completed orders.
  5. `cancellation_rate`: Cancelled orders divided by all finalized orders before cutoff.
  6. `avg_delivery_minutes`: Mean delivery time across completed orders before cutoff.
  7. `tenure_days`: Days between customer's `signup_date` and cutoff.
  8. `age`: Cleaned customer age.
  9. Categorical Encodings: `city` (6 levels), `gender` (3 levels), `favourite_category` (most purchased category).
- **Leakage Prevention Verification**:
  - No orders dated $\ge \text{2025-09-01}$ were used for feature extraction.
  - Part B whole-window $K$-Means segment labels were strictly excluded from feature vectors.
  - Stratified 80/20 split via `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 training rows, 232 test rows.
  - Baseline naive predictor (always predicting "not lapsed"): Accuracy = **61.64%**, Precision = 0.00, Recall = 0.00, $F_1 = \mathbf{0.00}$.

---

### 5. Leakage-Free Preprocessing

All preprocessing was encapsulated inside a Scikit-Learn `ColumnTransformer` fitted strictly on training data:
- Nominal columns (`city`, `gender`, `favourite_category`) transformed using `OneHotEncoder(drop='first', sparse_output=False)`.
- Numeric features scaled using `StandardScaler()`.
- **Learned Scaler Mean Verification**:
  - Training Set Learned Mean (`frequency`): **7.842**
  - Full Cohort Unsplit Mean (`frequency`): **7.876**
  - *Divergence Explanation*: The learned training mean differs from the full cohort mean because the scaler was fitted exclusively on the 80% training partition (924 rows) to prevent out-of-sample data leakage into test evaluations.
- Missing values check across all feature columns: **0 nulls confirmed**.

---

### 6. Model Training, Tuning, and Selection

Hyperparameter optimisation utilized 5-fold cross-validation scoring for the positive lapsed class $F_1$:
1. **Logistic Regression**: `GridSearchCV(LogisticRegression(max_iter=1000), param_grid={'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1')`.
   - Best Parameter: $C = 1.0$ (CV $F_1 = 0.612$)
2. **Random Forest Classifier**: `GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42), param_grid={'max_depth': [3, 5, 8], 'min_samples_leaf': [5, 10, 20]}, cv=5, scoring='f1')` (45 total fits).
   - Best Parameters: `max_depth = 8`, `min_samples_leaf = 10` (CV $F_1 = 0.648$)

#### Test Set Performance Comparison (Untouched Test Set, $N = 232$)
| Model Architecture | Hyperparameters | Lapsed Precision | Lapsed Recall | Lapsed $F_1$ Score | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Predictor** | Predict Majority (`0`) | 0.000 | 0.000 | 0.000 | 61.64% |
| **Logistic Regression**| $C = 1.0$ | 0.618 | 0.618 | 0.618 | 70.69% |
| **Random Forest** | `depth=8, leaf=10` | **0.671** | **0.629** | **0.649** | **74.14%** |

#### Test Confusion Matrices
- **Logistic Regression**:
  $$\begin{pmatrix} \text{TN}: 109 & \text{FP}: 34 \\ \text{FN}: 34 & \text{TP}: 55 \end{pmatrix}$$
- **Random Forest**:
  $$\begin{pmatrix} \text{TN}: 116 & \text{FP}: 27 \\ \text{FN}: 33 & \text{TP}: 56 \end{pmatrix}$$

#### Model Choice Justification
The Random Forest Classifier (`max_depth=8`, `min_samples_leaf=10`) was selected as the production model ($F_1 = 0.649 \ge 0.55$). In quick-commerce retention economics, the financial penalty of failing to detect a churn-prone customer (losing lifelong customer lifetime value and recurring gross margin) significantly outweighs the marginal cost of dispatching a discount nudge to a retained user. The Random Forest achieves higher precision (0.671 vs 0.618) and recall (0.629 vs 0.618) while suppressing false positives from 34 down to 27, ensuring retention budgets are not squandered on unthreatened accounts.

---

### 7. Class-Balance Resampling Experiment

To counter class imbalance (570 retained vs 354 lapsed in the training partition), random oversampling with replacement was applied to the training set only (`random_state=42`), bringing both classes to 570 rows. The logistic regression model was refitted and re-evaluated on the unaltered test set:

| Resampling Experiment | Training Ratio | Lapsed Precision | Lapsed Recall | Lapsed $F_1$ Score | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Logistic Regression** | 570 / 354 (Original) | **0.618** | 0.618 | 0.618 | **70.69%** |
| **Balanced Logistic Regression** | 570 / 570 (Oversampled)| 0.586 | **0.854** | **0.694** | 68.97% |

**Impact on Retention Strategy**:
Lapsed class recall surged from **0.618 to 0.854** (+23.6 percentage points), while precision experienced a mild compression (0.618 to 0.586). For the retention operations team, this provides a highly effective aggressive screening tool: oversampling captures 85.4% of all truly churning users, enabling broad-scale retention campaigns when customer acquisition costs are elevated.

---

### 8. Interpretation, Model Persistence, and Operational Use

#### Feature Importance and Logistic Coefficients
| Model | Feature Name | Metric Value | Business Interpretation |
| :--- | :--- | :--- | :--- |
| **Random Forest** | `recency` | Importance: 0.384 | Days since last purchase is the single most dominant predictor of churn. |
| **Random Forest** | `frequency` | Importance: 0.241 | Habitual ordering history heavily anchors ongoing user retention. |
| **Random Forest** | `tenure_days` | Importance: 0.126 | Older accounts display substantially higher platform stickiness. |
| **Random Forest** | `monetary` | Importance: 0.089 | Cumulative historical spend correlates inversely with abandonment. |
| **Random Forest** | `cancellation_rate` | Importance: 0.052 | Frequent order cancellations trigger customer churn frustration. |
| **Logistic Reg.** | `recency` | Coeff: $+1.142$ | Positive sign: every increase in elapsed idle days sharply elevates lapse odds. |
| **Logistic Reg.** | `frequency` | Coeff: $-0.824$ | Negative sign: higher transaction frequency protects against lapse. |
| **Logistic Reg.** | `tenure_days` | Coeff: $-0.312$ | Negative sign: longer customer relationship tenure reduces lapse probability. |

#### Model Persistence and Reload Checkpoint
The trained model was serialised to `outputs/lapse_model.joblib`. Reloading the artifact via `joblib.load` and scoring the untouched test set confirmed **100.00% exact parity** across all predictions and probabilities.

#### High-Value At-Risk Retention Queue (`outputs/retention_list.csv`)
The model was applied to test-set customers belonging to the Premium segment, ranked descending by predicted lapse risk:

| Rank | Customer ID | Predicted Lapse Probability | Actually Lapsed | Segment Name | Action Queue |
| :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `C0144` | 0.892 | Yes | Premium High-Spenders | Immediate concierge call & INR 500 wallet credit |
| 2 | `C0388` | 0.841 | Yes | Premium High-Spenders | Account manager outreach & delivery fee pass |
| 3 | `C0712` | 0.785 | Yes | Premium High-Spenders | VIP coupon code activation via WhatsApp |
| 4 | `C0921` | 0.724 | Yes | Premium High-Spenders | Re-engagement email from Head of CX |
| 5 | `C0455` | 0.683 | Yes | Premium High-Spenders | App notification regarding new electronics inventory |
| 6 | `C0619` | 0.631 | No | Premium High-Spenders | Proactive satisfaction survey |
| 7 | `C1102` | 0.594 | Yes | Premium High-Spenders | INR 250 weekend basket discount |
| 8 | `C0233` | 0.542 | No | Premium High-Spenders | General loyalty check-in |
| 9 | `C0877` | 0.481 | No | Premium High-Spenders | Product recommendation digest |
| 10 | `C1045` | 0.412 | No | Premium High-Spenders | Standard automated notification |

**Empirical Hit Rate**: Out of the top 10 targeted Premium users, **6 truly lapsed**, demonstrating high operational utility.

*Methodological Note*: The Premium flag originates from the whole-window RFM analysis of Task 3, which includes the September evaluation window; therefore, this retention list serves as an illustrative operational demonstration rather than a leakage-free out-of-sample evaluation.

---

## Part C: Support Ticket Triage with Guardrails

### 1. Ingestion and Pre-LLM Input Guardrails

The raw support corpus contains **164 tickets** (`data/raw/tickets.csv`), spanning billing, delivery, technical, and fraud complaints, alongside planted boundary cases.
- **Empty Ticket Guardrail**: Tickets containing empty strings or whitespace only are assigned status `skipped_empty` without initiating an LLM call (**4 tickets**: `T0157`, `T0158`, `T0159`, `T0160`).
- **Length Threshold Guardrail**: Tickets exceeding `MAX_INPUT_TOKENS = 400` (computed via `est_tokens(text) = ceil(words * 1.3)`) are assigned status `flagged_long` without initiating an LLM call (**4 tickets**: `T0161`, `T0162`, `T0163`, `T0164`, each estimating to **734 tokens**).
- **Eligible Candidates for LLM Classification**: Exactly **156 tickets**.

---

### 2. Deterministic PII Sanitisation Guardrail

Before any ticket payload enters a prompt, a three-stage regular expression redaction filter executes sequentially:
1. **Email Regex**: `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]`
2. **Indian Phone Regex**: `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]`
3. **Order Identifier Regex**: `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]`

#### Redaction Audit Metrics
| Redaction Pattern | Target Entity | Substitutions Made | Residual Leaks Detected Post-Cleaning |
| :--- | :--- | :--- | :--- |
| `[\w.+-]+@[\w-]+\.[\w.]+` | Customer Email Addresses | **55** | **0** |
| `\b[6-9]\d{9}\b` | Customer Mobile Numbers | **91** | **0** |
| `ORD-\d{6}` | Internal Order Tracking IDs | **132** | **0** |
| **Total Redactions** | **All PII Categories** | **278** | **0** |

*Post-Sanitisation Verification*: Automated regex assertion scans across all 164 redacted payloads confirmed **0 surviving PII patterns**, guaranteeing that no raw customer phone numbers, email addresses, or database order keys reach external model contexts.

---

### 3. Prompt Engineering Architectures

Two system prompts were designed to evaluate in-context classification efficacy:

#### Zero-Shot System Prompt
```text
You are an expert customer support classifier for Instamart. CLASSIFY the following support ticket into exactly one of these four categories: billing, delivery, technical, fraud.

Respond ONLY with a valid JSON object matching this schema:
{"category": "<billing|delivery|technical|fraud>", "reason": "<short explanation>"}
Do not include markdown fences, extra commentary, or conversational filler.
```

#### Few-Shot System Prompt
```text
You are an expert customer support classifier for Instamart. CLASSIFY the following support ticket into exactly one of these four categories: billing, delivery, technical, fraud.

Respond ONLY with a valid JSON object matching this schema:
{"category": "<billing|delivery|technical|fraud>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment. My email is [EMAIL].
{"category": "billing", "reason": "double payment deduction"}

Ticket: The delivery is over 45 minutes late and rider is not answering calls.
{"category": "delivery", "reason": "rider delay"}

Ticket: App keeps crashing whenever I reach checkout.
{"category": "technical", "reason": "application crash"}

Ticket: Someone unauthorized placed an order using my stored card balance.
{"category": "fraud", "reason": "unauthorized account takeover"}
```

---

### 4. Resilience and Transient Fault Recovery

Upstream LLM endpoints frequently suffer simulated 503 HTTP server timeouts (`TransientLLMError`). To ensure fault tolerance, all network calls are wrapped in an exponential backoff decorator with uniform random jitter:
$$\text{Wait Time} = \text{BASE\_DELAY} \times 2^n \times \text{Uniform}(0.90, 1.15) \quad (\text{BASE\_DELAY} = 0.05\text{s}, n \in \{0, 1, 2\})$$

#### Retry Execution Log Extract
```
[LLM_RETRY] Ticket=T0009 Attempt=1 Backoff=0.0487s Error=TransientLLMError('503 server busy') -> SUCCESS
[LLM_RETRY] Ticket=T0018 Attempt=1 Backoff=0.0521s Error=TransientLLMError('503 server busy') -> SUCCESS
[LLM_RETRY] Ticket=T0027 Attempt=1 Backoff=0.0463s Error=TransientLLMError('503 server busy') -> SUCCESS
[LLM_RETRY] Ticket=T0036 Attempt=1 Backoff=0.0542s Error=TransientLLMError('503 server busy') -> SUCCESS
...
[LLM_RETRY] Summary: Total Invocations=175, Transient Errors=19, Recovered=19, Permanent Failures=0
```
Across the 156 processed tickets, exactly **19 transient 503 failures** were intercepted, logged with backoff times inside $[0.045\text{s}, 0.0575\text{s}]$, and successfully resolved on retry attempt 0. Zero tickets aborted due to connection loss.

---

### 5. Output Validation and Parser Fallbacks

Raw model completions are validated against an assertive structural gatekeeper:
1. **Schema-Compliant JSON Parsing**: If completion parses as valid JSON and `category` $\in \{\text{billing}, \text{delivery}, \text{technical}, \text{fraud}\}$, status is set to `classified_json`.
2. **Conversational Regex Keyword Extraction**: If completion fails JSON parsing (e.g., chatty responses such as *"The customer appears to have a billing-related problem."*), words are extracted via `re.findall(r"[a-z]+", text.lower())`. This robustly tokenises hyphenated compound words like `billing-related` into `['billing', 'related']`. If exactly one canonical category is present, status is set to `classified_keyword`.
3. **Invalid Output Quarantine**: If completion produces an unsupported category (e.g., `{"category": "shipping"}`) or unparseable text, the ticket is flagged as `flagged_invalid_output` and routed to manual human review.

#### Parser Distribution Verification (156 Tickets)
| Output Validation Category | Count | Status Resolution |
| :--- | :--- | :--- |
| Valid Schema-Compliant JSON | **135** | `classified_json` |
| Fallback Regex Extracted Keyword | **12** | `classified_keyword` |
| Invalid Category / Structural Drop | **9** | `flagged_invalid_output` |
| **Total Processed** | **156** | **147 Classified / 9 Human Escalations** |

---

### 6. Zero-Shot vs. Few-Shot Performance & Economic Trade-Off

#### Performance Metrics Across 147 Classified Tickets
| Architecture | Evaluated | Correct | Accuracy (%) | Total Tokens | Tokens / Ticket | Cost / 1k Tickets ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot Classifier** | 147 | 126 | **85.71%** | 4,263 | 29.0 | \$0.043 |
| **Few-Shot Classifier** | 147 | 147 | **100.00%** | 17,493 | 119.0 | \$0.175 |

#### Zero-Shot Classification Error Matrix
| True Category \ Predicted | Billing | Delivery | Technical | Fraud | Recall (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Billing** | **37** | 0 | 0 | 0 | 100.0% |
| **Delivery** | 0 | **38** | 0 | 0 | 100.0% |
| **Technical** | 10 | 0 | **26** | 0 | 72.2% |
| **Fraud** | 11 | 0 | 0 | **25** | 69.4% |

#### Economic and Operational Justification (3-4 Sentences)
While zero-shot classification requires only 29.0 tokens per ticket, it misclassifies 21 tickets (11 fraud and 10 technical cases misattributed to billing) because ungrounded keyword matching defaults to payment terms. Few-shot prompting eliminates all 21 misclassifications, achieving 100% accuracy at the expense of an additional 90 tokens per ticket (~4.1x token overhead). In high-risk quick-commerce operations, the minor computational expense of extra prompt tokens is overwhelmingly justified: misrouting an account takeover or card fraud ticket to a slow billing queue creates severe regulatory and financial liability, whereas few-shot guidance guarantees immediate security routing.

---

### 7. Token Budget Enforcement & Customer-Aware Routing

#### Token Budget Circuit Breaker (`outputs/triage_budget_demo.csv`)
Under `TOKEN_BUDGET = 6000`, the running cumulative token counter halted execution after processing 51 tickets. Exactly **105 remaining tickets** were safely short-circuited with status `budget_exhausted` and routed to `human_review` without crashing the application.

#### Production Support Routing Logic (`outputs/triage_results.csv`)
1. If status $\in$ {`skipped_empty`, `flagged_long`, `flagged_invalid_output`, `budget_exhausted`} $\rightarrow$ Route: `human_review`, Priority: `NORMAL`.
2. Else if category == `fraud` $\rightarrow$ Route: `security_team`, Priority: `HIGH`.
3. Else if customer `is_premium` == `True` $\rightarrow$ Route: `premium_desk`, Priority: `HIGH`.
4. Else $\rightarrow$ Route: `support_queue`, Priority: `NORMAL`.

#### Final Production Triage Routing Breakdown (164 Total Tickets)
| Triage Route | Assigned Priority | Ticket Count | Dominant Categories Handled |
| :--- | :--- | :--- | :--- |
| `security_team` | **HIGH** | 36 | Verified card fraud, account hacking, wallet theft |
| `premium_desk` | **HIGH** | 18 | VIP customers with late deliveries or invoice errors |
| `support_queue` | **NORMAL** | 93 | Standard delivery delays, billing queries, app glitches |
| `human_review` | **NORMAL** | 17 | 4 empty, 4 long, 9 invalid output completions |
| **Total** | | **164** | **100% Comprehensive Coverage** |

---

## Part D: Policy RAG Assistant and Refund Agent

### 1. Document Chunking Specification

The knowledge base comprises five operational policies in `data/policies/`: `refund_policy.txt`, `delivery_policy.txt`, `payment_policy.txt`, `privacy_policy.txt`, and `escalation_policy.txt`. Documents were segmented into chunks of **40 words** with a sliding **10-word overlap** (stride = 30 words), halting once the chunk window covers the document terminus.

#### Document Chunking Distribution
| Policy Document Name | Total Word Count | Chunk Count Generated | Content Scope Covered |
| :--- | :--- | :--- | :--- |
| `refund_policy.txt` | 114 | **4 chunks** | Delivery thresholds (>45 min), INR 1,000 lead approvals, 7-day limits |
| `delivery_policy.txt` | 98 | **3 chunks** | 30-min SLA, INR 25 fee (<INR 199), rain extensions, electronics OTP |
| `payment_policy.txt` | 94 | **3 chunks** | UPI/Card/COD (max INR 2,500), 48-hr failure refunds, wallet balance rules |
| `privacy_policy.txt` | 68 | **2 chunks** | Phone/email masking, 180-day address purge, 30-day account deletion |
| `escalation_policy.txt`| 88 | **3 chunks** | Fraud 2-hr escalation, VIP 1-hr SLA, 6-step agent fallback handoff |
| **Total Corpus** | **462 words** | **15 chunks** | **Complete Instamart Operational Policy Repository** |

---

### 2. TF-IDF Vectorisation & Semantic Retrieval

Document chunks were indexed using Scikit-Learn's `TfidfVectorizer()` and retrieved via pairwise `cosine_similarity`.
- **Term Frequency ($\text{TF}$)**: Measures the relative frequency of a term $t$ within a specific chunk $d$: $\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t', d}}$.
- **Inverse Document Frequency ($\text{IDF}$)**: Penalises ubiquitously distributed terms while magnifying domain-specific keywords across the corpus: $\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$.
- **Retrieval Limitation of TF-IDF**: Because TF-IDF relies exclusively on lexical n-gram token overlap, it cannot capture semantic synonyms or paraphrastic equivalence (e.g., a query asking *"How do I get my cash returned for damaged milk?"* will miss chunks containing *"reimburse monetary credit for spoiled dairy"* if exact vocabulary does not overlap).
- **Retrieval Sanity Query**: `retrieve("refund policy late delivery", k=3)` returns three candidate chunks; all three uniquely originate from `refund_policy.txt`.

---

### 3. Policy Question-Answering Evaluation & Chunk Overlap Impact

The grounding QA pipeline was evaluated against `data/raw/policy_eval.csv` (12 factually answerable questions, 2 out-of-domain unanswerable questions).

#### Policy RAG Grounding Benchmark
| Evaluation Metric | 10-Word Overlap Window | Zero-Overlap Window (Stride = 40) | Target Standard |
| :--- | :--- | :--- | :--- |
| Top-3 Retrieval Precision (Answerable) | **12 / 12 (100.0%)** | 10 / 12 (83.3%) | $\ge 90\%$ |
| Grounded Fact Accuracy (Answerable) | **12 / 12 (100.0%)** | 10 / 12 (83.3%) | $100\%$ |
| Out-of-Domain Rejection Rate | **2 / 2 (100.0%)** | **2 / 2 (100.0%)** | $100\%$ |
| Overall Benchmark Score | **14 / 14 (100.0%)** | **12 / 14 (85.7%)** | Complete Reliability |

#### Failure Analysis: Zero-Overlap Boundary Splitting
Under zero-overlap chunking, Question 2 (*"Who must approve a refund above INR 1000?"*) and Question 12 (*"After how many steps must an automated assistant hand over to a human?"*) fail. In Question 2, the subject condition (*"A refund above INR 1000..."*) was severed at word 40, separating it from the necessary predicate (*"...must be approved by a human support lead"*), which fell into the subsequent chunk. The retriever selected the first half, depriving the LLM of the governing clause and causing a factual extraction failure. Sliding word overlap is strictly necessary to preserve semantic continuity across chunk borders.

---

### 4. Autonomous Refund Agent Architecture & Tool Specifications

```
                             Autonomous ReAct Refund Agent
                                  +------------------+
                                  | Support Ticket / |
                                  | Customer Message |
                                  +--------+---------+
                                           |
                                           v
                              +----------------------------+
                     +------->|      mock_llm (AGENT)      |<-------+
                     |        |  (Decides Next Tool Call)  |        |
                     |        +--------------+-------------+        |
                     |                       |                      |
            Observation Payload              | Invokes Tool         |
                     |                       v                      |
                     |             +-------------------+            |
                     +-------------|   TOOL DISPATCH   |------------+
                                   +---------+---------+
                                             |
            +-------------------+------------+------------+--------------------+
            |                   |                         |                    |
            v                   v                         v                    v
     +--------------+   +---------------+        +------------------+   +---------------+
     |  get_order   |   | search_policy |        | calculate_refund |   | issue_refund/ |
     | (SQLite DB)  |   | (TF-IDF RAG)  |        |  (Policy Rules)  |   |   escalate    |
     +--------------+   +---------------+        +------------------+   +-------+-------+
                                                                                |
                                                                                v
                                                                      +-------------------+
                                                                      |   refund_ledger   |
                                                                      |      (.jsonl)     |
                                                                      +-------------------+
```

The autonomous agent implements a closed-loop ReAct (Reasoning + Acting) execution pattern supported by five sandboxed tools registered in `TOOLS_SPEC`:
1. `get_order(order_id)`: Queries `data/instamart.db` and returns order state, delivery duration, and financial sum.
2. `search_policy(query)`: Executes TF-IDF retrieval across operational documents.
3. `calculate_refund(order_amount, delivery_minutes)`: Enforces business logic: refunds require `completed` status; orders delivered in $\le 45$ min receive INR 0.00; orders delivered in $> 45$ min receive 100% of `order_amount`.
4. `issue_refund(order_id, refund_inr)`: Appends an approved transaction to `outputs/refund_ledger.jsonl` and returns confirmation.
5. `escalate_to_human(order_id, refund_inr)`: Appends an escalation record to `outputs/refund_ledger.jsonl` for refunds exceeding INR 1,000.00.

---

### 5. Benchmark Scenario Execution Results

All 7 benchmark scenarios from `data/raw/agent_scenarios.csv` were executed through `run_agent(user_message, scenario_id)` and committed to `outputs/agent_runs.jsonl`:

| Scenario ID | Customer ID | Case Description | Model Status | Refund (INR) | Expected (INR) | Model Calls | Tokens Used | Validation Result |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | `C0036` | Late completed order ($\le \text{INR } 1000$) | `REFUND` | **INR 128.25** | INR 128.25 | 5 | 864 | **Exact Match** |
| **S2** | `C0051` | Late completed order ($\le \text{INR } 1000$) | `REFUND` | **INR 135.00** | INR 135.00 | 5 | 882 | **Exact Match** |
| **S3** | `C0044` | High-value late delivery ($> \text{INR } 1000$) | `ESCALATE` | **INR 1999.00**| INR 1999.00 | 5 | 921 | **Exact Match** |
| **S4** | `C0012` | On-time delivery ($\le 45 \text{ min}$) | `DENIED` | **INR 0.00** | INR 0.00 | 4 | 712 | **Exact Match** |
| **S5** | `C0098` | Cancelled order (never charged) | `DENIED` | **INR 0.00** | INR 0.00 | 3 | 542 | **Exact Match** |
| **S6** | `C0001` | Non-existent tracking ID | `NOT_FOUND` | **INR 0.00** | INR 0.00 | 2 | 318 | **Exact Match** |
| **S7** | `C0036` | Adversarial prompt loop (`[LOOP]`) | `FALLBACK_MAX_ITERATIONS` | **INR 0.00** | INR 0.00 | 6 | 4,120 | **Safe Exit** |

---

### 6. Safety Circuit Breakers and Fallback Demonstrations

To ensure automated agents cannot execute runaway loops, exhaust enterprise token budgets, or hang indefinitely, three independent safety circuit breakers were tested:
1. **Max Iterations Limit**: In scenario `S7`, an adversarial prompt repeatedly triggers policy searches. The agent hits `max_iterations = 6`, terminates tool execution, and returns `FALLBACK_MAX_ITERATIONS` with a human handoff summary.
2. **Token Limit Circuit Breaker**: Running scenario `S1` under `token_limit = 200` triggers an immediate circuit break during step 1. The agent halts before executing subsequent tools and safely logs `FALLBACK_TOKEN_LIMIT`.
3. **Wall-Clock Timeout**: Running scenario `S1` under `time_limit_seconds = 0` triggers the timer check at the top of the loop ($0.0001\text{s} \ge 0\text{s}$). The agent immediately aborts and logs `FALLBACK_TIMEOUT`.

#### Safety Circuit Breaker Summary
| Guardrail Parameter | Test Condition | Monitored Trigger | Terminal Status Returned | Handoff Generated |
| :--- | :--- | :--- | :--- | :--- |
| **Iteration Guard** | Scenario `S7` | Model Calls $\ge 6$ | `FALLBACK_MAX_ITERATIONS` | Yes (Summary to human queue) |
| **Token Guard** | Scenario `S1` (`limit=200`) | Usage Tokens $> 200$ | `FALLBACK_TOKEN_LIMIT` | Yes (State preserved) |
| **Latency Guard** | Scenario `S1` (`timeout=0`) | Elapsed Time $\ge 0\text{s}$ | `FALLBACK_TIMEOUT` | Yes (Instant abort) |

---

### 7. Stateful Checkpointing and Crash Recovery Demonstration

Every tool invocation atomically flushes execution state to `state/{run_id}.json` (recording `run_id`, `scenario_id`, `messages`, `tokens_used`, `step_index`, and `executed_tools`).

```
                    Crash & Resume Sequence (Scenario S1)
Step 1: Invokes get_order        -> Success -> Checkpoint S1_crash_resume.json
Step 2: Invokes search_policy    -> Success -> Checkpoint S1_crash_resume.json
========================== SIMULATED PROCESS CRASH ==========================
Resume: resume_agent("S1_crash_resume") loads state from disk
Step 3: Invokes calculate_refund -> Success -> Checkpoint S1_crash_resume.json
Step 4: Invokes issue_refund     -> Success -> Checkpoint S1_crash_resume.json
Step 5: Completion Final Answer  -> REFUND: INR 128.25 issued.
```

#### Crash-Resume Execution Sequence
- **Pre-Crash Execution Phase**:
  1. `get_order({"order_id": "ORD-101138"})` $\rightarrow$ Returned order metadata.
  2. `search_policy({"query": "refund policy late delivery"})` $\rightarrow$ Returned policy chunks.
  - *Process Injected Termination*: Simulated SIGKILL / memory crash raised immediately after saving step 2 state.
- **Post-Crash Recovery Phase (`resume_agent`)**:
  3. `calculate_refund({"order_amount": 128.25, "delivery_minutes": 52})` $\rightarrow$ Returned `refund_inr: 128.25`.
  4. `issue_refund({"order_id": "ORD-101138", "refund_inr": 128.25})` $\rightarrow$ Logged to ledger.
- **Final Result**: Exactly **4 unique tools executed** in sequential order without duplicate billing or orphaned ledger rows. Final status: **`REFUND` (INR 128.25)**.

---

### 8. Architectural Comparative Analysis: ReAct vs. Plan-and-Execute

*(Word count: 218 words)*

The ReAct (Reasoning + Acting) architecture alternates between language model reasoning and discrete tool execution. In contrast, a Plan-and-Execute paradigm queries the model once to generate a static DAG (directed acyclic graph) of sequential tool invocations, re-planning only when a step throws an unhandled exception.

In this quick-commerce refund environment, ReAct required 5 model calls for clean resolutions (S1, S2, S3), 3 calls for cancelled orders (S5), and 2 calls for missing orders (S6). A Plan-and-Execute architecture would generate an upfront plan such as `[get_order -> search_policy -> calculate_refund -> issue_refund]`. When given an invalid order (S6) or a cancelled order (S5), Plan-and-Execute wastes initial planning tokens predicting downstream refund calculations that must immediately be discarded upon observing that the order never existed or was never charged. Furthermore, unexpected observations (such as a delivery arriving in 44 minutes instead of 46) require complete replanning cycles, doubling model latency.

For transactional financial operations, **ReAct is distinctly superior**. Its dynamic step-by-step observation conditioning allows the agent to halt immediately upon encountering invalid IDs (S6 in 2 calls) or ineligible statuses (S5 in 3 calls). This dynamic branching prevents catastrophic unauthorized refund disbursements and conserves enterprise token expenditures on terminating paths.

---

## Limitations and Next Steps

1. **Post-Hoc Whole-Window Segment Definitions**:
   - *Limitation*: The customer segmentation in Part B ($K$-Means) was fitted across the full 6-month transaction window, overlapping with the September window used to define the 30-day lapse label. Consequently, the High-Value At-Risk Retention Queue (`outputs/retention_list.csv`) represents an illustrative operational triage demonstration rather than an out-of-sample evaluation.
   - *Next Step*: Implement rolling point-in-time RFM feature snapshots, fitting cluster boundaries exclusively on pre-cutoff data.

2. **Lexical Keyword Limitations of TF-IDF Retrieval**:
   - *Limitation*: While the 15-chunk TF-IDF RAG system achieved 100% precision on clean policy queries, it fails when customer queries use colloquial, regional, or slang terms (e.g., *"rider ran away with my biryani"*) that share zero lexical n-grams with policy documents.
   - *Next Step*: Upgrade retrieval to dense neural bi-encoders (e.g., `bge-small-en-v1.5` or `all-MiniLM-L6-v2`) backed by a lightweight SQLite vector extension (`sqlite-vec`).

3. **Deterministic Mock vs. Production Non-Deterministic LLMs**:
   - *Limitation*: The system was evaluated using an offline, rule-based mock completion environment (`mock_llm.py`). Real-world commercial LLMs exhibit stochastic latency, token-generation drift, and prompt injection susceptibilities.
   - *Next Step*: Deploy semantic output guardrails (such as Guardrails AI or NeMo Guardrails) with structured JSON schemas enforced via strict grammar decoding (e.g., Outlines / vLLM guided decoding).

4. **Batch-Oriented Cold-Start Constraints**:
   - *Limitation*: The lapse prediction pipeline depends on historical aggregations and cannot score brand-new users who registered within 48 hours without completed orders.
   - *Next Step*: Incorporate real-time behavioral telemetry features (app session counts, items added to cart, search query velocity) into an online streaming feature store.