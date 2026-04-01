
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})



df_mkt    = pd.read_csv("customer_marketing_data.csv")
df_orders = pd.read_csv("Clients-and-Order.csv")
df_rfm    = pd.read_csv("RFM-clients_new.csv")

print("=" * 60)
print("DATASET SHAPES")
print("=" * 60)
print(f"  customer_marketing_data : {df_mkt.shape}")
print(f"  Clients-and-Order       : {df_orders.shape}")
print(f"  RFM-clients_new         : {df_rfm.shape}")



print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
for name, df in [("customer_marketing_data", df_mkt),
                 ("Clients-and-Order", df_orders),
                 ("RFM-clients_new", df_rfm)]:
    print(f"\n  {name}: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("MISSING VALUES – customer_marketing_data")
print("=" * 60)
print(df_mkt.isnull().sum())

print("\n" + "=" * 60)
print("MISSING VALUES – RFM-clients_new")
print("=" * 60)
print(df_rfm.isnull().sum())


df = df_rfm.merge(df_mkt, on="client_id", how="left")
print(f"\nMerged dataset shape: {df.shape}")



print("\n" + "=" * 60)
print("SUMMARY STATISTICS – KEY VARIABLES")
print("=" * 60)
key_cols = ["recency", "frequency", "monetary", "CLV",
            "email_opens", "email_clicks", "website_visits", "age", "income"]
print(df[key_cols].describe().round(2).to_string())


impute_cols = ["website_visits", "email_opens", "age", "income"]
for col in impute_cols:
    median_val = df[col].median()
    n_missing  = df[col].isnull().sum()
    df[col]    = df[col].fillna(median_val)
    print(f"  Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

print("\n" + "=" * 60)
print("T1 – DATA EXPLORATION")
print("=" * 60)


print("\nCustomer count by region:")
print(df["region"].value_counts())

# ── 6b. Acquisition channel distribution ─────────────────────────────────────
print("\nAcquisition channel distribution:")
print(df["acquisition_channel"].value_counts())

# ── 6c. Discount usage ────────────────────────────────────────────────────────
print("\nDiscount usage (0=No, 1=Yes):")
print(df["discount_used"].value_counts())
disc_pct = df["discount_used"].mean() * 100
print(f"  → {disc_pct:.1f}% of customers used a discount")

# ── 6d. Correlations ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORRELATION MATRIX – email engagement vs spend/value")
print("=" * 60)
corr_cols = ["email_opens", "email_clicks", "monetary", "CLV",
             "website_visits", "frequency", "recency"]
print(df[corr_cols].corr().round(3).to_string())

# ── 6e. Discount users vs non-discount users ──────────────────────────────────
print("\n" + "=" * 60)
print("DISCOUNT USAGE vs CUSTOMER VALUE")
print("=" * 60)
disc_summary = df.groupby("discount_used")[["CLV", "monetary", "frequency"]].mean().round(2)
disc_summary.index = ["No Discount", "Used Discount"]
print(disc_summary.to_string())

# ── 6f. Acquisition channel vs CLV ────────────────────────────────────────────
print("\n" + "=" * 60)
print("ACQUISITION CHANNEL vs CLV / MONETARY")
print("=" * 60)
ch_summary = df.groupby("acquisition_channel").agg(
    count     = ("CLV", "count"),
    avg_CLV   = ("CLV", "mean"),
    avg_spend = ("monetary", "mean"),
    avg_freq  = ("frequency", "mean")
).round(2)
print(ch_summary.to_string())

# ── 6g. Region vs discount usage ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("DISCOUNT RATE BY REGION")
print("=" * 60)
print(df.groupby("region")["discount_used"].mean().round(3).mul(100).rename("discount_%"))

# 7. VISUALISATIONS – EDA

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("NovaRetail – Exploratory Data Analysis", fontsize=14, fontweight="bold")

# Frequency histogram
axes[0, 0].hist(df["frequency"], bins=20, color="#0e7c86", edgecolor="white")
axes[0, 0].set_title("Purchase Frequency Distribution")
axes[0, 0].set_xlabel("Frequency"); axes[0, 0].set_ylabel("Count")

# Monetary histogram
axes[0, 1].hist(df["monetary"], bins=20, color="#1a2e4a", edgecolor="white")
axes[0, 1].set_title("Monetary Spend Distribution (€)")
axes[0, 1].set_xlabel("Spend (€)"); axes[0, 1].set_ylabel("Count")

# CLV histogram
axes[0, 2].hist(df["CLV"], bins=20, color="#e07b39", edgecolor="white")
axes[0, 2].set_title("CLV Distribution")
axes[0, 2].set_xlabel("CLV (€)"); axes[0, 2].set_ylabel("Count")

# Email opens vs CLV scatter
axes[1, 0].scatter(df["email_opens"], df["CLV"], alpha=0.4, color="#0e7c86", s=20)
axes[1, 0].set_title(f"Email Opens vs CLV  (r={df['email_opens'].corr(df['CLV']):.3f})")
axes[1, 0].set_xlabel("Email Opens"); axes[1, 0].set_ylabel("CLV (€)")

# CLV by acquisition channel
ch_clv = df.groupby("acquisition_channel")["CLV"].mean().sort_values()
colors_bar = ["#c0392b", "#e07b39", "#f1c40f", "#2a9d5c", "#1a2e4a"]
axes[1, 1].barh(ch_clv.index, ch_clv.values, color=colors_bar)
axes[1, 1].set_title("Avg CLV by Acquisition Channel")
axes[1, 1].set_xlabel("Avg CLV (€)")

# Discount rate by region
reg_disc = df.groupby("region")["discount_used"].mean().sort_values() * 100
reg_colors = ["#2a9d5c", "#0e7c86", "#e07b39", "#c0392b"]
axes[1, 2].bar(reg_disc.index, reg_disc.values, color=reg_colors)
axes[1, 2].set_title("Discount Usage by Region (%)")
axes[1, 2].set_ylabel("Discount Rate (%)")
axes[1, 2].set_ylim(0, 65)

plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches="tight")
print("\nEDA plots saved → eda_plots.png")
plt.show()

# Income vs Monetary scatter
plt.figure(figsize=(7, 5))
plt.scatter(df["income"], df["monetary"], alpha=0.4, color="#1a2e4a", s=20)
r_inc_mon = df["income"].corr(df["monetary"])
plt.title(f"Income vs Monetary Spend  (r={r_inc_mon:.3f})")
plt.xlabel("Income (€)"); plt.ylabel("Monetary Spend (€)")
plt.tight_layout()
plt.savefig("income_vs_spend.png", bbox_inches="tight")
print("Income vs spend plot saved → income_vs_spend.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 7))
corr_mat = df[["recency", "frequency", "monetary", "CLV",
               "email_opens", "email_clicks", "website_visits",
               "income", "age", "discount_used"]].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, mask=mask, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap – Key Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
print("Correlation heatmap saved → correlation_heatmap.png")
plt.show()

#=
# 8. T2 – CUSTOMER SEGMENTATION (K-MEANS)


print("\n" + "=" * 60)
print("T2 – CUSTOMER SEGMENTATION")
print("=" * 60)

cluster_features = ["recency", "frequency", "monetary",
                    "website_visits", "email_opens", "discount_used"]

df_clust = df[cluster_features].copy()
print(f"\nClustering on features: {cluster_features}")
print(f"NaN values after imputation: {df_clust.isnull().sum().sum()}")

scaler = StandardScaler()
X      = scaler.fit_transform(df_clust)

# ── Silhouette & elbow analysis ───────────────────────────────────────────────
print("\nSilhouette scores and inertia by k:")
sil_scores = {}
inertias   = {}

for k in range(2, 8):
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_scores[k] = round(silhouette_score(X, labels), 4)
    inertias[k]   = round(km.inertia_, 1)
    print(f"  k={k}  silhouette={sil_scores[k]}  inertia={inertias[k]}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n→ Optimal k by silhouette: {best_k}  (score={sil_scores[best_k]})")

# Silhouette & elbow plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(sil_scores.keys()), list(sil_scores.values()),
         marker="o", color="#0e7c86", linewidth=2)
ax1.axvline(x=4, color="red", linestyle="--", alpha=0.6, label="Selected k=4")
ax1.set_title("Silhouette Score by k"); ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette")
ax1.legend()

ax2.plot(list(inertias.keys()), list(inertias.values()),
         marker="s", color="#1a2e4a", linewidth=2)
ax2.axvline(x=4, color="red", linestyle="--", alpha=0.6, label="Selected k=4")
ax2.set_title("Elbow Curve (Inertia) by k"); ax2.set_xlabel("k"); ax2.set_ylabel("Inertia")
ax2.legend()

plt.tight_layout()
plt.savefig("cluster_selection.png", bbox_inches="tight")
print("Cluster selection plots saved → cluster_selection.png")
plt.show()

# ── Final model: k=4 ──────────────────────────────────────────────────────────
km_final  = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X)

# ── Cluster profiles ──────────────────────────────────────────────────────────
profile = df.groupby("cluster").agg(
    size               = ("client_id",     "count"),
    avg_recency        = ("recency",        "mean"),
    avg_frequency      = ("frequency",      "mean"),
    avg_monetary       = ("monetary",       "mean"),
    avg_CLV            = ("CLV",            "mean"),
    avg_email_opens    = ("email_opens",    "mean"),
    avg_email_clicks   = ("email_clicks",   "mean"),
    avg_website_visits = ("website_visits", "mean"),
    pct_discount       = ("discount_used",  "mean"),
    avg_income         = ("income",         "mean"),
    avg_age            = ("age",            "mean"),
).round(2)

print("\n" + "=" * 60)
print("CLUSTER PROFILES (k=4)")
print("=" * 60)
print(profile.to_string())

# ── Segment labels ────────────────────────────────────────────────────────────
#   (based on profile inspection)
#   Cluster 0 → Discount Hunters   (100% discount, low CLV)
#   Cluster 1 → Engaged Potentials (high email, medium CLV)
#   Cluster 2 → Champions          (high freq, highest CLV)
#   Cluster 3 → Dormant            (low everything, largest group)

segment_map = {
    0: "Discount Hunters",
    1: "Engaged Potentials",
    2: "Champions",
    3: "Dormant",
}
df["segment"] = df["cluster"].map(segment_map)

print("\nSegment sizes:")
print(df["segment"].value_counts())

# ── Segment visualisations ────────────────────────────────────────────────────
seg_order  = ["Champions", "Engaged Potentials", "Discount Hunters", "Dormant"]
seg_colors = {"Champions": "#2a9d5c", "Engaged Potentials": "#0e7c86",
              "Discount Hunters": "#e07b39", "Dormant": "#888888"}
palette    = [seg_colors[s] for s in seg_order]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("NovaRetail – Segment Profiles", fontsize=14, fontweight="bold")

metrics = [
    ("avg_CLV",            "Avg CLV (€)"),
    ("avg_monetary",       "Avg Monetary Spend (€)"),
    ("avg_frequency",      "Avg Purchase Frequency"),
    ("avg_recency",        "Avg Recency (days)"),
    ("avg_email_opens",    "Avg Email Opens"),
    ("pct_discount",       "% Discount Usage"),
]
seg_profile = df.groupby("segment").agg(
    avg_CLV            = ("CLV",           "mean"),
    avg_monetary       = ("monetary",      "mean"),
    avg_frequency      = ("frequency",     "mean"),
    avg_recency        = ("recency",       "mean"),
    avg_email_opens    = ("email_opens",   "mean"),
    pct_discount       = ("discount_used", "mean"),
).loc[seg_order].round(2)

for ax, (col, label) in zip(axes.flat, metrics):
    vals   = seg_profile[col].values
    bars   = ax.bar(seg_order, vals, color=palette, edgecolor="white")
    ax.set_title(label, fontsize=10)
    ax.set_xticklabels(seg_order, rotation=20, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("segment_profiles.png", bbox_inches="tight")
print("Segment profile plots saved → segment_profiles.png")
plt.show()

# Scatter: Frequency vs Monetary coloured by segment
plt.figure(figsize=(8, 6))
for seg, grp in df.groupby("segment"):
    plt.scatter(grp["frequency"], grp["monetary"],
                label=seg, color=seg_colors[seg], alpha=0.6, s=25)
plt.title("Frequency vs Monetary Spend – by Segment")
plt.xlabel("Purchase Frequency"); plt.ylabel("Monetary Spend (€)")
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig("scatter_freq_monetary.png", bbox_inches="tight")
print("Scatter plot saved → scatter_freq_monetary.png")
plt.show()

# 9. T3 – MARKETING EFFECTIVENESS


print("\n" + "=" * 60)
print("T3 – MARKETING EFFECTIVENESS")
print("=" * 60)

# ── 9a. Email engagement vs CLV ───────────────────────────────────────────────
print("\n9a. Correlation of email metrics with CLV and monetary:")
for col in ["email_opens", "email_clicks", "website_visits"]:
    r_clv = df[col].corr(df["CLV"])
    r_mon = df[col].corr(df["monetary"])
    print(f"  {col:20s}  CLV r={r_clv:.3f}   Monetary r={r_mon:.3f}")

# Quintile analysis
df["email_quintile"] = pd.qcut(df["email_opens"], q=5,
                                labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
quintile_clv = df.groupby("email_quintile", observed=True)["CLV"].mean().round(0)
print("\nAvg CLV by email opens quintile:")
print(quintile_clv.to_string())

# ── 9b. Discount analysis ─────────────────────────────────────────────────────
print("\n9b. Discount users vs non-discount users:")
disc_detail = df.groupby("discount_used").agg(
    count    = ("CLV",      "count"),
    avg_CLV  = ("CLV",      "mean"),
    avg_mon  = ("monetary", "mean"),
    avg_freq = ("frequency","mean"),
).round(2)
disc_detail.index = ["No Discount", "Discount Used"]
print(disc_detail.to_string())

print("\nDiscount dependency by segment:")
seg_disc = df.groupby("segment")[["discount_used", "CLV", "monetary"]].mean().round(2)
print(seg_disc.loc[seg_order].to_string())

# ── 9c. Acquisition channel analysis ─────────────────────────────────────────
print("\n9c. Acquisition channel performance:")
ch_full = df.groupby("acquisition_channel").agg(
    count    = ("CLV",      "count"),
    avg_CLV  = ("CLV",      "mean"),
    avg_mon  = ("monetary", "mean"),
    avg_freq = ("frequency","mean"),
).sort_values("avg_CLV", ascending=False).round(2)
print(ch_full.to_string())

# Marketing effectiveness visualisations
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("NovaRetail – Marketing Effectiveness", fontsize=13, fontweight="bold")

# Email quintile CLV
axes[0].bar(quintile_clv.index, quintile_clv.values, color="#0e7c8680",
            edgecolor="#0e7c86")
axes[0].set_title("Avg CLV by Email Opens Quintile")
axes[0].set_xlabel("Email Opens Quintile"); axes[0].set_ylabel("Avg CLV (€)")
axes[0].tick_params(axis="x", rotation=30)

# Discount vs no discount CLV
labels = ["No Discount", "Discount Used"]
clv_vals = [df[df["discount_used"]==0]["CLV"].mean(),
            df[df["discount_used"]==1]["CLV"].mean()]
axes[1].bar(labels, clv_vals, color=["#2a9d5c", "#c0392b"])
axes[1].set_title("Avg CLV: Discount vs No Discount")
axes[1].set_ylabel("Avg CLV (€)")
for i, v in enumerate(clv_vals):
    axes[1].text(i, v + 30, f"€{v:.0f}", ha="center", fontsize=10)

# Channel CLV
ch_sorted = ch_full["avg_CLV"].sort_values()
bar_colors = ["#c0392b", "#e07b39", "#f1c40f", "#2a9d5c", "#1a2e4a"]
axes[2].barh(ch_sorted.index, ch_sorted.values, color=bar_colors)
axes[2].set_title("Avg CLV by Acquisition Channel")
axes[2].set_xlabel("Avg CLV (€)")

plt.tight_layout()
plt.savefig("marketing_effectiveness.png", bbox_inches="tight")
print("\nMarketing effectiveness plots saved → marketing_effectiveness.png")
plt.show()


# 10. FINAL SUMMARY PRINTOUT


print("\n" + "=" * 60)
print("FINAL SEGMENT SUMMARY TABLE")
print("=" * 60)
final_table = df.groupby("segment").agg(
    size      = ("client_id",    "count"),
    avg_CLV   = ("CLV",          "mean"),
    avg_spend = ("monetary",     "mean"),
    avg_freq  = ("frequency",    "mean"),
    disc_pct  = ("discount_used","mean"),
    email_avg = ("email_opens",  "mean"),
).loc[seg_order].round(1)
final_table["disc_pct"] = (final_table["disc_pct"] * 100).round(1)
final_table.columns = ["Size","Avg CLV (€)","Avg Spend (€)","Avg Freq","Discount %","Email Opens"]
print(final_table.to_string())

print("\n" + "=" * 60)
print("ACQUISITION CHANNEL SUMMARY")
print("=" * 60)
print(ch_full.to_string())

print("\n" + "=" * 60)
print("KEY CORRELATIONS")
print("=" * 60)
for pair, cols in [
    ("Email opens ↔ CLV",      ("email_opens",  "CLV")),
    ("Email clicks ↔ CLV",     ("email_clicks", "CLV")),
    ("Discount used ↔ CLV",    ("discount_used","CLV")),
    ("Monetary ↔ CLV",         ("monetary",     "CLV")),
    ("Income ↔ Monetary",      ("income",       "monetary")),
    ("Frequency ↔ CLV",        ("frequency",    "CLV")),
]:
    r = df[cols[0]].corr(df[cols[1]])
    print(f"  {pair:30s}  r = {r:.3f}")


df.to_csv("df_clustered_with_segments.csv", index=False)
print("\n✅ Clustered dataset exported → df_clustered_with_segments.csv")
print("✅ Analysis complete.")
