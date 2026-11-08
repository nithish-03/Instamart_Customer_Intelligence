import pandas as pd

segments = pd.read_csv("outputs/customer_segments.csv")
retention = pd.read_csv("outputs/retention_list.csv")

print("Customer segments shape:", segments.shape)
print("Retention list shape:", retention.shape)

print("\nPremium distribution:")
print(segments["is_premium"].value_counts(dropna=False))

print("\nMean monetary value by segment:")
print(
    segments.groupby(["segment_id", "segment_name"])
    .agg(
        mean_monetary=("monetary", "mean"),
        premium_flag=("is_premium", "max"),
        customer_count=("customer_id", "count")
    )
    .sort_values("mean_monetary", ascending=False)
)