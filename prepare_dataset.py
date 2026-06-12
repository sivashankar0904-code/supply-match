'''
Load products.parquet → extract product_id, product_title
Load examples.parquet → filter where esci_label == "E" (exact match)
Join on product_id
Output columns: variant_name (query), master_id (product_id), master_name (product_title)
Save to data/processed/entity_pairs.csv
'''

import pandas as pd
df_products = pd.read_parquet("data/raw/esci/shopping_queries_dataset_products.parquet").columns.tolist()
df_examples = pd.read_parquet("data/raw/esci/shopping_queries_dataset_examples.parquet").columns.tolist()

df_products = pd.read_parquet("data/raw/esci/shopping_queries_dataset_products.parquet", columns=["product_id", "product_title"])
df_examples = pd.read_parquet("data/raw/esci/shopping_queries_dataset_examples.parquet", columns=["query", "product_id", "esci_label"])
df_examples = df_examples[df_examples["esci_label"] == "E"]
df_merged = pd.merge(df_examples, df_products, on="product_id", how="inner")
df_merged = df_merged[["query", "product_id", "product_title"]]
df_merged.columns = ["variant_name", "master_id", "master_name"]
# df_merged.to_csv("data/processed/entity_pairs.csv", index=False)
'''
Problem — same variant_name mapping to multiple master_name.
ESCI "E" label means "exact enough for search result", not strict 1-to-1 entity match.
Fix options:

Use query_id + product_id as unique pair — treat each as independent training example (fine for embedding training, not for strict matching)
'''
df_merged["query_product_pair"] = df_merged["variant_name"] + "||" + df_merged["master_id"]
df_merged = df_merged.drop_duplicates(subset=["query_product_pair"])
df_merged = df_merged[["variant_name", "master_id", "master_name"]]
df_merged.to_csv("data/processed/entity_pairs.csv", index=False)
