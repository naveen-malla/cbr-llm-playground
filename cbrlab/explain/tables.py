import pandas as pd

def per_feature_table(per_feature: dict) -> pd.DataFrame:
    return pd.DataFrame({"feature": list(per_feature.keys()), "local_similarity": list(per_feature.values())})
