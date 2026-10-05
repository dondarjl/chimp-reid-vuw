"""
Shared parsing logic for the Freytag et al. 2016 annotation format, used by
both czoo_to_manifest.py and ctai_to_manifest.py. Not a dataset on its own —
just avoids duplicating the same parser twice.
"""
import pandas as pd


# _freytag_common.py

_NON_IDENTITY_LABELS = {"Adult"}  # Freytag Age_Group value, not a real individual —
                                    # confirmed by qualitative error analysis: three
                                    # "Adult" query images showed visibly distinct
                                    # individuals (different facial structure, ear
                                    # shape, skin tone), and CZoo's manifest has no
                                    # equivalent contamination, so this is isolated
                                    # to CTai's annotation format only.

def parse_freytag_annotations(annotation_path: str) -> pd.DataFrame:
    rows = []
    excluded_non_identity = 0
    with open(annotation_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.split()
            filename = fields[1]
            name = fields[fields.index("Name") + 1]
            if name in _NON_IDENTITY_LABELS:
                excluded_non_identity += 1
                continue
            rows.append({"filepath": filename, "identity": name})
    if excluded_non_identity:
        print(f"Excluding {excluded_non_identity} rows with non-identity Name label "
              f"(age-group placeholder, not a real individual): {_NON_IDENTITY_LABELS}")
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, test_fraction: float, seed: int,
                      min_images_per_identity: int = 4) -> pd.DataFrame:
    """80/20-style stratified holdout, per identity, seeded for reproducibility.

    Identities with fewer than `min_images_per_identity` images are dropped
    entirely — with fewer than 2 images there is no way to have both a train
    and a test example, so no split (stratified or otherwise) can place them
    validly. This is a data-quality decision, made explicit and reported,
    rather than a silent artifact of the split arithmetic.
    """
    df = df.copy()
    counts = df.groupby("identity").size()
    too_few = counts[counts < min_images_per_identity].index.tolist()
    if too_few:
        print(f"Excluding {len(too_few)} identities with < {min_images_per_identity} "
              f"images (cannot support both a train and test example): {too_few}")
    df = df[~df["identity"].isin(too_few)].copy()

    df["split"] = "train"
    for identity, group in df.groupby("identity"):
        n = len(group)
        n_test = max(1, round(test_fraction * n))
        n_test = min(n_test, n - 1)  # always leave >= 1 image for train
        test_idx = group.sample(n=n_test, random_state=seed).index
        df.loc[test_idx, "split"] = "test"
    return df
