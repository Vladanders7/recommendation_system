import pathlib
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample"

FEED_CSV = DATA_DIR / "feed_sample_lastNdays.csv"
USER_CSV = DATA_DIR / "user_data_sample.csv"
POSTS_CSV = DATA_DIR / "post_text_sample.csv"

OUT_USER = DATA_DIR / "user_features.csv"
OUT_POST = DATA_DIR / "post_features.csv"

ALPHA = 1000.0


def compute_post_ctr_smooth(feed: pd.DataFrame, alpha: float = 1000.0) -> pd.DataFrame:
    tmp = feed.copy()
    tmp["is_like"] = (tmp["action"] == "like").astype(int)

    post_agg = tmp.groupby("post_id", as_index=False).agg(
        views=("action", "count"),
        likes=("is_like", "sum"),
    )
    global_ctr = post_agg["likes"].sum() / max(1, post_agg["views"].sum())
    post_agg["post_ctr_smooth"] = (post_agg["likes"] + alpha * global_ctr) / (post_agg["views"] + alpha)
    return post_agg[["post_id", "post_ctr_smooth"]]


def main():
    feed = pd.read_csv(FEED_CSV)
    users = pd.read_csv(USER_CSV)
    posts = pd.read_csv(POSTS_CSV)

    # user_features
    user_cols = ["user_id", "age", "city", "country", "exp_group", "gender", "os", "source"]
    users_feat = users[user_cols].drop_duplicates("user_id").copy()
    users_feat.to_csv(OUT_USER, index=False)

    # post_features
    posts_feat = posts[["post_id", "topic", "text"]].copy()
    posts_feat["text_len"] = posts_feat["text"].fillna("").str.len().astype(int)
    posts_feat = posts_feat.drop(columns=["text"])

    ctr = compute_post_ctr_smooth(feed, alpha=ALPHA)
    posts_feat = posts_feat.merge(ctr, on="post_id", how="left")
    posts_feat["post_ctr_smooth"] = posts_feat["post_ctr_smooth"].fillna(0).astype(float)

    posts_feat.to_csv(OUT_POST, index=False)

    print("[OK] saved:")
    print(" -", OUT_USER)
    print(" -", OUT_POST)


if __name__ == "__main__":
    main()