#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================
DATA_DIR = os.getenv("DATA_DIR", "data/sample")
SEED = int(os.getenv("SEED", "42"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.2"))

ITERATIONS = int(os.getenv("ITERATIONS", "400"))
ALPHA = float(os.getenv("ALPHA", "1000.0"))

K = int(os.getenv("K", "5"))

# Limit per-user candidates for speed (sample if too many)
MAX_CAND_PER_USER = int(os.getenv("MAX_CAND_PER_USER", "300"))
MAX_USERS_EVAL = int(os.getenv("MAX_USERS_EVAL", "2000"))  # cap users for speed


# =========================
# Feature schema (match app/train)
# =========================
CAT_COLS = [
    "city", "country", "exp_group", "gender", "os", "source",
    "topic", "hour", "dow",
]
NUM_COLS = ["age", "text_len", "post_ctr_smooth"]
FEATURE_ORDER = ["user_id", "post_id"] + NUM_COLS + CAT_COLS
CAT_FOR_POOL = CAT_COLS + ["user_id", "post_id"]
CAT_IDX = [FEATURE_ORDER.index(c) for c in CAT_FOR_POOL]


def read_inputs(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feed = pd.read_csv(os.path.join(data_dir, "feed_sample_lastNdays.csv"))
    users = pd.read_csv(os.path.join(data_dir, "user_features.csv"))
    posts = pd.read_csv(os.path.join(data_dir, "post_text_sample.csv"))

    if "post_id" not in posts.columns and "id" in posts.columns:
        posts = posts.rename(columns={"id": "post_id"})
    if "post_id" not in feed.columns and "post" in feed.columns:
        feed = feed.rename(columns={"post": "post_id"})
    if "post_id" not in feed.columns and "id" in feed.columns:
        feed = feed.rename(columns={"id": "post_id"})

    need = {"user_id", "post_id", "timestamp", "action", "target"}
    miss = need - set(feed.columns)
    if miss:
        raise RuntimeError(f"feed_sample_lastNdays.csv missing columns: {sorted(miss)}")

    feed["timestamp"] = pd.to_datetime(feed["timestamp"])
    feed["target"] = pd.to_numeric(feed["target"], errors="coerce").fillna(0).astype(int)
    return feed, users, posts


def make_posts_static(posts: pd.DataFrame) -> pd.DataFrame:
    p = posts.copy()
    if "topic" not in p.columns:
        p["topic"] = "UNK"
    if "text" in p.columns:
        p["text_len"] = p["text"].fillna("").astype(str).str.len().astype(int)
    elif "text_len" not in p.columns:
        p["text_len"] = 0

    out = p[["post_id", "topic", "text_len"]].copy()
    out["post_id"] = out["post_id"].astype(int)
    out["topic"] = out["topic"].astype(str).fillna("UNK")
    out["text_len"] = pd.to_numeric(out["text_len"], errors="coerce").fillna(0).astype(float)
    return out


def ctr_from_train(train_views: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, float]:
    v = train_views.loc[train_views["action"] == "view", ["post_id", "target"]].copy()
    v["post_id"] = v["post_id"].astype(int)
    v["target"] = pd.to_numeric(v["target"], errors="coerce").fillna(0).astype(int)

    agg = v.groupby("post_id", as_index=False).agg(
        views=("target", "size"),
        likes=("target", "sum"),
    )
    total_views = float(agg["views"].sum())
    total_likes = float(agg["likes"].sum())
    global_ctr = (total_likes / total_views) if total_views > 0 else 0.0

    agg["post_ctr_smooth"] = (agg["likes"] + alpha * global_ctr) / (agg["views"] + alpha)
    return agg[["post_id", "post_ctr_smooth"]], float(global_ctr)


def build_Xy(
    view_rows: pd.DataFrame,
    users: pd.DataFrame,
    posts_static: pd.DataFrame,
    ctr_table: pd.DataFrame,
    global_ctr: float,
) -> tuple[pd.DataFrame, pd.Series]:
    df = view_rows[["user_id", "post_id", "timestamp", "target"]].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["post_id"] = df["post_id"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)

    # user feats
    user_cols = ["user_id", "age", "city", "country", "exp_group", "gender", "os", "source"]
    u = users[user_cols].drop_duplicates("user_id").copy()
    u["user_id"] = u["user_id"].astype(int)
    df = df.merge(u, on="user_id", how="left")

    # post static
    df = df.merge(posts_static, on="post_id", how="left")

    # CTR from train only (no leakage)
    ctr = ctr_table.copy()
    ctr["post_id"] = ctr["post_id"].astype(int)
    df = df.merge(ctr, on="post_id", how="left")
    df["post_ctr_smooth"] = pd.to_numeric(df["post_ctr_smooth"], errors="coerce").fillna(global_ctr).astype(float)

    # time ctx
    df["hour"] = df["timestamp"].dt.hour.astype(int).astype(str)
    df["dow"] = df["timestamp"].dt.dayofweek.astype(int).astype(str)

    # cleanup
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(float)
    df["text_len"] = pd.to_numeric(df["text_len"], errors="coerce").fillna(0).astype(float)
    df["topic"] = df["topic"].astype(str).fillna("UNK")
    for c in ["city", "country", "exp_group", "gender", "os", "source"]:
        df[c] = df[c].astype(str).fillna("UNK")

    # ids as cat strings
    df["user_id"] = df["user_id"].astype(str)
    df["post_id"] = df["post_id"].astype(str)

    X = df[FEATURE_ORDER].copy()
    y = df["target"].astype(int)
    return X, y


def user_level_hitrate_at_k(
    model: CatBoostClassifier,
    val_views: pd.DataFrame,
    users: pd.DataFrame,
    posts_static: pd.DataFrame,
    ctr_table: pd.DataFrame,
    global_ctr: float,
    k: int,
    max_cand_per_user: int,
    max_users: int,
    seed: int,
) -> tuple[float, int, float]:
    """
    User-level HitRate@k on validation:
      - candidates for user u = posts user actually viewed in validation
      - positives = posts user liked in validation (target=1 on views)
      - score all candidates and take top-k
      - hit(u)=1 if any liked post is in top-k

    Returns: (hitrate, used_users, avg_candidates)
    """
    rng = np.random.default_rng(seed)

    v = val_views.copy()
    v = v.loc[v["action"] == "view", ["user_id", "post_id", "timestamp", "target"]].copy()
    v["user_id"] = v["user_id"].astype(int)
    v["post_id"] = v["post_id"].astype(int)
    v["timestamp"] = pd.to_datetime(v["timestamp"])
    v["target"] = pd.to_numeric(v["target"], errors="coerce").fillna(0).astype(int)

    users_in_val = v["user_id"].unique()
    if len(users_in_val) > max_users:
        users_in_val = rng.choice(users_in_val, size=max_users, replace=False)

    hits = 0
    used = 0
    cand_sizes = []

    for u in users_in_val:
        u_df = v[v["user_id"] == u].copy()
        liked = set(u_df.loc[u_df["target"] == 1, "post_id"].tolist())
        if not liked:
            continue

        # candidates = viewed posts in val
        cand_posts = u_df["post_id"].unique()

        # subsample candidates for speed
        if len(cand_posts) > max_cand_per_user:
            cand_posts = rng.choice(cand_posts, size=max_cand_per_user, replace=False)

        # choose one realistic timestamp for context (first val view)
        t = u_df["timestamp"].iloc[0]

        req = pd.DataFrame({
            "user_id": [u] * len(cand_posts),
            "post_id": cand_posts,
            "timestamp": [t] * len(cand_posts),
            "target": [0] * len(cand_posts),
        })

        Xc, _ = build_Xy(req, users, posts_static, ctr_table, global_ctr)
        scores = model.predict_proba(Pool(Xc, cat_features=CAT_IDX))[:, 1]

        top_idx = np.argsort(-scores)[:k]
        top_posts = set(cand_posts[top_idx].tolist())

        if top_posts.intersection(liked):
            hits += 1
        used += 1
        cand_sizes.append(len(cand_posts))

    hr = hits / max(1, used)
    avg_cands = float(np.mean(cand_sizes)) if cand_sizes else 0.0
    return float(hr), int(used), float(avg_cands)


def main() -> None:
    print(f"[i] data_dir={DATA_DIR}")
    feed, users, posts = read_inputs(DATA_DIR)

    # views only
    views_all = feed.loc[feed["action"] == "view"].copy()
    views_all = views_all.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(views_all) * (1.0 - VAL_FRAC))
    train_views = views_all.iloc[:split_idx].copy()
    val_views = views_all.iloc[split_idx:].copy()

    print(f"[i] events: train={len(train_views)}, val={len(val_views)}")

    posts_static = make_posts_static(posts)

    # leakage-free CTR from train only
    ctr_tbl, global_ctr = ctr_from_train(train_views, alpha=ALPHA)
    print(f"[i] global_ctr_train={global_ctr:.6f} (alpha={ALPHA})")

    # train
    X_train, y_train = build_Xy(train_views, users, posts_static, ctr_tbl, global_ctr)
    print(f"[i] X_train shape: {X_train.shape}, positives={int(y_train.sum())}")

    model = CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=0.15,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(Pool(X_train, y_train, cat_features=CAT_IDX))

    # classification metric on validation (AUC)
    X_val, y_val = build_Xy(val_views, users, posts_static, ctr_tbl, global_ctr)
    proba = model.predict_proba(Pool(X_val, cat_features=CAT_IDX))[:, 1]
    auc = roc_auc_score(y_val, proba)

    # ranking metric on validation (user-level, realistic candidates)
    hr, used_users, avg_cands = user_level_hitrate_at_k(
        model=model,
        val_views=val_views,
        users=users,
        posts_static=posts_static,
        ctr_table=ctr_tbl,
        global_ctr=global_ctr,
        k=K,
        max_cand_per_user=MAX_CAND_PER_USER,
        max_users=MAX_USERS_EVAL,
        seed=SEED,
    )

    print("\n=== OFFLINE EVALUATION (leakage-free) ===")
    print(f"AUC (val): {auc:.4f}")
    print(f"User-level HitRate@{K} (candidates = user's viewed posts in val): {hr:.4f}")
    print(f"Users evaluated (with >=1 like in val): {used_users}")
    print(f"Avg candidates per user: {avg_cands:.1f} (cap={MAX_CAND_PER_USER})")
    print("========================================\n")


if __name__ == "__main__":
    main()