from __future__ import annotations

# ===== Imports =====
import os
import pathlib
import pickle
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text
from catboost import CatBoostClassifier, Pool


# =====================
# Paths & Config
# =====================
HERE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path().resolve()

DB_URL = (
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

# Горизонт и лимиты выгрузки
TRAIN_DAYS = 30
SAMPLE_FEED_ROWS = 600_000
STEP_DAYS = 1
BATCH_LIMIT = 100_000

# Файлы (рядом со скриптом)
FEED_CSV  = HERE / "feed_sample_lastNdays.csv"
USER_CSV  = HERE / "user_data_sample.csv"      # <- хотим все user_id
POSTS_CSV = HERE / "post_text_sample.csv"
MODEL_CBM = HERE / "model.cbm"
META_PKL  = HERE / "meta.pkl"


# =====================
# DB engine + helpers
# =====================
engine = create_engine(DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})

def sql_df(query: str, **params) -> pd.DataFrame:
    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = 60000"))
        return pd.read_sql(text(query), conn, params=params)

# >>> CHANGED: добавили batch_load_sql для чтения полной user_data чанками
def batch_load_sql(query: str, chunksize: int = 200_000) -> pd.DataFrame:
    eng = create_engine(DB_URL, pool_pre_ping=True)
    conn = eng.connect().execution_options(stream_results=True)
    parts = []
    try:
        for part in pd.read_sql(query, conn, chunksize=chunksize):
            parts.append(part)
    finally:
        conn.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def stream_feed_window_to_csv(
    since, ts_max, out_path: str,
    step_days: int = STEP_DAYS,
    row_cap: int = SAMPLE_FEED_ROWS
) -> int:
    written = 0
    header_written = False
    cur_start = since

    p = pathlib.Path(out_path)
    if p.exists():
        p.unlink()

    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = 600000"))

        while cur_start < ts_max and written < row_cap:
            cur_end = min(cur_start + pd.Timedelta(days=step_days), ts_max)

            last_ts = None
            last_post_id = None

            while written < row_cap:
                if last_ts is None:
                    sql = text("""
                        SELECT user_id, post_id, timestamp, action, target
                        FROM feed_data
                        WHERE timestamp >= :a AND timestamp < :b
                        ORDER BY timestamp, post_id
                        LIMIT :lim
                    """)
                    params = {"a": cur_start, "b": cur_end, "lim": BATCH_LIMIT}
                else:
                    sql = text("""
                        SELECT user_id, post_id, timestamp, action, target
                        FROM feed_data
                        WHERE timestamp >= :a AND timestamp < :b
                          AND (timestamp, post_id) > (:ts, :pid)
                        ORDER BY timestamp, post_id
                        LIMIT :lim
                    """)
                    params = {"a": cur_start, "b": cur_end, "ts": last_ts, "pid": last_post_id, "lim": BATCH_LIMIT}

                df = pd.read_sql(sql, conn, params=params)
                if df.empty:
                    break

                df.to_csv(out_path, mode="a", index=False, header=(not header_written))
                header_written = True
                written += len(df)
                if written >= row_cap:
                    break

                last_row = df.iloc[-1]
                last_ts = last_row["timestamp"]
                last_post_id = int(last_row["post_id"])

            cur_start = cur_end

    return written

def fetch_by_ids(table: str, id_col: str, ids: List[int], chunk_size: int = 50_000) -> pd.DataFrame:
    out: List[pd.DataFrame] = []
    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = 600000"))
        n = len(ids)
        if n == 0:
            return pd.DataFrame()
        for i in range(0, n, chunk_size):
            chunk = ids[i:i+chunk_size]
            df = pd.read_sql(
                text(f"SELECT * FROM {table} WHERE {id_col} = ANY(:ids)"),
                conn,
                params={"ids": chunk},
            )
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# =====================
# Feature engineering
# =====================
def build_features(feed: pd.DataFrame, users: pd.DataFrame, posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list]:
    # Берём только показы
    df = feed.loc[feed["action"] == "view", ["user_id", "post_id", "timestamp", "target"]].copy()
    df["target"] = df["target"].fillna(0).astype(int)

    # user-фичи
    user_cols = ["user_id", "age", "city", "country", "exp_group", "gender", "os", "source"]
    users_small = users[user_cols].drop_duplicates("user_id")
    df = df.merge(users_small, on="user_id", how="left")

    # post-фичи
    posts_small = posts[["post_id", "topic", "text"]].copy()
    posts_small["text_len"] = posts_small["text"].fillna("").str.len().astype(int)
    posts_small = posts_small.drop(columns=["text"])
    df = df.merge(posts_small, on="post_id", how="left")

    tmp = feed.assign(is_like=(feed["action"] == "like").astype(int))
    post_agg = tmp.groupby("post_id", as_index=False).agg(
        views=("action", "count"),
        likes=("is_like", "sum")
    )
    global_ctr = post_agg["likes"].sum() / max(1, post_agg["views"].sum())
    alpha = 1000.0
    post_agg["post_ctr_smooth"] = (post_agg["likes"] + alpha * global_ctr) / (post_agg["views"] + alpha)
    df = df.merge(post_agg[["post_id", "post_ctr_smooth"]], on="post_id", how="left")

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df = df.drop(columns=["timestamp"])

    cat_cols = ["city", "country", "exp_group", "gender", "os", "source", "topic", "hour", "dow"]
    num_cols = ["age", "text_len", "post_ctr_smooth"]

    feature_cols = ["user_id", "post_id"] + num_cols + cat_cols
    X = df[feature_cols].copy()
    y = df["target"]

    for c in cat_cols + ["user_id", "post_id"]:
        X[c] = X[c].astype("category").astype(str)

    cat_idx = [X.columns.get_loc(c) for c in (cat_cols + ["user_id", "post_id"])]
    return X, y, feature_cols, cat_idx


# =====================
# Train & Save (CatBoost)
# =====================
def train_and_save() -> pathlib.Path:
    print("[i] Получаю ts_max из feed_data…")
    ts_max = pd.to_datetime(sql_df("SELECT MAX(timestamp) AS ts_max FROM feed_data").loc[0, "ts_max"])
    since = ts_max - pd.Timedelta(days=TRAIN_DAYS)
    print(f"[i] Окно сырой ленты: {since} — {ts_max} (последние {TRAIN_DAYS} дней)")

    print("[i] Стримлю feed_data в CSV чанками…")
    rows = stream_feed_window_to_csv(since, ts_max, str(FEED_CSV), step_days=STEP_DAYS, row_cap=SAMPLE_FEED_ROWS)
    print(f"[i] streamed rows: {rows}")
    if rows <= 0:
        raise RuntimeError("Не удалось выгрузить feed_sample.")

    feed = pd.read_csv(FEED_CSV, parse_dates=["timestamp"]).sort_values("timestamp")
    feed.to_csv(FEED_CSV, index=False)
    print("[i] feed_sample shape:", feed.shape)

    # уникальные id из окна — пригодятся только для постов
    uids = feed["user_id"].dropna().astype(int).unique().tolist()
    pids = feed["post_id"].dropna().astype(int).unique().tolist()
    print(f"[i] unique users in feed window: {len(uids)}, unique posts: {len(pids)}")

    # >>> CHANGED: user_data берём ПОЛНОСТЬЮ, для ВСЕХ user_id
    print("[i] Выгружаю ПОЛНУЮ user_data (все пользователи)…")
    users = batch_load_sql("""
        SELECT user_id, age, city, country, exp_group, gender, os, source
        FROM user_data
    """)
    users = users.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    users.to_csv(USER_CSV, index=False)

    # посты тянем только нужные (это не мешает тому, что user_data полная)
    print("[i] Выгружаю post_text_df по нужным post_id…")
    posts = fetch_by_ids("post_text_df", "post_id", pids, chunk_size=50_000)
    posts.to_csv(POSTS_CSV, index=False)

    print("[i] Собираю признаки…")
    X, y, feature_cols, cat_idx = build_features(feed, users, posts)
    print("[i] X shape:", X.shape, "y:", y.shape)

    print("[i] Обучаю CatBoostClassifier…")
    train_pool = Pool(X, y, cat_features=cat_idx)
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=False
    )
    model.fit(train_pool)

    model.save_model(str(MODEL_CBM), format="cbm")

    meta = {"feature_cols": feature_cols, "cat_idx": cat_idx}
    with open(META_PKL, "wb") as f:
        pickle.dump(meta, f)

    print("[✓] Saved:", MODEL_CBM)
    return MODEL_CBM


# =====================
# LOADER for LMS (строго по шаблону)
# =====================
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = "/workdir/user_input/model"
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    local_path = str(MODEL_CBM.resolve())
    model_path = get_model_path(local_path)

    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


# =====================
# entrypoint для локального обучения
# =====================
if __name__ == "__main__":
    train_and_save()