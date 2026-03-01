from __future__ import annotations

# ===== Imports =====
import os
from typing import List, Optional
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException, Query

# --- fallback схема PostGet (в чекере импортируется из schema) ---
try:
    from schema import PostGet
except Exception:
    from pydantic import BaseModel, ConfigDict

    class PostGet(BaseModel):
        id: int
        text: str
        topic: str
        model_config = ConfigDict(from_attributes=True)

# =====================
# CONFIG
# =====================
USER_TABLE = "vladislav_andreev_user_features_lesson_22"
POST_TABLE = "vladislav_andreev_post_features_lesson_22"

# =====================
# LOCAL/DB switch
# =====================
# 1 = load from Postgres (as in checker), 0 = load from local CSV samples
USE_DB = (os.environ.get("IS_LMS") == "1") or (os.getenv("USE_DB", "0") == "1")
DATA_DIR = os.getenv("DATA_DIR", "data/sample")

# DB config (ONLY for DB mode)
DB_URL = os.getenv("DATABASE_URL")
engine: Optional[object] = None
if USE_DB:
    if not DB_URL:
        raise RuntimeError("DATABASE_URL env var is required in DB mode (USE_DB=1 or IS_LMS=1).")
    engine = create_engine(DB_URL, pool_pre_ping=True)

# Порядок и типы признаков — как в обучении
CAT_COLS = [
    "city", "country", "exp_group", "gender", "os", "source",
    "topic", "hour", "dow",
]
NUM_COLS = ["age", "text_len", "post_ctr_smooth"]
FEATURE_ORDER = ["user_id", "post_id"] + NUM_COLS + CAT_COLS
CAT_FOR_POOL = CAT_COLS + ["user_id", "post_id"]
CAT_IDX = [FEATURE_ORDER.index(c) for c in CAT_FOR_POOL]

K_RECS = 5                 # ровно 5 по ТЗ
TOP_CANDIDATES = 2000      # пул кандидатов (баланс скорость/качество)

# =====================
# Model loader (per spec)
# =====================
def get_model_path(_: str) -> str:
    """
    In LMS checker model file is available at /workdir/user_input/model.
    Locally we expect it at artifacts/model.cbm.
    You can override via MODEL_PATH env var.
    """
    if os.environ.get("IS_LMS") == "1":
        return "/workdir/user_input/model"
    return os.getenv("MODEL_PATH", "artifacts/model.cbm")


def load_models() -> CatBoostClassifier:
    model_path = get_model_path("")
    model = CatBoostClassifier()
    model.load_model(model_path)
    # немного ускорим инференс
    try:
        model.set_params(thread_count=4)
    except Exception:
        pass
    return model

# =====================
# Helpers (DB & loading)
# =====================
def batch_load_sql(query: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Читаем большие таблицы из БД чанками, чтобы не взорвать память."""
    if not USE_DB or engine is None:
        raise RuntimeError("batch_load_sql called in local mode. Set USE_DB=1 and DATABASE_URL.")
    conn = engine.connect().execution_options(stream_results=True)
    parts = []
    try:
        for part in pd.read_sql(query, conn, chunksize=chunksize):
            parts.append(part)
    finally:
        conn.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_posts_texts(post_ids: list[int]) -> pd.DataFrame:
    """Достаём text/topic из исходной post_text_df для нужных post_id (батчами)."""
    if not post_ids:
        return pd.DataFrame(columns=["post_id", "text", "topic"])
    if engine is None:
        raise RuntimeError("fetch_posts_texts requires DB mode engine.")

    out = []
    step = 50_000
    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = 120000"))
        for i in range(0, len(post_ids), step):
            chunk = post_ids[i: i + step]
            df = pd.read_sql(
                text("SELECT post_id, text, topic FROM post_text_df WHERE post_id = ANY(:ids)"),
                conn,
                params={"ids": chunk},
            )
            out.append(df)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["post_id", "text", "topic"])

# =====================
# Loaders (DB or local CSV)
# =====================
def load_user_feats() -> pd.DataFrame:
    """
    В local-mode читаем уже подготовленную витрину user_features.csv
    (чтобы local inference = DB inference по источнику фич).
    """
    if USE_DB:
        df = batch_load_sql(f"""
            SELECT user_id, age, city, country, exp_group, gender, os, source
            FROM {USER_TABLE}
        """)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, "user_features.csv"))

    df["user_id"] = df["user_id"].astype(int)
    return df


def load_post_feats() -> pd.DataFrame:
    """
    В local-mode читаем уже подготовленную витрину post_features.csv
    и НЕ считаем CTR внутри сервиса.
    """
    if USE_DB:
        df = batch_load_sql(f"""
            SELECT post_id, topic, text_len, post_ctr_smooth
            FROM {POST_TABLE}
        """)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, "post_features.csv"))

    df["post_id"] = df["post_id"].astype(int)
    df["topic"] = df["topic"].astype(str).fillna("UNK")
    df["text_len"] = pd.to_numeric(df["text_len"], errors="coerce").fillna(0).astype(float)
    df["post_ctr_smooth"] = pd.to_numeric(df["post_ctr_smooth"], errors="coerce").fillna(0).astype(float)
    return df


def load_post_texts(post_ids: list[int]) -> pd.DataFrame:
    if USE_DB:
        return fetch_posts_texts(post_ids)

    df = pd.read_csv(os.path.join(DATA_DIR, "post_text_sample.csv"))

    if "post_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "post_id"})

    if "post_id" not in df.columns:
        raise RuntimeError("post_text_sample.csv must contain 'post_id' (or 'id') column")

    if "text" not in df.columns:
        df["text"] = ""
    if "topic" not in df.columns:
        df["topic"] = "UNK"

    df["post_id"] = df["post_id"].astype(int)
    return df[["post_id", "text", "topic"]].copy()

# =====================
# Cold-start helpers
# =====================
def make_default_user_row(user_id: int) -> pd.DataFrame:
    """Строка юзера по умолчанию, если в таблице фичей его нет (мягкий cold-start)."""
    return pd.DataFrame([{
        "user_id": str(user_id),
        "age": 0.0,
        "city": "UNK",
        "country": "UNK",
        "exp_group": "UNK",
        "gender": "UNK",
        "os": "UNK",
        "source": "UNK",
    }])


def get_user_row(user_id: int, user_tbl: pd.DataFrame) -> pd.DataFrame:
    r = user_tbl[user_tbl["user_id"] == user_id]
    if r.empty:
        return make_default_user_row(user_id)
    r = r.iloc[[0]].copy()
    r["user_id"] = str(user_id)
    for c in ["city", "country", "exp_group", "gender", "os", "source"]:
        r[c] = r[c].astype(str).fillna("UNK")
    r["age"] = pd.to_numeric(r["age"], errors="coerce").fillna(0).astype(float)
    return r[["user_id", "age", "city", "country", "exp_group", "gender", "os", "source"]].reset_index(drop=True)

# =====================
# Global objects (load once at startup)
# =====================
app = FastAPI(title="Karpov Recsys Service")

# 1) модель
try:
    MODEL: CatBoostClassifier = load_models()
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# 2) фичи пользователей
try:
    USER_FEATS: pd.DataFrame = load_user_feats()
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить фичи пользователей: {e}")

# 3) фичи постов
try:
    POST_FEATS: pd.DataFrame = load_post_feats()
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить фичи постов: {e}")

# 4) тексты постов (для ответа) — делаем один раз
try:
    _all_post_ids = POST_FEATS["post_id"].tolist()
    POST_TEXTS = load_post_texts(_all_post_ids)
    POST_TEXTS["post_id"] = POST_TEXTS["post_id"].astype(int)
    POST_TEXTS_MAP = POST_TEXTS.set_index("post_id")[["text", "topic"]].to_dict(orient="index")
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить текст постов: {e}")

# 5) КЭШ кандидатов: предсортированный пул + приведение типов ОДИН раз
CAND_BASE = POST_FEATS.sort_values("post_ctr_smooth", ascending=False).head(TOP_CANDIDATES).copy()
CAND_BASE["post_id"] = CAND_BASE["post_id"].astype(str)
CAND_BASE["topic"] = CAND_BASE["topic"].astype(str).fillna("UNK")
CAND_BASE["text_len"] = pd.to_numeric(CAND_BASE["text_len"], errors="coerce").fillna(0).astype(float)
CAND_BASE["post_ctr_smooth"] = pd.to_numeric(CAND_BASE["post_ctr_smooth"], errors="coerce").fillna(0).astype(float)

# 6) Ручной кэш профилей пользователей (без functools)
USER_ROW_CACHE: dict[int, pd.DataFrame] = {}

def get_user_row_cached(user_id: int) -> pd.DataFrame:
    r = USER_ROW_CACHE.get(user_id)
    if r is not None:
        return r
    r = get_user_row(user_id, USER_FEATS)
    USER_ROW_CACHE[user_id] = r
    return r

# =====================
# Быстрая сборка матрицы предсказаний
# =====================
def build_predict_matrix(user_id: int, when: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Профиль пользователя (кэш) × кандидаты (кэш) × контекст (час/день) → X.
    """
    # профиль
    u = get_user_row_cached(user_id)  # 1 строка, уже с нужными типами

    # кандидаты — готовые
    cand = CAND_BASE.copy()

    # контекст времени (строки, как в train)
    cand["hour"] = str(when.hour)
    cand["dow"] = str(when.weekday())

    # дешёвое дублирование профиля N раз и склейка
    U = pd.concat([u] * len(cand), ignore_index=True)
    X = pd.concat([U.reset_index(drop=True), cand.reset_index(drop=True)], axis=1)

    # строгий порядок фич
    X = X[FEATURE_ORDER]

    # meta
    cand_meta = cand[["post_id", "topic"]].copy()
    cand_meta["post_id"] = cand_meta["post_id"].astype(int)
    return X, cand_meta

# =====================
# Endpoint
# =====================
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    id: int,
    time: datetime,
    limit: int = Query(10, ge=1, le=50),  # по ТЗ возвращаем 5, limit игнорируем
) -> List[PostGet]:
    """
    Возвращает РОВНО 5 постов (по заданию).
    """
    # 1) сборка X
    X, cand_meta = build_predict_matrix(user_id=id, when=time)

    # 2) скоринг
    pool = Pool(X, cat_features=CAT_IDX)
    try:
        proba = MODEL.predict_proba(pool)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # 3) top-k
    cand_meta = cand_meta.assign(p=proba)
    top = cand_meta.sort_values("p", ascending=False).head(K_RECS)

    # 4) ответ
    resp: List[PostGet] = []
    for row in top.itertuples(index=False):
        pid = int(row.post_id)
        tt = POST_TEXTS_MAP.get(pid)
        txt = tt["text"] if tt and "text" in tt else ""
        tpc = tt["topic"] if tt and "topic" in tt else str(getattr(row, "topic", "unknown"))
        resp.append(PostGet(id=pid, text=txt, topic=tpc))

    return resp