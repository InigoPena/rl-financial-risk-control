import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

BASE_URL = "https://finnhub.io/api/v1"
TOKEN = "d3m03tpr01qkjssdop9gd3m03tpr01qkjssdopa0"  # pon tu API key en esta variable de entorno

START_DATE = "2000-11-08"
END_DATE   = "2025-11-14"

# -----------------------
# Helpers
# -----------------------
def to_unix(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def finnhub_get(path: str, params: dict, retries: int = 5):
    """GET con retry simple (429 / fallos de red)."""
    if not TOKEN:
        raise RuntimeError("Falta FINNHUB_API_KEY en tu entorno.")

    params = dict(params)
    params["token"] = TOKEN

    for i in range(retries):
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            # Rate limit
            time.sleep(1.5 + i)
            continue
        # Otros errores
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise RuntimeError(f"Error {r.status_code} en {path}: {msg}")

    raise RuntimeError(f"Demasiados 429 (rate limit) en {path}")

def symbol_search(query: str) -> pd.DataFrame:
    data = finnhub_get("/search", {"q": query})
    # data = {"count":..., "result":[{symbol, description, ...}, ...]}
    return pd.DataFrame(data.get("result", []))

# -----------------------
# 1) Encuentra el símbolo (DXY o proxy)
# -----------------------
candidates = symbol_search("DXY")
print("Candidates Shape:", candidates.shape)

SYMBOL = "OANDA:EUR_USD"
if not candidates.empty and "symbol" in candidates.columns:
    SYMBOL = candidates.iloc[0]["symbol"]

print(f"Usando símbolo: {SYMBOL} (Fallback a EUR_USD si DXY falla)")

# -----------------------
# 2) Descarga candles (diario) en chunks
# -----------------------
start_ts = to_unix(START_DATE)
end_ts   = to_unix(END_DATE)

# Para evitar límites, troceamos en bloques de ~5 años
chunk_days = 365 * 5
chunk_seconds = chunk_days * 24 * 3600

all_frames = []

cur = start_ts
while cur < end_ts:
    cur_end = min(cur + chunk_seconds, end_ts)

    endpoint = "/stock/candle"
    if "OANDA" in SYMBOL or ":" in SYMBOL:
        endpoint = "/forex/candle"

    payload = finnhub_get(
        endpoint,
        {
            "symbol": SYMBOL,
            "resolution": "D",
            "from": cur,
            "to": cur_end,
        },
    )

    # payload esperado:
    # {"c":[...], "h":[...], "l":[...], "o":[...], "s":"ok", "t":[...], "v":[...]}
    if payload.get("s") != "ok":
        # a veces Finnhub devuelve {"s":"no_data"}
        print(f"Sin datos en rango {from_unix(cur).date()} -> {from_unix(cur_end).date()}")
        cur = cur_end + 1
        continue

    df = pd.DataFrame({
        "Date": pd.to_datetime(payload["t"], unit="s", utc=True).date,
        "Open": payload["o"],
        "High": payload["h"],
        "Low":  payload["l"],
        "Close": payload["c"],
        "Volume": payload.get("v", [None]*len(payload["t"])),
    })

    all_frames.append(df)

    # Pequeña pausa para no ir al límite
    time.sleep(0.25)
    cur = cur_end + 1

usd = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["Date"]).sort_values("Date")

# Filtra exactamente al rango que quieres (por seguridad)
usd["Date"] = pd.to_datetime(usd["Date"])
usd = usd[(usd["Date"] >= START_DATE) & (usd["Date"] <= END_DATE)]

# -----------------------
# 3) Guarda CSV con el formato de columnas
# -----------------------
out_path = "usd_finnhub_2000_2025.csv"
usd.to_csv(out_path, index=False)
print(f"Guardado: {out_path} | filas: {len(usd)} | rango: {usd['Date'].min().date()} -> {usd['Date'].max().date()}")
