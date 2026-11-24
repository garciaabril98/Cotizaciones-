"""
Descarga de cotizaciones (Yahoo Finance) para el sistema de drawdowns sectoriales.

Modos:
  python descargar.py historico   -> carga histórica por industria desde 2005 (reanudable)
  python descargar.py diario      -> últimas sesiones de todo el universo

Salidas (carpeta data/):
  historico/<industria>.csv.gz   fecha, ticker, cierre_aj, volumen
  historico/_mercado.csv.gz      benchmark, VIX, petróleo, tipos, divisas...
  diario/AAAA-MM-DD.csv.gz       últimas sesiones (se solapan; se deduplican al analizar)
  fallidos.csv                   tickers sin datos (para corregir el mapeo)
  estado.json                    resumen de la última ejecución
"""
import json
import re
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

RAIZ = Path(__file__).parent
DATOS = RAIZ / "data"
HIST = DATOS / "historico"
DIARIO = DATOS / "diario"
INICIO = "2005-01-01"
LOTE = 40                 # tickers por petición
PAUSA = 3                 # segundos entre lotes (evita bloqueos de Yahoo)
PRESUPUESTO_MIN = 150     # minutos máximos por ejecución histórica

# Series de mercado: benchmark, riesgo, materias primas, tipos y divisas
MERCADO = [
    "URTH", "IWDA.AS", "EUNL.DE",            # MSCI World (USD / EUR)
    "^GSPC", "^STOXX", "^VIX", "^VSTOXX",    # índices y volatilidad
    "BZ=F", "CL=F", "NG=F", "GC=F", "HG=F",  # Brent, WTI, gas, oro, cobre
    "^TNX", "^IRX", "^FVX",                  # tipos EE. UU.
    "EURUSD=X", "EURGBP=X", "EURCHF=X", "EURSEK=X", "EURNOK=X", "EURDKK=X",
    "DX-Y.NYB", "BDRY",                      # dólar, proxy Baltic Dry
]

COLUMNAS = ["fecha", "ticker", "cierre_aj", "volumen"]


def slug(texto: str) -> str:
    texto = texto.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", texto).strip("-")


def a_formato_largo(datos: pd.DataFrame, lote: list[str]) -> list[pd.DataFrame]:
    """Convierte la salida de yf.download (columnas ticker/campo) a filas."""
    filas = []
    if datos is None or datos.empty:
        return filas
    if not isinstance(datos.columns, pd.MultiIndex):
        datos = pd.concat({lote[0]: datos}, axis=1)
    disponibles = set(datos.columns.get_level_values(0))
    for t in lote:
        if t not in disponibles:
            continue
        sub = datos[t]
        if "Close" not in sub.columns:
            continue
        sub = sub[["Close"] + (["Volume"] if "Volume" in sub.columns else [])]
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        sub = sub.reset_index()
        sub.columns = ["fecha", "cierre_aj"] + (["volumen"] if sub.shape[1] == 3 else [])
        if "volumen" not in sub.columns:
            sub["volumen"] = pd.NA
        sub["ticker"] = t
        filas.append(sub[COLUMNAS])
    return filas


def descargar(tickers: list[str], **kwargs) -> tuple[pd.DataFrame, list[str]]:
    trozos, sin_datos = [], []
    for i in range(0, len(tickers), LOTE):
        lote = tickers[i:i + LOTE]
        datos = None
        for intento in range(4):
            try:
                datos = yf.download(
                    lote, auto_adjust=True, progress=False, threads=True,
                    group_by="ticker", **kwargs,
                )
                break
            except Exception as e:  # bloqueos o errores de red: espera y reintenta
                print(f"  aviso lote {i}: {e} (reintento {intento + 1})")
                time.sleep(30 * (intento + 1))
        filas = a_formato_largo(datos, lote)
        con_datos = {f["ticker"].iat[0] for f in filas}
        sin_datos += [t for t in lote if t not in con_datos]
        trozos += filas
        time.sleep(PAUSA)
    if not trozos:
        return pd.DataFrame(columns=COLUMNAS), sin_datos
    df = pd.concat(trozos, ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    df["cierre_aj"] = df["cierre_aj"].astype(float).round(6)
    return df, sin_datos


def guardar_fallidos(nuevos: list[str], resueltos: set[str]) -> int:
    ruta = DATOS / "fallidos.csv"
    previos = set(pd.read_csv(ruta)["ticker"]) if ruta.exists() else set()
    total = sorted((previos | set(nuevos)) - resueltos)
    pd.DataFrame({"ticker": total}).to_csv(ruta, index=False)
    return len(total)


def guardar_estado(info: dict) -> None:
    ruta = DATOS / "estado.json"
    estado = json.loads(ruta.read_text()) if ruta.exists() else {}
    estado.update(info)
    ruta.write_text(json.dumps(estado, indent=2, ensure_ascii=False))


def modo_historico(universo: pd.DataFrame) -> None:
    HIST.mkdir(parents=True, exist_ok=True)
    total = int(universo["industria"].nunique())
    ya = [r for r in HIST.glob("*.csv.gz") if not r.name.startswith("_")]
    if len(ya) >= total and (HIST / "_mercado.csv.gz").exists():
        print("Histórico ya completo: nada que hacer.")
        return
    t0 = time.time()
    pendientes = []

    ruta_mercado = HIST / "_mercado.csv.gz"
    if not ruta_mercado.exists():
        df, _ = descargar(MERCADO, start=INICIO)
        if not df.empty:
            df.to_csv(ruta_mercado, index=False, compression="gzip")
        print(f"_mercado: {df['ticker'].nunique()} series")

    fallidos_total, ok_total = [], set()
    bloqueos = 0
    for industria, grupo in universo.groupby("industria"):
        ruta = HIST / f"{slug(industria)}.csv.gz"
        if ruta.exists():
            continue
        if (time.time() - t0) / 60 > PRESUPUESTO_MIN:
            pendientes.append(industria)
            continue
        df, sin_datos = descargar(grupo["yahoo"].tolist(), start=INICIO)
        if df["ticker"].nunique() < 0.2 * len(grupo):
            # Probable bloqueo de Yahoo: no se marca como hecha y se reintenta luego
            print(f"{industria}: cobertura insuficiente, se reintentará")
            pendientes.append(industria)
            bloqueos += 1
            if bloqueos >= 3:
                print("Tres industrias seguidas sin datos: se detiene la ejecución.")
                break
            continue
        bloqueos = 0
        df.to_csv(ruta, index=False, compression="gzip")
        fallidos_total += sin_datos
        ok_total |= set(df["ticker"].unique())
        print(f"{industria}: {df['ticker'].nunique()}/{len(grupo)} con datos")

    hechas = len([r for r in HIST.glob("*.csv.gz") if not r.name.startswith("_")])
    n_fallidos = guardar_fallidos(fallidos_total, ok_total)
    guardar_estado({
        "historico_actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "industrias_completadas": hechas,
        "industrias_totales": int(universo["industria"].nunique()),
        "historico_completo": hechas >= int(universo["industria"].nunique()),
        "tickers_fallidos": n_fallidos,
    })
    print("Histórico completo." if hechas >= total else f"Completadas {hechas}/{total} industrias.")


def modo_diario(universo: pd.DataFrame) -> None:
    DIARIO.mkdir(parents=True, exist_ok=True)
    tickers = universo["yahoo"].tolist()
    ruta_f = DATOS / "fallidos.csv"
    if ruta_f.exists():  # no insistir con los que nunca han devuelto datos
        fallidos = set(pd.read_csv(ruta_f)["ticker"])
        tickers = [t for t in tickers if t not in fallidos]

    df, sin_datos = descargar(tickers + MERCADO, period="7d")
    hoy = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if df.empty:
        guardar_estado({"diario_error": f"{hoy}: sin datos (posible bloqueo de Yahoo)"})
        sys.exit("Sin datos en la descarga diaria.")
    df.to_csv(DIARIO / f"{hoy}.csv.gz", index=False, compression="gzip")
    guardar_estado({
        "diario_actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ultima_sesion": df["fecha"].max() if not df.empty else None,
        "diario_tickers_con_datos": int(df["ticker"].nunique()),
        "diario_tickers_sin_datos": len(sin_datos),
    })
    print(f"Diario {hoy}: {df['ticker'].nunique()} tickers, última sesión {df['fecha'].max() if not df.empty else '-'}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "diario"
    universo = pd.read_csv(RAIZ / "tickers.csv")
    DATOS.mkdir(exist_ok=True)
    if modo == "historico":
        modo_historico(universo)
    elif modo == "diario":
        modo_diario(universo)
    else:
        sys.exit("Modo no reconocido: usar 'historico' o 'diario'")
