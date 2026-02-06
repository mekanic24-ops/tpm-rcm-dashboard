# streamlit_app.py

import zipfile
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import streamlit as st

# OpenAI (para Asistente de Confiabilidad)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="DESEMPEÑO OPERACIONAL DE LA FLOTA", layout="wide")

ZIP_NAME = "TPM_modelo_normalizado_CSV.zip"
DATA_DIR = Path("data_normalizada")

# =========================================================
# HELPERS
# =========================================================
# -------------------------
# IA: Asistente de Confiabilidad
# -------------------------
def get_openai_client():
    """Crea el cliente OpenAI usando Streamlit secrets."""
    if OpenAI is None:
        return None
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        return None
    return OpenAI(api_key=key)

def df_to_markdown(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 10) -> str:
    """Convierte un DataFrame a tabla Markdown (recortada) para contexto del LLM."""
    if df is None or df.empty:
        return "(sin filas)"
    d = df.copy()
    if d.shape[1] > max_cols:
        d = d.iloc[:, :max_cols]
    if d.shape[0] > max_rows:
        d = d.head(max_rows)
    return d.to_markdown(index=False)

def build_reliability_context(
    *,
    vista_disp: str,
    date_range,
    cult_choice: str,
    turn_choice: str,
    prop_sel: str,
    fam_sel: str,
    trc_sel: str,
    imp_sel: str,
    proc_name_sel: str,
    eq_sel: str,
    sis_sel: str,
    com_sel: str,
    par_sel: str,
    kpis: dict,
    df_lvl: pd.DataFrame,
    top_tables: dict,
) -> str:
    """Arma el contexto (texto) que se enviará al asistente."""
    filtros = {
        "vista_kpis": vista_disp,
        "fechas": str(date_range),
        "cultivo": cult_choice,
        "turno": turn_choice,
        "propietario": prop_sel,
        "familia": fam_sel,
        "tractor": trc_sel,
        "implemento": imp_sel,
        "proceso": proc_name_sel,
        "cascada_equipo": eq_sel,
        "cascada_subunidad": sis_sel,
        "cascada_componente": com_sel,
        "cascada_parte": par_sel,
    }

    prefer = [
        "EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO",
        "SUBUNIDAD", "SUB_UNIDAD", "COMPONENTE", "PARTE", "PIEZA",
        "DOWNTIME_HR", "VERBO_TECNICO", "CAUSA_FALLA",
    ]
    cols = [c for c in prefer if c in df_lvl.columns]
    df_view = df_lvl[cols].copy() if cols else df_lvl.copy()

    ctx = f"""Eres un **Asistente de Confiabilidad y Mantenimiento** (RCM/TPM). 
Responde SIEMPRE en español, con formato claro y accionable para taller/campo.

## 1) Filtros activos
{filtros}

## 2) KPIs calculados (ya filtrados)
{kpis}

## 3) Muestra de fallas (FALLAS_DETALLE) filtradas (primeras filas)
{df_to_markdown(df_view, max_rows=20, max_cols=12)}

## 4) Top 10 (ya calculados)
{top_tables}
"""
    return ctx
def ensure_data_unzipped():
    if DATA_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = Path(ZIP_NAME)
    if not zip_path.exists():
        st.error(f"No encontré el archivo {ZIP_NAME} en el repo. Súbelo junto al script.")
        st.stop()
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_", regex=False)
    )
    return df

def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.strip()

def infer_tipo_equipo_from_code(x: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s.startswith("TRC"):
        return "TRACTOR"
    return "IMPLEMENTO"

def _coerce_downtime_to_hr(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    if v.notna().any():
        p95 = np.nanpercentile(v.dropna(), 95)
        # si está en minutos (p95>240min), convierte a horas
        if pd.notna(p95) and p95 > 240:
            v = v / 60.0
    return v

@st.cache_data(show_spinner=False)
def load_tables() -> dict:
    ensure_data_unzipped()

    def r(name):
        p = DATA_DIR / name
        if not p.exists():
            st.error(f"Falta el archivo: {p}")
            st.stop()
        return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)

    turnos = r("TURNOS.csv")
    horometros = r("HOROMETROS_TURNO.csv")
    eventos = r("EVENTOS_TURNO.csv")
    operadores = r("OPERADORES.csv")
    lotes = r("LOTES.csv")
    fallas_cat = r("FALLAS_CATALOGO.csv")
    cat_proceso = r("CAT_PROCESO.csv")

    # Catálogo EQUIPOS (opcional, para filtros Propietario/Familia)
    equipos_cat = None
    for cand in [Path("EQUIPOS.csv"), DATA_DIR / "EQUIPOS.csv"]:
        if cand.exists():
            try:
                equipos_cat = pd.read_csv(cand, encoding="utf-8-sig", low_memory=False)
            except Exception:
                equipos_cat = pd.read_csv(cand, low_memory=False)
            break


    fallas_detalle = None
    p_det = DATA_DIR / "FALLAS_DETALLE_NORMALIZADO.csv"
    if p_det.exists():
        fallas_detalle = pd.read_csv(p_det, encoding="utf-8-sig", low_memory=False)

    # Tipos base
    turnos["FECHA"] = pd.to_datetime(turnos["FECHA"], errors="coerce")
    horometros["TO_HORO"] = pd.to_numeric(horometros["TO_HORO"], errors="coerce")
    eventos["DT_MIN"] = pd.to_numeric(eventos["DT_MIN"], errors="coerce")

    def norm_str(s: pd.Series) -> pd.Series:
        return safe_norm_str_series(s)

    for col in ["ID_TURNO", "ID_TRACTOR", "ID_IMPLEMENTO", "ID_LOTE", "ID_OPERADOR", "ID_PROCESO", "TURNO"]:
        if col in turnos.columns:
            turnos[col] = norm_str(turnos[col])

    for col in ["ID_TURNO", "ID_EQUIPO", "TIPO_EQUIPO"]:
        if col in horometros.columns:
            horometros[col] = norm_str(horometros[col])

    for col in ["ID_TURNO", "CATEGORIA_EVENTO", "ID_EQUIPO_AFECTADO", "ID_FALLA"]:
        if col in eventos.columns:
            eventos[col] = norm_str(eventos[col])

    if "ID_FALLA" in fallas_cat.columns:
        fallas_cat["ID_FALLA"] = norm_str(fallas_cat["ID_FALLA"])

    cat_proceso["ID_PROCESO"] = norm_str(cat_proceso["ID_PROCESO"])
    cat_proceso["NOMBRE_PROCESO"] = cat_proceso["NOMBRE_PROCESO"].astype(str).str.strip()

    operadores["ID_OPERADOR"] = norm_str(operadores["ID_OPERADOR"])
    operadores["NOMBRE_OPERADOR"] = operadores["NOMBRE_OPERADOR"].astype(str).str.strip()

    lotes["ID_LOTE"] = norm_str(lotes["ID_LOTE"])
    lotes["CULTIVO"] = lotes["CULTIVO"].astype(str)


    if equipos_cat is not None and not equipos_cat.empty:
        equipos_cat = norm_cols(equipos_cat)

        # Normalizar código
        if "EQUIPOS" in equipos_cat.columns:
            equipos_cat["EQUIPOS"] = norm_str(equipos_cat["EQUIPOS"])
        elif "EQUIPO" in equipos_cat.columns and "EQUIPOS" not in equipos_cat.columns:
            equipos_cat["EQUIPOS"] = norm_str(equipos_cat["EQUIPO"])

        # Normalizar propietario/familia
        if "PROPIETARIO2" in equipos_cat.columns:
            equipos_cat["PROPIETARIO2"] = equipos_cat["PROPIETARIO2"].astype(str).str.strip().str.upper()
        elif "PROPIETARIO" in equipos_cat.columns:
            equipos_cat["PROPIETARIO2"] = equipos_cat["PROPIETARIO"].astype(str).str.strip().str.upper()

        if "FAMILIA" in equipos_cat.columns:
            equipos_cat["FAMILIA"] = equipos_cat["FAMILIA"].astype(str).str.strip().str.upper()

        # Quitar basura típica
        for c in ["PROPIETARIO2", "FAMILIA"]:
            if c in equipos_cat.columns:
                equipos_cat[c] = equipos_cat[c].replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})


    # ==============================
    # FALLAS_DETALLE_NORMALIZADO
    # ==============================
    if fallas_detalle is not None:
        fallas_detalle = norm_cols(fallas_detalle)

        if "ID" in fallas_detalle.columns and "ID_TURNO" not in fallas_detalle.columns:
            fallas_detalle["ID_TURNO"] = norm_str(fallas_detalle["ID"])

        for c in ["ID_TURNO", "EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO", "TIPO_EQUIPO"]:
            if c in fallas_detalle.columns:
                fallas_detalle[c] = norm_str(fallas_detalle[c])

        if "DOWNTIME_HR" in fallas_detalle.columns:
            fallas_detalle["DOWNTIME_HR"] = _coerce_downtime_to_hr(fallas_detalle["DOWNTIME_HR"])
        else:
            tf = find_first_col(
                fallas_detalle,
                ["T_FALLA", "TFALLA", "TIEMPO_FALLA", "TIEMPO_DE_FALLA", "DOWNTIME", "DT_HR"]
            )
            if tf is not None:
                fallas_detalle["DOWNTIME_HR"] = _coerce_downtime_to_hr(fallas_detalle[tf])
            else:
                fallas_detalle["DOWNTIME_HR"] = np.nan

        alias_map = {
            "SUB_UNIDAD": "SUBUNIDAD",
            "PIEZA": "PARTE",
            "VERBO": "VERBO_TECNICO",
            "VERBO_TECN": "VERBO_TECNICO",
            "CAUSA": "CAUSA_FALLA",
            "CAUSA_RAIZ": "CAUSA_FALLA",
            "CAUSA_INMEDIATA": "CAUSA_FALLA",
        }
        for src, dst in alias_map.items():
            if src in fallas_detalle.columns and dst not in fallas_detalle.columns:
                fallas_detalle[dst] = fallas_detalle[src]

        eq_col = find_first_col(fallas_detalle, ["EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO"])
        if "TIPO_EQUIPO" not in fallas_detalle.columns or fallas_detalle["TIPO_EQUIPO"].isna().all():
            if eq_col is not None:
                fallas_detalle["TIPO_EQUIPO"] = fallas_detalle[eq_col].apply(infer_tipo_equipo_from_code)
        else:
            fallas_detalle["TIPO_EQUIPO"] = fallas_detalle["TIPO_EQUIPO"].astype(str).str.upper().str.strip()

    return {
        "turnos": turnos,
        "horometros": horometros,
        "eventos": eventos,
        "operadores": operadores,
        "lotes": lotes,
        "fallas_cat": fallas_cat,
        "cat_proceso": cat_proceso,
        "equipos_cat": equipos_cat,
        "fallas_detalle": fallas_detalle,
    }

def normalize_cultivo(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    s = s.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
    if s in ["PALTA", "PALTO"]:
        return "PALTO"
    if s in ["ARANDANO", "ARÁNDANO"]:
        return "ARANDANO"
    return s

def normalize_turno(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    s = s.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
    if s in ["D", "DIA", "DÍA", "DAY"]:
        return "DIA"
    if s in ["N", "NOCHE", "NIGHT"]:
        return "NOCHE"
    return s

def build_enriched_turnos(turnos, operadores, lotes, equipos_cat=None):
    t = turnos.copy()
    op_map = dict(zip(operadores["ID_OPERADOR"], operadores["NOMBRE_OPERADOR"]))
    t["OPERADOR_NOMBRE"] = t["ID_OPERADOR"].map(op_map)

    lote_map = dict(zip(lotes["ID_LOTE"], lotes["CULTIVO"]))
    t["CULTIVO"] = t["ID_LOTE"].map(lote_map)
    t["CULTIVO"] = t["CULTIVO"].apply(normalize_cultivo)

    if "TURNO" in t.columns:
        t["TURNO_NORM"] = t["TURNO"].apply(normalize_turno)
    else:
        t["TURNO_NORM"] = None


    # Enriquecer con Propietario/Familia por Tractor e Implemento (si existe EQUIPOS.csv)
    if equipos_cat is not None and not equipos_cat.empty and "EQUIPOS" in equipos_cat.columns:
        e = equipos_cat.copy()
        e = e.dropna(subset=["EQUIPOS"]).copy()
        e_key = e["EQUIPOS"].astype(str)
        prop_map = dict(zip(e_key, e["PROPIETARIO2"])) if "PROPIETARIO2" in e.columns else {}
        fam_map = dict(zip(e_key, e["FAMILIA"])) if "FAMILIA" in e.columns else {}

        # Tractor
        if "ID_TRACTOR" in t.columns:
            t["TRC_PROPIETARIO"] = t["ID_TRACTOR"].astype(str).map(prop_map)
            t["TRC_FAMILIA"] = t["ID_TRACTOR"].astype(str).map(fam_map)
        else:
            t["TRC_PROPIETARIO"] = np.nan
            t["TRC_FAMILIA"] = np.nan

        # Implemento
        if "ID_IMPLEMENTO" in t.columns:
            t["IMP_PROPIETARIO"] = t["ID_IMPLEMENTO"].astype(str).map(prop_map)
            t["IMP_FAMILIA"] = t["ID_IMPLEMENTO"].astype(str).map(fam_map)
        else:
            t["IMP_PROPIETARIO"] = np.nan
            t["IMP_FAMILIA"] = np.nan

    return t

def fmt_num(x, dec=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{dec}f}"

def fmt_pct(x, dec=1):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:,.{dec}f}%"

def kpi_card_html(title: str, value: str, hint: Optional[str] = None, color: Optional[str] = None) -> str:
    hint_text = hint if hint else "&nbsp;"
    border = f"border:2px solid {color};" if color else "border:1px solid #e5e7eb;"
    return f"""
      <div class="kpi" style="{border}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-hint">{hint_text}</div>
      </div>
    """

def render_kpi_row(cards_html: List[str], height=190, big=False):
    value_size = 48 if big else 40
    html = f"""
    <html>
      <head>
        <style>
          .kpi-row{{
            display:flex; gap:18px; justify-content:center; align-items:flex-start;
            flex-wrap:wrap; margin:6px 0 0 0;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          }}
          .kpi{{
            width:300px; min-height:140px; padding:10px 12px; text-align:center;
            box-sizing:border-box; border-radius:12px; background:#fff;
          }}
          .kpi-title{{ font-size:14px; font-weight:800; color:#111; margin:0 0 10px 0; line-height:1.2; }}
          .kpi-value{{ font-size:{value_size}px; font-weight:600; line-height:1.05; margin:0 0 8px 0; color:#111; }}
          .kpi-hint{{ font-size:12px; color:#6b7280; line-height:1.2; min-height:16px; }}
        </style>
      </head>
      <body style="margin:0;padding:0;">
        <div class="kpi-row">
          {''.join(cards_html)}
        </div>
      </body>
    </html>
    """
    components.html(html, height=height)

def center_title(txt: str) -> str:
    return f"<div style='text-align:center;font-weight:800;font-size:18px;margin:0 0 6px 0;'>{txt}</div>"

def disp_color(d):
    if d is None or pd.isna(d):
        return None
    if d < 0.90:
        return "#ef4444"
    if d < 0.95:
        return "#22c55e"
    return "#3b82f6"

def mttr_color_3(x):
    if x is None or pd.isna(x):
        return None
    if x < 1.2:
        return "#3b82f6"
    if x < 2.5:
        return "#22c55e"
    return "#ef4444"

def mtbf_color(x):
    if x is None or pd.isna(x):
        return None
    if x < 100:
        return "#ef4444"
    if x < 500:
        return "#22c55e"
    return "#3b82f6"

def pareto_chart(df: pd.DataFrame, dim_col: str, val_col: str, top_n: int, title: str):
    if dim_col not in df.columns:
        st.info(f"No existe columna: {dim_col}")
        return
    tmp = df.copy()
    tmp[dim_col] = tmp[dim_col].astype(str).replace({"nan": "(Vacío)"}).fillna("(Vacío)")
    g = tmp.groupby(dim_col, dropna=False)[val_col].sum().sort_values(ascending=False).reset_index()
    g = g.head(int(top_n)).copy()
    g["ACUM"] = g[val_col].cumsum()
    total = g[val_col].sum()
    g["ACUM_PCT"] = np.where(total > 0, g["ACUM"] / total, 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=g[dim_col], y=g[val_col], name=val_col))
    fig.add_trace(go.Scatter(x=g[dim_col], y=g["ACUM_PCT"], name="% Acum", yaxis="y2", mode="lines+markers"))

    fig.update_layout(
        title=title,
        title_x=0.5,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Down Time (h)"),
        yaxis2=dict(title="% Acum", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

# =========================================================
# LOAD
# =========================================================
tables = load_tables()
turnos = build_enriched_turnos(tables["turnos"], tables["operadores"], tables["lotes"], tables.get("equipos_cat"))
horometros = tables["horometros"]
eventos = tables["eventos"]
cat_proceso = tables["cat_proceso"]
fallas_detalle = tables["fallas_detalle"]

proc_map: Dict[str, str] = dict(zip(cat_proceso["ID_PROCESO"].astype(str), cat_proceso["NOMBRE_PROCESO"].astype(str)))

# =========================================================
# SIDEBAR: PAGE + FILTROS GLOBALES
# =========================================================
st.sidebar.header("Navegación")
page = st.sidebar.radio("Página", ["Dashboard", "Paretos", "Técnico"], index=0, key="page")

st.sidebar.divider()
st.sidebar.header("Filtros globales")

vista_disp = st.sidebar.radio(
    "Vista de KPIs",
    ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
    index=0,
    key="vista_disp"
)

min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
    key="date_range",
)

df_base = turnos.copy()
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1])
    df_base = df_base[(df_base["FECHA"] >= d1) & (df_base["FECHA"] <= d2)]

cult_label_to_val = {"(Todos)": None, "Palto": "PALTO", "Arandano": "ARANDANO"}
cult_choice = st.sidebar.radio("Cultivo", ["(Todos)", "Palto", "Arandano"], index=0, key="cult_btn")
cult_sel = cult_label_to_val[cult_choice]
df_tmp = df_base.copy()
if cult_sel:
    df_tmp = df_tmp[df_tmp["CULTIVO"] == cult_sel]

turn_label_to_val = {"(Todos)": None, "Día": "DIA", "Noche": "NOCHE"}
turn_choice = st.sidebar.radio("Turno", ["(Todos)", "Día", "Noche"], index=0, key="turn_btn")
turn_sel = turn_label_to_val[turn_choice]
if turn_sel:
    df_tmp = df_tmp[df_tmp["TURNO_NORM"] == turn_sel]

# Propietario / Familia (desde EQUIPOS.csv, aplicado a Tractor y/o Implemento en el turno)
prop_opts = ["(Todos)"]
if "TRC_PROPIETARIO" in df_tmp.columns or "IMP_PROPIETARIO" in df_tmp.columns:
    vals = pd.concat([
        df_tmp.get("TRC_PROPIETARIO", pd.Series(dtype=str)),
        df_tmp.get("IMP_PROPIETARIO", pd.Series(dtype=str)),
    ], ignore_index=True)
    vals = vals.dropna().astype(str).str.upper().unique().tolist()
    vals = sorted([v for v in vals if v and v != "NAN"])
    prop_opts += vals

prop_sel = st.sidebar.selectbox("Propietario (AGK/ALQUILADO)", prop_opts, index=0, key="prop_sel")
if prop_sel != "(Todos)" and ("TRC_PROPIETARIO" in df_tmp.columns or "IMP_PROPIETARIO" in df_tmp.columns):
    m_prop = False
    if "TRC_PROPIETARIO" in df_tmp.columns:
        m_prop = m_prop | (df_tmp["TRC_PROPIETARIO"].astype(str).str.upper() == str(prop_sel))
    if "IMP_PROPIETARIO" in df_tmp.columns:
        m_prop = m_prop | (df_tmp["IMP_PROPIETARIO"].astype(str).str.upper() == str(prop_sel))
    df_tmp = df_tmp[m_prop]

fam_opts = ["(Todos)"]
if "TRC_FAMILIA" in df_tmp.columns or "IMP_FAMILIA" in df_tmp.columns:
    vals = pd.concat([
        df_tmp.get("TRC_FAMILIA", pd.Series(dtype=str)),
        df_tmp.get("IMP_FAMILIA", pd.Series(dtype=str)),
    ], ignore_index=True)
    vals = vals.dropna().astype(str).str.upper().unique().tolist()
    vals = sorted([v for v in vals if v and v != "NAN"])
    fam_opts += vals

fam_sel = st.sidebar.selectbox("Familia", fam_opts, index=0, key="fam_sel")
if fam_sel != "(Todos)" and ("TRC_FAMILIA" in df_tmp.columns or "IMP_FAMILIA" in df_tmp.columns):
    m_fam = False
    if "TRC_FAMILIA" in df_tmp.columns:
        m_fam = m_fam | (df_tmp["TRC_FAMILIA"].astype(str).str.upper() == str(fam_sel))
    if "IMP_FAMILIA" in df_tmp.columns:
        m_fam = m_fam | (df_tmp["IMP_FAMILIA"].astype(str).str.upper() == str(fam_sel))
    df_tmp = df_tmp[m_fam]

trc_opts = ["(Todos)"] + sorted(df_tmp["ID_TRACTOR"].dropna().astype(str).unique().tolist())
trc_sel = st.sidebar.selectbox("Tractor", trc_opts, index=0, key="trc_sel")
if trc_sel != "(Todos)":
    df_tmp = df_tmp[df_tmp["ID_TRACTOR"].astype(str) == str(trc_sel)]

imp_opts = ["(Todos)"] + sorted(df_tmp["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
imp_sel = st.sidebar.selectbox("Implemento", imp_opts, index=0, key="imp_sel")
if imp_sel != "(Todos)":
    df_tmp = df_tmp[df_tmp["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]

# Proceso (ordenado por TO implemento)
ids_for_rank = set(df_tmp["ID_TURNO"].astype(str).tolist())
h_rank = horometros[horometros["ID_TURNO"].astype(str).isin(ids_for_rank)].copy()
h_rank = h_rank[h_rank["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()

turn_proc = df_tmp[["ID_TURNO", "ID_PROCESO"]].copy()
turn_proc["ID_TURNO"] = turn_proc["ID_TURNO"].astype(str)

h_rank = h_rank.merge(turn_proc, on="ID_TURNO", how="left")
to_by_proc = h_rank.groupby("ID_PROCESO", dropna=True)["TO_HORO"].sum().sort_values(ascending=False)

proc_ids_ordered = [str(x) for x in to_by_proc.index.astype(str).tolist()]
proc_names_ordered = [proc_map.get(pid, f"Proceso {pid}") for pid in proc_ids_ordered]
proc_options = ["(Todos)"] + proc_names_ordered

prev_proc = st.session_state.get("proc_btn", "(Todos)")
if prev_proc not in proc_options:
    prev_proc = "(Todos)"

proc_name_sel = st.sidebar.radio(
    "Proceso (ordenado por TO)",
    proc_options,
    index=proc_options.index(prev_proc),
    key="proc_btn",
)

df_f = df_tmp.copy()
id_proceso_sel = None
if proc_name_sel != "(Todos)":
    name_to_id = {proc_map.get(pid, f"Proceso {pid}"): pid for pid in proc_ids_ordered}
    id_proceso_sel = name_to_id.get(proc_name_sel)
    if id_proceso_sel is not None:
        df_f = df_f[df_f["ID_PROCESO"].astype(str) == str(id_proceso_sel)]

# =========================================================
# SELECCIÓN FINAL (global)
# =========================================================
turnos_sel = df_f.copy()
ids_turno = set(turnos_sel["ID_TURNO"].astype(str).tolist())

horo_sel = horometros[horometros["ID_TURNO"].astype(str).isin(ids_turno)].copy()
ev_sel = eventos[eventos["ID_TURNO"].astype(str).isin(ids_turno)].copy()

fd_sel = None
if fallas_detalle is not None:
    if "ID_TURNO" in fallas_detalle.columns:
        fd_sel = fallas_detalle[fallas_detalle["ID_TURNO"].astype(str).isin(ids_turno)].copy()
    else:
        fd_sel = fallas_detalle.copy()

# =========================================================
# KPI GLOBAL (base en EVENTOS + HOROMETROS)
# =========================================================
to_trac = float(horo_sel.loc[horo_sel["TIPO_EQUIPO"].astype(str).str.upper() == "TRACTOR", "TO_HORO"].sum())
to_imp = float(horo_sel.loc[horo_sel["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO", "TO_HORO"].sum())

ev_fallas = ev_sel[ev_sel["CATEGORIA_EVENTO"].astype(str).str.upper() == "FALLA"].copy()
if not ev_fallas.empty:
    ev_fallas["DT_HR"] = pd.to_numeric(ev_fallas["DT_MIN"], errors="coerce") / 60.0
else:
    ev_fallas["DT_HR"] = pd.Series(dtype=float)

def fallas_de_equipo(cod_equipo: str):
    return ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str) == str(cod_equipo)]

if vista_disp == "Sistema (TRC+IMP)":
    to_base = max(to_imp, 0.0)
    dt_base = float(ev_fallas["DT_HR"].sum()) if not ev_fallas.empty else 0.0
    n_base = int(len(ev_fallas))
elif vista_disp == "Tractor":
    to_base = max(to_trac, 0.0)
    if trc_sel != "(Todos)":
        ev_b = fallas_de_equipo(trc_sel)
    else:
        trcs = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    dt_base = float(ev_b["DT_HR"].sum()) if not ev_b.empty else 0.0
    n_base = int(len(ev_b))
else:  # Implemento
    to_base = max(to_imp, 0.0)
    if imp_sel != "(Todos)":
        ev_b = fallas_de_equipo(imp_sel)
    else:
        imps = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]
    dt_base = float(ev_b["DT_HR"].sum()) if not ev_b.empty else 0.0
    n_base = int(len(ev_b))

mttr_hr = (dt_base / n_base) if n_base > 0 else np.nan
mtbf_hr = (to_base / n_base) if n_base > 0 else np.nan
disp = (to_base / (to_base + dt_base)) if (to_base is not None and (to_base + dt_base) > 0) else np.nan

# =========================================================
# PÁGINAS
# =========================================================
if page == "Dashboard":
    st.title("DESEMPEÑO OPERACIONAL DE LA FLOTA")
    st.caption("INDICADORES DE CONFIABILIDAD - MANTENIBILIDAD - DISPONIBILIDAD")
    st.caption(f"Vista actual de KPIs: **{vista_disp}**")

    row1 = [
        kpi_card_html("Tiempo de Operación (h)", fmt_num(to_base)),
        kpi_card_html("Downtime Fallas (h)", fmt_num(dt_base)),
        kpi_card_html("Fallas", f"{n_base:,}"),
    ]
    render_kpi_row(row1, height=190, big=False)

    row2 = [
        kpi_card_html("MTTR (h/falla)", fmt_num(mttr_hr), color=mttr_color_3(mttr_hr),
                      hint="Azul <1.2 | Verde 1.2–2.5 | Rojo >2.5"),
        kpi_card_html("MTBF (h/falla)", fmt_num(mtbf_hr), color=mtbf_color(mtbf_hr),
                      hint="Rojo <100 | Verde 100–500 | Azul >500"),
        kpi_card_html("Disponibilidad", fmt_pct(disp), color=disp_color(disp),
                      hint="Rojo <90% | Verde 90–95% | Azul >95%"),
    ]
    render_kpi_row(row2, height=210, big=True)

    st.divider()

    # =========================================================
    # EVOLUCIÓN POR MES-AÑO (solo en Dashboard)
    # =========================================================
    st.subheader("Evolución por mes-año (MTTR, MTBF, Disponibilidad, Fallas, TO y Down Time)")

    base = turnos.copy()

    if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
        d1 = pd.to_datetime(date_range[0])
        d2 = pd.to_datetime(date_range[1])
        base = base[(base["FECHA"] >= d1) & (base["FECHA"] <= d2)]

    if cult_sel:
        base = base[base["CULTIVO"] == cult_sel]
    if turn_sel:
        base = base[base["TURNO_NORM"] == turn_sel]
    if trc_sel != "(Todos)":
        base = base[base["ID_TRACTOR"].astype(str) == str(trc_sel)]
    if imp_sel != "(Todos)":
        base = base[base["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]
    if id_proceso_sel is not None:
        base = base[base["ID_PROCESO"].astype(str) == str(id_proceso_sel)]

    if base.empty:
        st.info("Con los filtros actuales, no hay datos para construir la evolución mensual.")
    else:
        base = base.copy()
        base["MES"] = base["FECHA"].dt.to_period("M").astype(str)
        ids = set(base["ID_TURNO"].astype(str).tolist())

        h = horometros[horometros["ID_TURNO"].astype(str).isin(ids)].copy()
        e = eventos[eventos["ID_TURNO"].astype(str).isin(ids)].copy()

        e = e[e["CATEGORIA_EVENTO"].astype(str).str.upper() == "FALLA"].copy()
        e["DT_HR"] = pd.to_numeric(e["DT_MIN"], errors="coerce") / 60.0

        turn_mes = base[["ID_TURNO", "MES", "ID_TRACTOR", "ID_IMPLEMENTO"]].copy()
        turn_mes["ID_TURNO"] = turn_mes["ID_TURNO"].astype(str)

        h2 = h.merge(turn_mes[["ID_TURNO", "MES"]], on="ID_TURNO", how="left")

        if vista_disp == "Tractor":
            h2 = h2[h2["TIPO_EQUIPO"].astype(str).str.upper() == "TRACTOR"].copy()
        elif vista_disp == "Implemento":
            h2 = h2[h2["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()
        else:
            h2 = h2[h2["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()

        to_mes = h2.groupby("MES", dropna=True)["TO_HORO"].sum().reset_index(name="TO_HR")

        e2 = e.merge(turn_mes, on="ID_TURNO", how="left")

        if vista_disp == "Tractor":
            trcs = base["ID_TRACTOR"].dropna().astype(str).unique().tolist()
            e2 = e2[e2["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
        elif vista_disp == "Implemento":
            imps = base["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
            e2 = e2[e2["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]

        dt_mes = e2.groupby("MES", dropna=True).agg(
            DT_HR=("DT_HR", "sum"),
            FALLAS=("DT_HR", "size")
        ).reset_index()

        evo = to_mes.merge(dt_mes, on="MES", how="left").fillna({"DT_HR": 0.0, "FALLAS": 0})
        evo["FALLAS"] = evo["FALLAS"].astype(int)

        evo["MTTR_HR"] = np.where(evo["FALLAS"] > 0, evo["DT_HR"] / evo["FALLAS"], np.nan)
        evo["MTBF_HR"] = np.where(evo["FALLAS"] > 0, evo["TO_HR"] / evo["FALLAS"], np.nan)
        evo["DISP"] = np.where((evo["TO_HR"] + evo["DT_HR"]) > 0, evo["TO_HR"] / (evo["TO_HR"] + evo["DT_HR"]), np.nan)
        evo = evo.sort_values("MES", ascending=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig1 = px.bar(evo, x="MES", y="MTTR_HR", title="MTTR (h/falla) por mes")
            fig1.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig1, width="stretch")
        with r1c2:
            fig2 = px.bar(evo, x="MES", y="MTBF_HR", title="MTBF (h/falla) por mes")
            fig2.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig2, width="stretch")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig3 = px.bar(evo, x="MES", y="DISP", title="Disponibilidad (TO/(TO+DT)) por mes")
            fig3.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20), yaxis_tickformat=".0%")
            st.plotly_chart(fig3, width="stretch")
        with r2c2:
            fig4 = px.bar(evo, x="MES", y="FALLAS", title="Cantidad de fallas por mes")
            fig4.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig4, width="stretch")

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            fig5 = px.bar(evo, x="MES", y="TO_HR", title="Tiempo de Operación (TO) por mes (h)")
            fig5.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig5, width="stretch")
        with r3c2:
            fig6 = px.bar(evo, x="MES", y="DT_HR", title="Down Time por mes (h)")
            fig6.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig6, width="stretch")

    st.subheader("Descargar datos filtrados")
    st.download_button(
        "Descargar TURNOS filtrado (CSV)",
        data=turnos_sel.to_csv(index=False).encode("utf-8-sig"),
        file_name="TURNOS_filtrado.csv",
        mime="text/csv",
    )
    st.download_button(
        "Descargar HOROMETROS filtrado (CSV)",
        data=horo_sel.to_csv(index=False).encode("utf-8-sig"),
        file_name="HOROMETROS_TURNO_filtrado.csv",
        mime="text/csv",
    )
    st.download_button(
        "Descargar EVENTOS filtrado (CSV)",
        data=ev_sel.to_csv(index=False).encode("utf-8-sig"),
        file_name="EVENTOS_filtrado.csv",
        mime="text/csv",
    )
    if fd_sel is not None:
        st.download_button(
            "Descargar FALLAS_DETALLE filtrado (CSV)",
            data=fd_sel.to_csv(index=False).encode("utf-8-sig"),
            file_name="FALLAS_DETALLE_filtrado.csv",
            mime="text/csv",
        )

elif page == "Paretos":
    st.title("Paretos de Fallas + Mapas de calor")
    st.caption("Paretos por Down Time (h) y heatmaps por Equipo vs Nivel (SUBUNIDAD/COMPONENTE/PARTE).")

    if fd_sel is None or fd_sel.empty:
        st.info("No se encontró **FALLAS_DETALLE_NORMALIZADO.csv** o no hay datos con los filtros actuales.")
        st.stop()

    fd_view = norm_cols(fd_sel.copy())

    if "ID_TURNO" not in fd_view.columns and "ID" in fd_view.columns:
        fd_view["ID_TURNO"] = safe_norm_str_series(fd_view["ID"])

    if "DOWNTIME_HR" not in fd_view.columns:
        tf = find_first_col(fd_view, ["T_FALLA", "TFALLA", "TIEMPO_FALLA", "TIEMPO_DE_FALLA", "DOWNTIME", "DT_HR"])
        if tf is not None:
            fd_view["DOWNTIME_HR"] = _coerce_downtime_to_hr(fd_view[tf])
        else:
            fd_view["DOWNTIME_HR"] = np.nan
    fd_view["DOWNTIME_HR"] = pd.to_numeric(fd_view["DOWNTIME_HR"], errors="coerce").fillna(0.0)

    if "SUBUNIDAD" not in fd_view.columns and "SUB_UNIDAD" in fd_view.columns:
        fd_view["SUBUNIDAD"] = fd_view["SUB_UNIDAD"]
    if "PARTE" not in fd_view.columns and "PIEZA" in fd_view.columns:
        fd_view["PARTE"] = fd_view["PIEZA"]
    if "CAUSA_FALLA" not in fd_view.columns and "CAUSA" in fd_view.columns:
        fd_view["CAUSA_FALLA"] = fd_view["CAUSA"]

    equipo_col = find_first_col(fd_view, ["EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO"])
    if equipo_col is None:
        st.info("No encontré columna de Equipo (EQUIPO/ID_EQUIPO/ID_EQUIPO_AFECTADO).")
        st.stop()

    if "TIPO_EQUIPO" not in fd_view.columns or fd_view["TIPO_EQUIPO"].isna().all():
        fd_view["TIPO_EQUIPO"] = fd_view[equipo_col].apply(infer_tipo_equipo_from_code)
    else:
        fd_view["TIPO_EQUIPO"] = fd_view["TIPO_EQUIPO"].astype(str).str.upper().str.strip()

    if vista_disp == "Tractor":
        fd_view = fd_view[fd_view["TIPO_EQUIPO"] == "TRACTOR"].copy()
    elif vista_disp == "Implemento":
        fd_view = fd_view[fd_view["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

    if fd_view.empty:
        st.info("Con los filtros actuales, no hay registros en FALLAS_DETALLE para construir Paretos/Heatmap.")
        st.stop()

    st.subheader("Paretos (Down Time h)")
    top_n = st.slider("Top N (Pareto)", min_value=5, max_value=40, value=10, step=1, key="pareto_topn")

    c1, c2, c3 = st.columns(3)
    with c1:
        pareto_chart(fd_view, "SUBUNIDAD", "DOWNTIME_HR", top_n, f"Pareto SUBUNIDAD (Top {top_n})")
    with c2:
        pareto_chart(fd_view, "COMPONENTE", "DOWNTIME_HR", top_n, f"Pareto COMPONENTE (Top {top_n})")
    with c3:
        pareto_chart(fd_view, "PARTE", "DOWNTIME_HR", top_n, f"Pareto PARTE (Top {top_n})")

    c4, c5 = st.columns(2)
    with c4:
        pareto_chart(fd_view, "VERBO_TECNICO", "DOWNTIME_HR", top_n, f"Pareto VERBO TÉCNICO (Top {top_n})")
    with c5:
        pareto_chart(fd_view, "CAUSA_FALLA", "DOWNTIME_HR", top_n, f"Pareto CAUSA DE FALLA (Top {top_n})")

    st.divider()

    # -------------------------
    # Controles compartidos TopX/TopY
    # -------------------------
    st.subheader("Mapas de calor (Equipos vs Nivel)")
    y_options = [c for c in ["SUBUNIDAD", "COMPONENTE", "PARTE"] if c in fd_view.columns]
    if not y_options:
        st.info("No encontré columnas para el eje Y (SUBUNIDAD/COMPONENTE/PARTE).")
        st.stop()

    colA, colB, colC = st.columns(3)
    with colA:
        hm_y_level = st.selectbox("Eje Y (nivel)", y_options, index=0, key="hm_y_level_shared")
    with colB:
        hm_top_equipos = st.slider("Top equipos (X)", min_value=10, max_value=200, value=40, step=5, key="hm_top_equipos_shared")
    with colC:
        hm_top_y = st.slider("Top Y (filas)", min_value=5, max_value=80, value=20, step=5, key="hm_top_y_shared")

    # -------------------------
    # HEATMAP 1: Down Time (h)
    # -------------------------
    st.markdown("### Heatmap 1 — Down Time (h)")

    tmp_dt = fd_view.copy()
    tmp_dt[equipo_col] = tmp_dt[equipo_col].astype(str).str.upper().str.strip()
    tmp_dt[hm_y_level] = tmp_dt[hm_y_level].astype(str).str.upper().str.strip()

    top_eq_dt = (
        tmp_dt.groupby(equipo_col)["DOWNTIME_HR"].sum()
        .sort_values(ascending=False)
        .head(int(hm_top_equipos))
        .index.tolist()
    )
    top_y_dt = (
        tmp_dt.groupby(hm_y_level)["DOWNTIME_HR"].sum()
        .sort_values(ascending=False)
        .head(int(hm_top_y))
        .index.tolist()
    )

    tmp_dt = tmp_dt[tmp_dt[equipo_col].isin(top_eq_dt) & tmp_dt[hm_y_level].isin(top_y_dt)].copy()

    if tmp_dt.empty:
        st.info("No hay datos para el heatmap de Down Time con los Top seleccionados.")
    else:
        pivot_dt = tmp_dt.pivot_table(
            index=hm_y_level,
            columns=equipo_col,
            values="DOWNTIME_HR",
            aggfunc="sum",
            fill_value=0.0
        )
        pivot_dt = pivot_dt.loc[pivot_dt.sum(axis=1).sort_values(ascending=False).index]
        pivot_dt = pivot_dt[pivot_dt.sum(axis=0).sort_values(ascending=False).index]

        fig_hm_dt = px.imshow(
            pivot_dt,
            aspect="auto",
            labels=dict(x="Equipos", y=hm_y_level, color="Down Time (h)"),
        )
        fig_hm_dt.update_layout(
            title=f"Down Time (h) — X: Equipos | Y: {hm_y_level}",
            title_x=0.5,
            margin=dict(l=20, r=20, t=70, b=40),
        )
        st.plotly_chart(fig_hm_dt, width="stretch")

    # -------------------------
    # HEATMAP 2: # Fallas (count)
    # -------------------------
    st.markdown("### Heatmap 2 — Cantidad de fallas (#)")

    tmp_ct = fd_view.copy()
    tmp_ct[equipo_col] = tmp_ct[equipo_col].astype(str).str.upper().str.strip()
    tmp_ct[hm_y_level] = tmp_ct[hm_y_level].astype(str).str.upper().str.strip()

    top_eq_ct = (
        tmp_ct.groupby(equipo_col).size()
        .sort_values(ascending=False)
        .head(int(hm_top_equipos))
        .index.tolist()
    )
    top_y_ct = (
        tmp_ct.groupby(hm_y_level).size()
        .sort_values(ascending=False)
        .head(int(hm_top_y))
        .index.tolist()
    )

    tmp_ct = tmp_ct[tmp_ct[equipo_col].isin(top_eq_ct) & tmp_ct[hm_y_level].isin(top_y_ct)].copy()

    if tmp_ct.empty:
        st.info("No hay datos para el heatmap de Cantidad de fallas con los Top seleccionados.")
    else:
        pivot_ct = tmp_ct.pivot_table(
            index=hm_y_level,
            columns=equipo_col,
            values="DOWNTIME_HR",  # no importa cuál; usamos size
            aggfunc="size",
            fill_value=0
        )
        pivot_ct = pivot_ct.loc[pivot_ct.sum(axis=1).sort_values(ascending=False).index]
        pivot_ct = pivot_ct[pivot_ct.sum(axis=0).sort_values(ascending=False).index]

        fig_hm_ct = px.imshow(
            pivot_ct,
            aspect="auto",
            labels=dict(x="Equipos", y=hm_y_level, color="# Fallas"),
        )
        fig_hm_ct.update_layout(
            title=f"# Fallas — X: Equipos | Y: {hm_y_level}",
            title_x=0.5,
            margin=dict(l=20, r=20, t=70, b=40),
        )
        st.plotly_chart(fig_hm_ct, width="stretch")

else:  # page == "Técnico"
    st.title("Dashboard Técnico (Para acción y mejora)")
    st.caption("Objetivo: ¿Qué falla? ¿Dónde intervenir primero? ¿Preventivo o correctivo? ¿Qué atacar con RCM?")

    # ---------------------------------
    # Base de fallas (ya filtrada por panel global)
    # ---------------------------------
    if fd_sel is None or fd_sel.empty:
        st.warning("Con los filtros actuales, no hay registros en FALLAS_DETALLE para mostrar en 'Técnico'.")
        st.stop()

    fd = norm_cols(fd_sel.copy())

    if "ID_TURNO" not in fd.columns and "ID" in fd.columns:
        fd["ID_TURNO"] = safe_norm_str_series(fd["ID"])

    # Downtime
    if "DOWNTIME_HR" not in fd.columns:
        tf = find_first_col(fd, ["T_FALLA", "TFALLA", "TIEMPO_FALLA", "TIEMPO_DE_FALLA", "DOWNTIME", "DT_HR", "DT_MIN"])
        if tf is not None:
            if tf == "DT_MIN":
                fd["DOWNTIME_HR"] = pd.to_numeric(fd[tf], errors="coerce") / 60.0
            else:
                fd["DOWNTIME_HR"] = _coerce_downtime_to_hr(fd[tf])
        else:
            fd["DOWNTIME_HR"] = 0.0
    fd["DOWNTIME_HR"] = pd.to_numeric(fd["DOWNTIME_HR"], errors="coerce").fillna(0.0)

    # Alias columnas
    if "SUBUNIDAD" not in fd.columns and "SUB_UNIDAD" in fd.columns:
        fd["SUBUNIDAD"] = fd["SUB_UNIDAD"]
    if "PARTE" not in fd.columns and "PIEZA" in fd.columns:
        fd["PARTE"] = fd["PIEZA"]

    equipo_col = find_first_col(fd, ["EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO"])
    if equipo_col is None:
        st.error("FALLAS_DETALLE no tiene columna de EQUIPO/ID_EQUIPO/ID_EQUIPO_AFECTADO.")
        st.stop()

    # Tipificación / vista
    fd[equipo_col] = fd[equipo_col].astype(str).str.upper().str.strip()

    trc_ids = set(turnos_sel["ID_TRACTOR"].dropna().astype(str).str.upper().tolist())
    imp_ids = set(turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).str.upper().tolist())

    # Restringir fallas según vista KPI (para que DT y #fallas cambien como en Dashboard)
    if vista_disp == "Tractor":
        fd_vista = fd[fd[equipo_col].isin(trc_ids)].copy()
        horo_vista = horo_sel[horo_sel["TIPO_EQUIPO"].astype(str).str.upper() == "TRACTOR"].copy()
    elif vista_disp == "Implemento":
        fd_vista = fd[fd[equipo_col].isin(imp_ids)].copy()
        horo_vista = horo_sel[horo_sel["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()
    else:
        # Sistema (TRC+IMP): fallas de ambos; TO base sigue siendo implemento (como en Dashboard)
        fd_vista = fd[fd[equipo_col].isin(trc_ids.union(imp_ids))].copy()
        horo_vista = horo_sel[horo_sel["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()

    if fd_vista.empty:
        st.warning("Con los filtros actuales y la vista seleccionada, no hay fallas para mostrar en 'Técnico'.")
        st.stop()

    # Columnas de niveles
    sistema_col = "SUBUNIDAD" if "SUBUNIDAD" in fd_vista.columns else ("SUB_UNIDAD" if "SUB_UNIDAD" in fd_vista.columns else None)
    comp_col = "COMPONENTE" if "COMPONENTE" in fd_vista.columns else None
    parte_col = "PARTE" if "PARTE" in fd_vista.columns else None

    # ---------------------------------
    # 1) Cascada (sobre el conjunto ya filtrado por panel global + vista)
    # ---------------------------------
    st.subheader("1) Filtros jerárquicos (cascada)")

    eq_all = ["(Todos)"] + sorted(fd_vista[equipo_col].dropna().astype(str).unique().tolist())

    c0a, c0b = st.columns([3, 2])
    with c0a:
        eq_sel = st.selectbox("Equipo", eq_all, index=0, key="tec_eq")
    with c0b:
        st.caption("Cascada: Equipo → Sub unidad → Componente → Parte")

    df_lvl = fd_vista.copy()
    if eq_sel != "(Todos)":
        df_lvl = df_lvl[df_lvl[equipo_col].astype(str) == str(eq_sel)].copy()

    def casc_select(label, colname, df_in, key, disabled=False):
        if disabled or colname is None or colname not in df_in.columns:
            st.selectbox(label, ["(No disponible)"], index=0, key=key, disabled=True)
            return "(No disponible)", df_in
        opts = ["(Todos)"] + sorted(df_in[colname].dropna().astype(str).unique().tolist())
        val = st.selectbox(label, opts, index=0, key=key)
        if val != "(Todos)":
            df_in = df_in[df_in[colname].astype(str) == str(val)].copy()
        return val, df_in

    c1, c2, c3 = st.columns(3)
    with c1:
        sis_sel, df_lvl = casc_select("Sub unidad", sistema_col, df_lvl, "tec_sis", disabled=False)
    with c2:
        com_sel, df_lvl = casc_select("Componente", comp_col, df_lvl, "tec_com", disabled=(comp_col is None))
    with c3:
        par_sel, df_lvl = casc_select("Parte", parte_col, df_lvl, "tec_par", disabled=(parte_col is None))

    st.divider()

    # ---------------------------------
    # 2) KPIs técnicos (responden a panel global + cascada + vista)
    # ---------------------------------
    st.subheader("2) KPIs técnicos (ya filtrados)")

    # TO base: igual a Dashboard para comparación; si eliges un equipo, usa TO real de ese equipo
    if eq_sel != "(Todos)":
        to_real = float(horo_sel.loc[horo_sel["ID_EQUIPO"].astype(str).str.upper() == str(eq_sel), "TO_HORO"].sum())
    else:
        to_real = float(horo_vista["TO_HORO"].sum()) if not horo_vista.empty else 0.0

    n_fallas = int(len(df_lvl))
    dt_hr = float(df_lvl["DOWNTIME_HR"].sum()) if not df_lvl.empty else 0.0
    mttr = (dt_hr / n_fallas) if n_fallas > 0 else np.nan
    mtbf = (to_real / n_fallas) if n_fallas > 0 else np.nan
    disp_tec = (to_real / (to_real + dt_hr)) if (to_real + dt_hr) > 0 else np.nan

    cards = [
        kpi_card_html("Horas operadas reales (TO)", fmt_num(to_real), hint="Misma base que Dashboard; si eliges un equipo, usa su TO real"),
        kpi_card_html("Down Time (h)", fmt_num(dt_hr), hint="Suma de DOWNTIME_HR (FALLAS_DETALLE)"),
        kpi_card_html("N° de fallas", f"{n_fallas:,}", hint="Conteo (FALLAS_DETALLE)"),
        kpi_card_html("MTTR (h/falla)", fmt_num(mttr), hint="DT / #fallas"),
        kpi_card_html("MTBF (h/falla)", fmt_num(mtbf), hint="TO / #fallas"),
        kpi_card_html("Disponibilidad", fmt_pct(disp_tec), hint="TO / (TO + DT)"),
    ]
    render_kpi_row(cards, height=205)

    # ---------------------------------
    # 2B) Asistente de Confiabilidad (IA) — usa filtros + DF ya filtrados
    # ---------------------------------
    st.subheader("🧠 Asistente de Confiabilidad (IA)")
    st.caption("Haz preguntas sobre MTTR/MTBF/Disponibilidad y te devolverá recomendaciones accionables basadas en los datos filtrados.")

    client = get_openai_client()
    if client is None:
        st.info("Para activar el asistente, configura **OPENAI_API_KEY** en *Streamlit Cloud → App settings → Secrets*.")
    else:
        # Preparar Top 10 rápidos para contexto (sobre el nivel más detallado disponible)
        top_level = comp_col if (comp_col and comp_col in df_lvl.columns) else parte_col
        top_tables = {}
        if top_level is not None and top_level in df_lvl.columns and not df_lvl.empty:
            g_ai = df_lvl.groupby(top_level, dropna=True).agg(
                FALLAS=("DOWNTIME_HR", "size"),
                DT_HR=("DOWNTIME_HR", "sum")
            ).reset_index().rename(columns={top_level: "ITEM"})
            g_ai["ITEM"] = g_ai["ITEM"].astype(str)
            g_ai["MTTR_HR"] = np.where(g_ai["FALLAS"] > 0, g_ai["DT_HR"] / g_ai["FALLAS"], np.nan)
            g_ai["MTBF_HR"] = np.where(g_ai["FALLAS"] > 0, to_real / g_ai["FALLAS"], np.nan)

            top_tables = {
                "top_level": top_level,
                "top_mttr": g_ai.dropna(subset=["MTTR_HR"]).sort_values("MTTR_HR", ascending=False).head(10)[["ITEM","FALLAS","DT_HR","MTTR_HR"]],
                "top_mtbf": g_ai.dropna(subset=["MTBF_HR"]).sort_values("MTBF_HR", ascending=True).head(10)[["ITEM","FALLAS","DT_HR","MTBF_HR"]],
                "top_dt": g_ai.sort_values("DT_HR", ascending=False).head(10)[["ITEM","FALLAS","DT_HR"]],
            }

        left, right = st.columns([2, 1], gap="large")

        with right:
            st.markdown("#### Contexto usado por el asistente")
            st.write("**KPIs filtrados**")
            st.json({
                "TO_real_h": float(to_real),
                "DT_h": float(dt_hr),
                "fallas": int(n_fallas),
                "MTTR_h_falla": None if pd.isna(mttr) else float(mttr),
                "MTBF_h_falla": None if pd.isna(mtbf) else float(mtbf),
                "Disponibilidad": None if pd.isna(disp_tec) else float(disp_tec),
                "Vista": vista_disp,
            }, expanded=False)

            st.write("**Muestra de fallas filtradas**")
            st.dataframe(df_lvl.head(30), use_container_width=True, height=260)

            if isinstance(top_tables, dict) and top_tables.get("top_level"):
                st.write(f"**Top 10 (nivel: {top_tables['top_level']})**")
                st.dataframe(top_tables["top_mttr"], use_container_width=True, height=220)

            if st.button("🧹 Limpiar chat", key="ai_clear"):
                st.session_state.pop("ai_msgs", None)
                st.rerun()

        with left:
            if "ai_msgs" not in st.session_state:
                st.session_state.ai_msgs = [
                    {"role": "assistant", "content": "Hola. Soy tu **Asistente de Confiabilidad**. Pregúntame, por ejemplo: *¿Qué debo atacar primero para subir disponibilidad?*"}
                ]

            for m in st.session_state.ai_msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            user_q = st.chat_input("Escribe tu consulta (usa los filtros actuales)…", key="ai_input")
            if user_q:
                st.session_state.ai_msgs.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                kpis = {
                    "TO_real_h": float(to_real),
                    "DT_h": float(dt_hr),
                    "fallas": int(n_fallas),
                    "MTTR_h_falla": None if pd.isna(mttr) else float(mttr),
                    "MTBF_h_falla": None if pd.isna(mtbf) else float(mtbf),
                    "Disponibilidad": None if pd.isna(disp_tec) else float(disp_tec),
                }
                top_tables_text = {}
                if isinstance(top_tables, dict) and top_tables.get("top_level"):
                    top_tables_text = {
                        "top_level": top_tables["top_level"],
                        "top_mttr": top_tables["top_mttr"].to_dict(orient="records"),
                        "top_mtbf": top_tables["top_mtbf"].to_dict(orient="records"),
                        "top_dt": top_tables["top_dt"].to_dict(orient="records"),
                    }

                ctx = build_reliability_context(
                    vista_disp=vista_disp,
                    date_range=date_range,
                    cult_choice=cult_choice,
                    turn_choice=turn_choice,
                    prop_sel=prop_sel,
                    fam_sel=fam_sel,
                    trc_sel=trc_sel,
                    imp_sel=imp_sel,
                    proc_name_sel=proc_name_sel,
                    eq_sel=eq_sel,
                    sis_sel=sis_sel if "sis_sel" in locals() else "(N/A)",
                    com_sel=com_sel if "com_sel" in locals() else "(N/A)",
                    par_sel=par_sel if "par_sel" in locals() else "(N/A)",
                    kpis=kpis,
                    df_lvl=df_lvl,
                    top_tables=top_tables_text,
                )

                instructions = (
                    "Eres un asistente senior de confiabilidad (RCM II / TPM) para una flota agrícola. "
                    "Usa SOLO el contexto provisto (KPIs y tablas) para sustentar tus conclusiones. "
                    "Entrega: (1) diagnóstico breve, (2) 3-7 acciones priorizadas, (3) hipótesis de causa raíz, "
                    "(4) qué datos faltan para confirmar, y (5) qué preventivos/predictivos sugerir."
                )

                with st.chat_message("assistant"):
                    with st.spinner("Pensando…"):
                        try:
                            model_name = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
                        except Exception:
                            model_name = "gpt-4o-mini"

                    from openai import RateLimitError:
         		try:
   			   resp = client.responses.create(
       			       model=model_name,
       			       input=[
           		           {"role": "system", "content": instructions},
            		           {"role": "user", "content": user_prompt},
        		       ],
    			   )
   			   answer = resp.output_text  # o el extractor que ya estés usando
			except RateLimitError:
    			   st.error("La API de OpenAI rechazó la solicitud por falta de cuota (billing). Activa Billing o recarga crédito en OpenAI Platform.")
    			   answer = None
			except Exception as e:
    			   st.error(f"Error llamando a OpenAI: {e}")
    			   answer = None

			if answer:
    			   st.write(answer)

    st.divider()

    # ---------------------------------
    # 3) Análisis por nivel (barras) — 3 gráficos en una fila
    # ---------------------------------
    st.subheader("3) Análisis de fallas por nivel (barras)")

    def bar_fallas(df_in, col, title, top=15):
        if col is None or col not in df_in.columns:
            return None
        g = df_in.groupby(col, dropna=True).size().reset_index(name="FALLAS")
        g[col] = g[col].astype(str)
        g = g.sort_values("FALLAS", ascending=False).head(int(top))
        if g.empty:
            return None
        fig = px.bar(g, x="FALLAS", y=col, orientation="h", title=title)
        fig.update_layout(title_x=0.5, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="")
        return fig

    top_barras = st.slider("Top por gráfico", min_value=5, max_value=30, value=15, step=1, key="tec_top_barras")

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        fig = bar_fallas(df_lvl, sistema_col, "Fallas por Sub unidad", top=top_barras)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para Sub unidad.")
    with bc2:
        fig = bar_fallas(df_lvl, comp_col, "Fallas por Componente", top=top_barras)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para Componente.")
    with bc3:
        fig = bar_fallas(df_lvl, parte_col, "Fallas por Parte", top=top_barras)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para Parte.")

    st.divider()

    # ---------------------------------
    # 4) Top técnicos (dónde atacar primero)
    # ---------------------------------
    st.subheader("4) Top técnicos (dónde atacar primero)")

    top_level = comp_col if (comp_col and comp_col in df_lvl.columns) else parte_col
    if top_level is None or top_level not in df_lvl.columns:
        st.info("No hay COMPONENTE/PARTE para construir Top técnicos.")
    else:
        g = df_lvl.groupby(top_level, dropna=True).agg(
            FALLAS=("DOWNTIME_HR", "size"),
            DT_HR=("DOWNTIME_HR", "sum")
        ).reset_index().rename(columns={top_level: "ITEM"})
        g["ITEM"] = g["ITEM"].astype(str)
        g["MTTR_HR"] = np.where(g["FALLAS"] > 0, g["DT_HR"] / g["FALLAS"], np.nan)
        g["MTBF_HR"] = np.where(g["FALLAS"] > 0, to_real / g["FALLAS"], np.nan)

        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(center_title(f"Top 10 {top_level} con MTTR alto"), unsafe_allow_html=True)
            d = g.dropna(subset=["MTTR_HR"]).sort_values("MTTR_HR", ascending=False).head(10)
            st.plotly_chart(px.bar(d, x="MTTR_HR", y="ITEM", orientation="h"), use_container_width=True)
        with t2:
            st.markdown(center_title(f"Top 10 {top_level} con MTBF bajo"), unsafe_allow_html=True)
            d = g.dropna(subset=["MTBF_HR"]).sort_values("MTBF_HR", ascending=True).head(10)
            st.plotly_chart(px.bar(d, x="MTBF_HR", y="ITEM", orientation="h"), use_container_width=True)
        with t3:
            st.markdown(center_title(f"Top 10 {top_level} con Down Time alto"), unsafe_allow_html=True)
            d = g.sort_values("DT_HR", ascending=False).head(10)
            st.plotly_chart(px.bar(d, x="DT_HR", y="ITEM", orientation="h"), use_container_width=True)
