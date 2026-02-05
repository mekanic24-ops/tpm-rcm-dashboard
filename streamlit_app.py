# streamlit_app.py
import zipfile
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="DESEMPEÑO OPERACIONAL DE LA FLOTA", layout="wide")

ZIP_NAME = "TPM_modelo_normalizado_CSV.zip"
DATA_DIR = Path("data_normalizada")

# =========================================================
# HELPERS
# =========================================================
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
    # Regla simple: TRC = tractor. Si tienes otros prefijos, agrégalos aquí.
    if s.startswith("TRC"):
        return "TRACTOR"
    return "IMPLEMENTO"


def _coerce_downtime_to_hr(x: pd.Series) -> pd.Series:
    """Convierte una serie a horas; si parece estar en minutos, divide entre 60."""
    v = pd.to_numeric(x, errors="coerce")
    if v.notna().any():
        p95 = np.nanpercentile(v.dropna(), 95)
        # Si el p95 es muy grande, casi seguro está en minutos
        if pd.notna(p95) and p95 > 240:  # >240h es raro; en minutos sería >14400
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

    # NUEVO: fallas detalle técnico
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

    # Normalización IDs comunes
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

    # ==============================
    # FALLAS_DETALLE_NORMALIZADO
    # ==============================
    if fallas_detalle is not None:
        fallas_detalle = norm_cols(fallas_detalle)

        # 1) ID -> ID_TURNO (clave para que filtre por rango de fechas)
        if "ID" in fallas_detalle.columns and "ID_TURNO" not in fallas_detalle.columns:
            fallas_detalle["ID_TURNO"] = norm_str(fallas_detalle["ID"])

        # 2) Normalizar cadenas claves
        for c in ["ID_TURNO", "EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO", "TIPO_EQUIPO"]:
            if c in fallas_detalle.columns:
                fallas_detalle[c] = norm_str(fallas_detalle[c])

        # 3) Downtime: T_FALLA / etc -> DOWNTIME_HR (con heurística min->hr)
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

        # 4) Aliases (los tuyos vienen como SUB_UNIDAD / PIEZA / CAUSA / VERBO_TECNICO)
        alias_map = {
            "SUB_UNIDAD": "SUBUNIDAD",
            "SUBSISTEMA": "SUBUNIDAD",
            "PIEZA": "PARTE",
            "VERBO": "VERBO_TECNICO",
            "VERBO_TECN": "VERBO_TECNICO",
            "VERBO_TECNICO_FALLA": "VERBO_TECNICO",
            "CAUSA": "CAUSA_FALLA",
            "CAUSA_RAIZ": "CAUSA_FALLA",
            "CAUSA_INMEDIATA": "CAUSA_FALLA",
        }
        for src, dst in alias_map.items():
            if src in fallas_detalle.columns and dst not in fallas_detalle.columns:
                fallas_detalle[dst] = fallas_detalle[src]

        # 5) Si no hay TIPO_EQUIPO, lo inferimos por EQUIPO
        eq_col = find_first_col(fallas_detalle, ["EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO"])
        if "TIPO_EQUIPO" not in fallas_detalle.columns or fallas_detalle["TIPO_EQUIPO"].isna().all():
            if eq_col is not None:
                fallas_detalle["TIPO_EQUIPO"] = fallas_detalle[eq_col].apply(infer_tipo_equipo_from_code)

    return {
        "turnos": turnos,
        "horometros": horometros,
        "eventos": eventos,
        "operadores": operadores,
        "lotes": lotes,
        "fallas_cat": fallas_cat,
        "cat_proceso": cat_proceso,
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


def build_enriched_turnos(turnos, operadores, lotes):
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

    return t


def mttr_color_3(v):
    if v is None or pd.isna(v):
        return None
    if v < 1.2:
        return "#1f77b4"
    if v <= 2.5:
        return "#2ca02c"
    return "#d62728"


def mtbf_color(v):
    if v is None or pd.isna(v):
        return None
    if v < 100:
        return "#d62728"
    if 100 <= v <= 500:
        return "#2ca02c"
    return "#1f77b4"


def disp_color(d):
    if d is None or pd.isna(d):
        return None
    p = d * 100
    if p < 90:
        return "#d62728"
    if p <= 95:
        return "#2ca02c"
    return "#1f77b4"


def fmt_num(x, dec=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{dec}f}"


def fmt_pct(x, dec=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:,.{dec}f}%"


def kpi_card_html(title: str, value: str, color: Optional[str] = None, hint: Optional[str] = None) -> str:
    val_style = "color:#111;"
    if color:
        val_style = f"color:{color};"
    hint_text = hint if hint else "&nbsp;"
    return f"""
      <div class="kpi">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="{val_style}">{value}</div>
        <div class="kpi-hint">{hint_text}</div>
      </div>
    """


def render_kpi_row(cards_html: List[str], big: bool = False):
    row_class = "kpi-row big" if big else "kpi-row"
    html = f"""
    <html>
      <head>
        <style>
          .kpi-row{{
            display:flex; gap:28px; justify-content:center; align-items:flex-start;
            flex-wrap:wrap; margin:6px 0 0 0;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          }}
          .kpi{{ width:300px; min-height:150px; padding:8px 10px; text-align:center; box-sizing:border-box; }}
          .kpi-title{{ font-size:16px; font-weight:800; color:#111; margin:0 0 10px 0; line-height:1.2; }}
          .kpi-value{{ font-size:52px; font-weight:400; line-height:1.05; margin:0 0 8px 0; }}
          .kpi-hint{{ font-size:12px; color:#6b7280; line-height:1.2; min-height:16px; }}
          .kpi-row.big .kpi{{ width:320px; min-height:165px; }}
          .kpi-row.big .kpi-value{{ font-size:56px; }}
        </style>
      </head>
      <body style="margin:0;padding:0;">
        <div class="{row_class}">
          {''.join(cards_html)}
        </div>
      </body>
    </html>
    """
    components.html(html, height=185 if big else 175)


def center_title(txt: str) -> str:
    return f"<div style='text-align:center;font-weight:800;font-size:18px;margin:0 0 4px 0;'>{txt}</div>"


# =========================================================
# LOAD
# =========================================================
tables = load_tables()
turnos = build_enriched_turnos(tables["turnos"], tables["operadores"], tables["lotes"])
horometros = tables["horometros"]
eventos = tables["eventos"]
cat_proceso = tables["cat_proceso"]
fallas_detalle = tables["fallas_detalle"]

proc_map: Dict[str, str] = dict(zip(cat_proceso["ID_PROCESO"].astype(str), cat_proceso["NOMBRE_PROCESO"].astype(str)))

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Filtros")

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

# Cultivo (botones)
cult_label_to_val = {"(Todos)": None, "Palto": "PALTO", "Arandano": "ARANDANO"}
cult_choice = st.sidebar.radio("Cultivo", ["(Todos)", "Palto", "Arandano"], index=0, key="cult_btn")
cult_sel = cult_label_to_val[cult_choice]
df_tmp = df_base.copy()
if cult_sel:
    df_tmp = df_tmp[df_tmp["CULTIVO"] == cult_sel]

# Turno (botones)
turn_label_to_val = {"(Todos)": None, "Día": "DIA", "Noche": "NOCHE"}
turn_choice = st.sidebar.radio("Turno", ["(Todos)", "Día", "Noche"], index=0, key="turn_btn")
turn_sel = turn_label_to_val[turn_choice]
if turn_sel:
    df_tmp = df_tmp[df_tmp["TURNO_NORM"] == turn_sel]

# Tractor / Implemento
trc_opts = ["(Todos)"] + sorted(df_tmp["ID_TRACTOR"].dropna().astype(str).unique().tolist())
trc_sel = st.sidebar.selectbox("Tractor", trc_opts, index=0, key="trc_sel")
if trc_sel != "(Todos)":
    df_tmp = df_tmp[df_tmp["ID_TRACTOR"].astype(str) == str(trc_sel)]

imp_opts = ["(Todos)"] + sorted(df_tmp["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
imp_sel = st.sidebar.selectbox("Implemento", imp_opts, index=0, key="imp_sel")
if imp_sel != "(Todos)":
    df_tmp = df_tmp[df_tmp["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]

# Proceso (botones) ordenado por TO del implemento
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

vista_disp = st.sidebar.radio(
    "KPIs basados en:",
    ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
    index=0,
    key="vista_disp",
)

# =========================================================
# SELECCIÓN FINAL
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
# KPI GLOBAL
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
else:
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
# TITULOS
# =========================================================
st.title("DESEMPEÑO OPERACIONAL DE LA FLOTA")
st.caption("INDICADORES DE CONFIABILIDAD - MANTENIBILIDAD - DISPONIBILIDAD")
st.caption(f"Vista actual de KPIs: **{vista_disp}**")

# =========================================================
# KPIs
# =========================================================
row1 = [
    kpi_card_html("Tiempo de Operación (h)", fmt_num(to_base)),
    kpi_card_html("Downtime Fallas (h)", fmt_num(dt_base)),
    kpi_card_html("Fallas", f"{n_base:,}"),
]
render_kpi_row(row1, big=False)

row2 = [
    kpi_card_html("MTTR (h/falla)", fmt_num(mttr_hr), color=mttr_color_3(mttr_hr),
                  hint="Azul <1.2 | Verde 1.2–2.5 | Rojo >2.5"),
    kpi_card_html("MTBF (h/falla)", fmt_num(mtbf_hr), color=mtbf_color(mtbf_hr),
                  hint="Rojo <100 | Verde 100–500 | Azul >500"),
    kpi_card_html("Disponibilidad", fmt_pct(disp), color=disp_color(disp),
                  hint="Rojo <90% | Verde 90–95% | Azul >95%"),
]
render_kpi_row(row2, big=True)

st.divider()

# =========================================================
# EVOLUCIÓN POR MES-AÑO (MTTR/MTBF/DISP/FALLAS/TO/DT)
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
    base["MES"] = base["FECHA"].dt.to_period("M").astype(str)
    ids = set(base["ID_TURNO"].astype(str).tolist())

    h = horometros[horometros["ID_TURNO"].astype(str).isin(ids)].copy()
    e = eventos[eventos["ID_TURNO"].astype(str).isin(ids)].copy()

    e = e[e["CATEGORIA_EVENTO"].astype(str).str.upper() == "FALLA"].copy()
    e["DT_HR"] = pd.to_numeric(e["DT_MIN"], errors="coerce") / 60.0

    turn_mes = base[["ID_TURNO", "MES", "ID_TRACTOR", "ID_IMPLEMENTO"]].copy()
    turn_mes["ID_TURNO"] = turn_mes["ID_TURNO"].astype(str)

    # TO por mes según vista
    h2 = h.merge(turn_mes[["ID_TURNO", "MES"]], on="ID_TURNO", how="left")
    if vista_disp == "Tractor":
        h2 = h2[h2["TIPO_EQUIPO"].astype(str).str.upper() == "TRACTOR"].copy()
    elif vista_disp == "Implemento":
        h2 = h2[h2["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()
    else:
        # Sistema (TRC+IMP): por tu lógica original, TO base = TO del implemento
        h2 = h2[h2["TIPO_EQUIPO"].astype(str).str.upper() == "IMPLEMENTO"].copy()

    to_mes = h2.groupby("MES", dropna=True)["TO_HORO"].sum().reset_index(name="TO_HR")

    # DT/Fallas por mes según vista
    e2 = e.merge(turn_mes, on="ID_TURNO", how="left")

    if vista_disp == "Tractor":
        trcs = base["ID_TRACTOR"].dropna().astype(str).unique().tolist()
        e2 = e2[e2["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    elif vista_disp == "Implemento":
        imps = base["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
        e2 = e2[e2["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]
    else:
        # Sistema (TRC+IMP): deja ambos (tractor + implemento)
        pass

    dt_mes = e2.groupby("MES", dropna=True).agg(
        DT_HR=("DT_HR", "sum"),
        FALLAS=("DT_HR", "size")
    ).reset_index()

    evo = to_mes.merge(dt_mes, on="MES", how="left").fillna({"DT_HR": 0.0, "FALLAS": 0})
    evo["MTTR_HR"] = np.where(evo["FALLAS"] > 0, evo["DT_HR"] / evo["FALLAS"], np.nan)
    evo["MTBF_HR"] = np.where(evo["FALLAS"] > 0, evo["TO_HR"] / evo["FALLAS"], np.nan)
    evo["DISP"] = np.where((evo["TO_HR"] + evo["DT_HR"]) > 0, evo["TO_HR"] / (evo["TO_HR"] + evo["DT_HR"]), np.nan)
    evo = evo.sort_values("MES", ascending=True)

    # ======= FILA 1 (MTTR / MTBF)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig1 = px.bar(evo, x="MES", y="MTTR_HR", title="MTTR (h/falla) por mes")
        fig1.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig1, use_container_width=True)
    with r1c2:
        fig2 = px.bar(evo, x="MES", y="MTBF_HR", title="MTBF (h/falla) por mes")
        fig2.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ======= FILA 2 (DISP / FALLAS)
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig3 = px.bar(evo, x="MES", y="DISP", title="Disponibilidad (TO/(TO+DT)) por mes")
        fig3.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20), yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)
    with r2c2:
        fig4 = px.bar(evo, x="MES", y="FALLAS", title="Cantidad de fallas por mes")
        fig4.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig4, use_container_width=True)

    # ======= FILA 3 (TO / DOWN TIME)
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig5 = px.bar(evo, x="MES", y="TO_HR", title="Tiempo de Operación (TO) por mes (h)")
        fig5.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig5, use_container_width=True)
    with r3c2:
        fig6 = px.bar(evo, x="MES", y="DT_HR", title="Down Time por mes (h)")
        fig6.update_layout(title_x=0.5, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig6, use_container_width=True)

# =========================================================
# TOP 10 TÉCNICO (SUBUNIDAD / COMPONENTE / PARTE / VERBO / CAUSA)
# (YA ES SENSIBLE A RANGO DE FECHAS porque fd_sel viene filtrado por ids_turno)
# =========================================================
st.subheader("Top 10 Técnico (Subunidad / Componente / Parte / Verbo / Causa) por Tiempo de Falla (h)")

if fd_sel is None:
    st.info("No se encontró **FALLAS_DETALLE_NORMALIZADO.csv** dentro del ZIP (data_normalizada/).")
else:
    fd_sel = norm_cols(fd_sel)

    # ---- Asegurar ID_TURNO (por si el CSV trae 'ID' y no 'ID_TURNO') ----
    if "ID_TURNO" not in fd_sel.columns and "ID" in fd_sel.columns:
        fd_sel["ID_TURNO"] = safe_norm_str_series(fd_sel["ID"])

    # ---- Asegurar DOWNTIME_HR ----
    if "DOWNTIME_HR" not in fd_sel.columns:
        tf = find_first_col(fd_sel, ["T_FALLA", "TFALLA", "TIEMPO_FALLA", "TIEMPO_DE_FALLA", "DOWNTIME", "DT_HR"])
        if tf is not None:
            fd_sel["DOWNTIME_HR"] = _coerce_downtime_to_hr(fd_sel[tf])
        else:
            fd_sel["DOWNTIME_HR"] = np.nan

    # ---- Aliases por si vienen con otros nombres ----
    if "SUBUNIDAD" not in fd_sel.columns and "SUB_UNIDAD" in fd_sel.columns:
        fd_sel["SUBUNIDAD"] = fd_sel["SUB_UNIDAD"]
    if "PARTE" not in fd_sel.columns and "PIEZA" in fd_sel.columns:
        fd_sel["PARTE"] = fd_sel["PIEZA"]
    if "CAUSA_FALLA" not in fd_sel.columns and "CAUSA" in fd_sel.columns:
        fd_sel["CAUSA_FALLA"] = fd_sel["CAUSA"]

    # Detectar columna de equipo
    equipo_col = find_first_col(fd_sel, ["EQUIPO", "ID_EQUIPO", "ID_EQUIPO_AFECTADO"])
    if equipo_col is None:
        st.warning("FALLAS_DETALLE_NORMALIZADO no tiene columna EQUIPO/ID_EQUIPO. Se mostrará sin filtro por equipo.")
        fd_sel["__EQUIPO__"] = None
        equipo_col = "__EQUIPO__"

    # Asegurar TIPO_EQUIPO (si no viene, inferir por prefijo del equipo)
    if "TIPO_EQUIPO" not in fd_sel.columns or fd_sel["TIPO_EQUIPO"].isna().all():
        fd_sel["TIPO_EQUIPO"] = fd_sel[equipo_col].apply(infer_tipo_equipo_from_code)
    else:
        fd_sel["TIPO_EQUIPO"] = fd_sel["TIPO_EQUIPO"].astype(str).str.upper().str.strip()
        mask_bad = fd_sel["TIPO_EQUIPO"].isin(["", "NAN", "NONE"])
        if mask_bad.any():
            fd_sel.loc[mask_bad, "TIPO_EQUIPO"] = fd_sel.loc[mask_bad, equipo_col].apply(infer_tipo_equipo_from_code)

    # ---- FILTRO FINO por selección TRC/IMP + vista ----
    fd_view = fd_sel.copy()

    # 1) Si el usuario seleccionó tractor/implemento específico, filtrar por esos códigos
    allowed_equipos = []
    if trc_sel != "(Todos)":
        allowed_equipos.append(str(trc_sel).upper())
    if imp_sel != "(Todos)":
        allowed_equipos.append(str(imp_sel).upper())

    if allowed_equipos and equipo_col != "__EQUIPO__":
        fd_view = fd_view[fd_view[equipo_col].astype(str).str.upper().isin(allowed_equipos)].copy()

    # 2) Luego aplicar vista
    if vista_disp == "Tractor":
        fd_view = fd_view[fd_view["TIPO_EQUIPO"] == "TRACTOR"].copy()
    elif vista_disp == "Implemento":
        fd_view = fd_view[fd_view["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

    # Limpiar downtime
    fd_view["DOWNTIME_HR"] = pd.to_numeric(fd_view["DOWNTIME_HR"], errors="coerce").fillna(0.0)

    if fd_view.empty:
        st.info("Con los filtros actuales, no hay registros en FALLAS_DETALLE para construir el Top 10 técnico.")
    else:
        def top10_bar(df, col, title):
            if col not in df.columns:
                st.info(f"No existe la columna **{col}** en FALLAS_DETALLE_NORMALIZADO.")
                return
            g = (
                df.groupby(col, dropna=True)["DOWNTIME_HR"].sum()
                .reset_index()
                .sort_values("DOWNTIME_HR", ascending=False)
                .head(10)
            )
            g[col] = g[col].astype(str)
            fig = px.bar(g, x="DOWNTIME_HR", y=col, orientation="h")
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Tiempo de Falla (h)",
                yaxis_title="",
            )
            st.markdown(center_title(title), unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

        # 3 gráficos arriba
        c1, c2, c3 = st.columns(3)
        with c1:
            top10_bar(fd_view, "SUBUNIDAD", "Top 10 SUBUNIDAD")
        with c2:
            top10_bar(fd_view, "COMPONENTE", "Top 10 COMPONENTE")
        with c3:
            top10_bar(fd_view, "PARTE", "Top 10 PARTE")

        # 2 gráficos abajo
        c4, c5 = st.columns(2)
        with c4:
            top10_bar(fd_view, "VERBO_TECNICO", "Top 10 VERBO TÉCNICO")
        with c5:
            top10_bar(fd_view, "CAUSA_FALLA", "Top 10 CAUSA DE FALLA")

st.divider()

# =========================================================
# DESCARGAS
# =========================================================
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
