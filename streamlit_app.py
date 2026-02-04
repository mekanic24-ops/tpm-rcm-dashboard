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

    # Tipos base
    turnos["FECHA"] = pd.to_datetime(turnos["FECHA"], errors="coerce")
    horometros["TO_HORO"] = pd.to_numeric(horometros["TO_HORO"], errors="coerce")
    eventos["DT_MIN"] = pd.to_numeric(eventos["DT_MIN"], errors="coerce")

    def norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).str.replace(".0", "", regex=False).str.strip()

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
    # Asegurar columna nombre (por si viene como PROCESO)
    if "NOMBRE_PROCESO" not in cat_proceso.columns and "PROCESO" in cat_proceso.columns:
        cat_proceso = cat_proceso.rename(columns={"PROCESO": "NOMBRE_PROCESO"})
    cat_proceso["NOMBRE_PROCESO"] = cat_proceso["NOMBRE_PROCESO"].astype(str).str.strip()

    operadores["ID_OPERADOR"] = norm_str(operadores["ID_OPERADOR"])
    operadores["NOMBRE_OPERADOR"] = operadores["NOMBRE_OPERADOR"].astype(str).str.strip()

    lotes["ID_LOTE"] = norm_str(lotes["ID_LOTE"])
    lotes["CULTIVO"] = lotes["CULTIVO"].astype(str)

    return {
        "turnos": turnos,
        "horometros": horometros,
        "eventos": eventos,
        "operadores": operadores,
        "lotes": lotes,
        "fallas_cat": fallas_cat,
        "cat_proceso": cat_proceso,
    }

def normalize_cultivo(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    s = s.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
    if s in ["PALTA", "PALTO"]:
        return "PALTO"
    if s in ["ARANDANO", "ARANDANO "]:
        return "ARANDANO"
    return s

def build_enriched_turnos(turnos, operadores, lotes):
    t = turnos.copy()
    op_map = dict(zip(operadores["ID_OPERADOR"], operadores["NOMBRE_OPERADOR"]))
    t["OPERADOR_NOMBRE"] = t["ID_OPERADOR"].map(op_map)

    lote_map = dict(zip(lotes["ID_LOTE"], lotes["CULTIVO"]))
    t["CULTIVO"] = t["ID_LOTE"].map(lote_map)
    t["CULTIVO"] = t["CULTIVO"].apply(normalize_cultivo)
    return t

def mttr_color_3(v):
    """<1.2 azul | 1.2-2.5 verde | >2.5 rojo"""
    if v is None or pd.isna(v):
        return None
    if v < 1.2:
        return "#1f77b4"  # azul
    if v <= 2.5:
        return "#2ca02c"  # verde
    return "#d62728"      # rojo

def mtbf_color(v):
    if v is None or pd.isna(v):
        return None
    if v < 100:
        return "#d62728"  # rojo
    if 100 <= v <= 500:
        return "#2ca02c"  # verde
    return "#1f77b4"      # azul

def disp_color(d):
    """d viene como ratio 0-1"""
    if d is None or pd.isna(d):
        return None
    p = d * 100
    if p < 90:
        return "#d62728"   # rojo
    if p <= 95:
        return "#2ca02c"   # verde
    return "#1f77b4"       # azul

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
            display:flex;
            gap:28px;
            justify-content:center;
            align-items:flex-start;
            flex-wrap:wrap;
            margin: 6px 0 0 0;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          }}
          .kpi{{
            width: 320px;
            min-height: 150px;
            padding: 8px 10px;
            text-align:center;
            box-sizing:border-box;
          }}
          .kpi-title{{
            font-size: 16px;
            font-weight: 800;
            color: #111;
            margin: 0 0 10px 0;
            line-height: 1.2;
          }}
          .kpi-value{{
            font-size: 56px;
            font-weight: 400;
            line-height: 1.05;
            margin: 0 0 8px 0;
          }}
          .kpi-hint{{
            font-size: 12px;
            color: #6b7280;
            line-height: 1.2;
            min-height: 16px;
          }}
          .kpi-row.big .kpi{{
            width: 340px;
            min-height: 165px;
          }}
          .kpi-row.big .kpi-value{{
            font-size: 60px;
          }}
        </style>
      </head>
      <body style="margin:0;padding:0;">
        <div class="{row_class}">
          {''.join(cards_html)}
        </div>
      </body>
    </html>
    """
    height = 185 if big else 175
    components.html(html, height=height)

def safe_radio_index(options: List[str], current: str, default: str) -> int:
    """Evita resets raros cuando cambian opciones."""
    if current in options:
        return options.index(current)
    if default in options:
        return options.index(default)
    return 0

def compute_proceso_order_by_to_imp(
    turnos_base: pd.DataFrame,
    horometros: pd.DataFrame,
    proc_map: Dict[str, str],
) -> List[str]:
    """
    Retorna lista de nombres de proceso ordenados por TO del implemento (desc),
    usando la base ya filtrada (fecha/cultivo/tractor/implemento/turno).
    """
    if turnos_base.empty:
        return []

    ids = set(turnos_base["ID_TURNO"].astype(str).tolist())
    h = horometros[horometros["ID_TURNO"].astype(str).isin(ids)].copy()
    h = h[h["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

    if h.empty:
        # si no hay TO de implemento, orden “natural”
        procs = sorted(turnos_base["ID_PROCESO"].dropna().astype(str).unique().tolist())
        return [proc_map.get(pid, f"PROCESO {pid}") for pid in procs]

    tmap = turnos_base[["ID_TURNO", "ID_PROCESO"]].copy()
    tmap["ID_TURNO"] = tmap["ID_TURNO"].astype(str)
    h2 = h.merge(tmap, on="ID_TURNO", how="left")

    g = (
        h2.groupby("ID_PROCESO", dropna=True)["TO_HORO"]
        .sum()
        .reset_index()
        .sort_values("TO_HORO", ascending=False)
    )
    g["ID_PROCESO"] = g["ID_PROCESO"].astype(str)

    ordered_names = []
    for pid in g["ID_PROCESO"].tolist():
        ordered_names.append(proc_map.get(pid, f"PROCESO {pid}"))

    # añade los que no salieron (por TO=0)
    all_pids = turnos_base["ID_PROCESO"].dropna().astype(str).unique().tolist()
    for pid in all_pids:
        name = proc_map.get(pid, f"PROCESO {pid}")
        if name not in ordered_names:
            ordered_names.append(name)

    return ordered_names

# =========================================================
# LOAD
# =========================================================
tables = load_tables()
turnos = build_enriched_turnos(tables["turnos"], tables["operadores"], tables["lotes"])
horometros = tables["horometros"]
eventos = tables["eventos"]
cat_proceso = tables["cat_proceso"]

proc_map = dict(zip(cat_proceso["ID_PROCESO"].astype(str), cat_proceso["NOMBRE_PROCESO"].astype(str)))
proc_name_to_id = {}
for pid, name in proc_map.items():
    # si hay duplicados, nos quedamos con el primero
    proc_name_to_id.setdefault(name, pid)

# =========================================================
# SESSION STATE (para que NO se resetee)
# =========================================================
if "f_proc_name" not in st.session_state:
    st.session_state.f_proc_name = "(Todos)"
if "f_cultivo" not in st.session_state:
    st.session_state.f_cultivo = "(Todos)"
if "f_turno" not in st.session_state:
    st.session_state.f_turno = "(Todos)"
if "f_vista" not in st.session_state:
    st.session_state.f_vista = "Sistema (TRC+IMP)"

# =========================================================
# SIDEBAR FILTERS (INTERACTIVO, SIN BOTÓN)
# =========================================================
st.sidebar.header("Filtros")

# Fecha
min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
    key="f_date_range",
)

# Base por fechas
df_base = turnos.copy()
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1])
    df_base = df_base[(df_base["FECHA"] >= d1) & (df_base["FECHA"] <= d2)]

# Cultivo en BOTONES
cult_options = ["(Todos)", "Palto", "Arandano"]
cult_index = safe_radio_index(cult_options, st.session_state.f_cultivo, "(Todos)")
cult_sel_label = st.sidebar.radio("Cultivo", cult_options, index=cult_index, key="f_cultivo")
cult_sel = "(Todos)"
if cult_sel_label == "Palto":
    cult_sel = "PALTO"
elif cult_sel_label == "Arandano":
    cult_sel = "ARANDANO"

df_for_proc = df_base.copy()
if cult_sel != "(Todos)":
    df_for_proc = df_for_proc[df_for_proc["CULTIVO"] == cult_sel]

# Tractor
trc_opts = ["(Todos)"] + sorted(df_for_proc["ID_TRACTOR"].dropna().astype(str).unique().tolist())
trc_idx = safe_radio_index(trc_opts, st.session_state.get("f_trc", "(Todos)"), "(Todos)")
trc_sel = st.sidebar.selectbox("Tractor", trc_opts, index=trc_idx, key="f_trc")
df_for_proc2 = df_for_proc.copy()
if trc_sel != "(Todos)":
    df_for_proc2 = df_for_proc2[df_for_proc2["ID_TRACTOR"].astype(str) == str(trc_sel)]

# Implemento
imp_opts = ["(Todos)"] + sorted(df_for_proc2["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
imp_idx = safe_radio_index(imp_opts, st.session_state.get("f_imp", "(Todos)"), "(Todos)")
imp_sel = st.sidebar.selectbox("Implemento", imp_opts, index=imp_idx, key="f_imp")
df_for_proc3 = df_for_proc2.copy()
if imp_sel != "(Todos)":
    df_for_proc3 = df_for_proc3[df_for_proc3["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]

# Turno en BOTONES: (Todos) / Día / Noche
turno_options = ["(Todos)", "Día", "Noche"]
turno_index = safe_radio_index(turno_options, st.session_state.f_turno, "(Todos)")
turno_label = st.sidebar.radio("Turno", turno_options, index=turno_index, key="f_turno")
turno_sel = "(Todos)"
if turno_label == "Día":
    turno_sel = "D"
elif turno_label == "Noche":
    turno_sel = "N"

df_for_proc4 = df_for_proc3.copy()
if turno_sel != "(Todos)":
    df_for_proc4 = df_for_proc4[df_for_proc4["TURNO"].astype(str) == str(turno_sel)]

# Proceso (radio) ordenado por TO del implemento (desc), mostrando SOLO nombres
proc_names_ordered = compute_proceso_order_by_to_imp(df_for_proc4, horometros, proc_map)
proc_options = ["(Todos)"] + proc_names_ordered

# si tu selección previa ya no está, mantenemos "(Todos)"
proc_index = safe_radio_index(proc_options, st.session_state.f_proc_name, "(Todos)")
proc_name_sel = st.sidebar.radio(
    "Proceso (ordenado por TO implemento)",
    proc_options,
    index=proc_index,
    key="f_proc_name",
)

id_proceso_sel = None
df_f = df_for_proc4.copy()
if proc_name_sel != "(Todos)":
    id_proceso_sel = proc_name_to_id.get(proc_name_sel)
    if id_proceso_sel is not None:
        df_f = df_f[df_f["ID_PROCESO"].astype(str) == str(id_proceso_sel)]
    else:
        st.sidebar.warning("No encuentro el ID del proceso seleccionado en CAT_PROCESO.")

# Vista (KPI base)
vista_options = ["Sistema (TRC+IMP)", "Tractor", "Implemento"]
vista_index = safe_radio_index(vista_options, st.session_state.f_vista, "Sistema (TRC+IMP)")
vista_disp = st.sidebar.radio(
    "Disponibilidad / MTBF / MTTR basados en:",
    vista_options,
    index=vista_index,
    key="f_vista",
)

# =========================================================
# SELECCIÓN FINAL
# =========================================================
turnos_sel = df_f.copy()
ids_turno = set(turnos_sel["ID_TURNO"].astype(str).tolist())

horo_sel = horometros[horometros["ID_TURNO"].astype(str).isin(ids_turno)].copy()
ev_sel = eventos[eventos["ID_TURNO"].astype(str).isin(ids_turno)].copy()

# =========================================================
# KPI GLOBAL
# =========================================================
to_trac = float(horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "TRACTOR", "TO_HORO"].sum())
to_imp = float(horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO", "TO_HORO"].sum())

ev_fallas = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_fallas.empty:
    ev_fallas["DT_HR"] = pd.to_numeric(ev_fallas["DT_MIN"], errors="coerce") / 60.0
else:
    ev_fallas["DT_HR"] = pd.Series(dtype=float)

def filter_fallas_for_vista(ev_fallas_df: pd.DataFrame, vista: str, tsel: pd.DataFrame) -> pd.DataFrame:
    """Devuelve eventos de falla aplicables según vista."""
    if ev_fallas_df.empty:
        return ev_fallas_df

    if vista == "Sistema (TRC+IMP)":
        return ev_fallas_df  # todas las fallas del sistema (tractor y/o implemento)

    if vista == "Tractor":
        trcs = tsel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
        return ev_fallas_df[ev_fallas_df["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]

    # Implemento
    imps = tsel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
    return ev_fallas_df[ev_fallas_df["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]

# Base TO para KPIs (según lo definido)
if vista_disp == "Tractor":
    to_base = to_trac
else:
    # Sistema y Implemento -> TO neto del implemento
    to_base = to_imp

ev_base = filter_fallas_for_vista(ev_fallas, vista_disp, turnos_sel)

dt_base = float(ev_base["DT_HR"].sum()) if not ev_base.empty else 0.0
n_base = int(len(ev_base)) if not ev_base.empty else 0

mttr_hr = (dt_base / n_base) if n_base > 0 else np.nan
mtbf_hr = (to_base / n_base) if n_base > 0 else np.nan

# NUEVA FORMULA DE DISPONIBILIDAD
disp = (to_base / (to_base + dt_base)) if (to_base is not None and (to_base + dt_base) > 0) else np.nan

# =========================================================
# TITULOS
# =========================================================
st.title("DESEMPEÑO OPERACIONAL DE LA FLOTA")
st.caption("INDICADORES DE CONFIABILIDAD - MANTENIBILIDAD - DISPONIBILIDAD")
st.caption(f"Vista actual de KPIs: **{vista_disp}**")

# =========================================================
# KPIs (6 cards: 3 arriba + 3 abajo)
# =========================================================
row1 = [
    kpi_card_html("Tiempo de Operación (h)", fmt_num(to_imp)),          # pedido: renombrar
    kpi_card_html("Downtime Fallas (h)", fmt_num(dt_base)),
    kpi_card_html("Fallas", f"{n_base:,}"),
]
render_kpi_row(row1, big=False)

row2 = [
    kpi_card_html(
        "MTTR (h/falla)",
        fmt_num(mttr_hr),
        color=mttr_color_3(mttr_hr),
        hint="Azul <1.2 | Verde 1.2–2.5 | Rojo >2.5",
    ),
    kpi_card_html(
        "MTBF (h/falla)",
        fmt_num(mtbf_hr),
        color=mtbf_color(mtbf_hr),
        hint="Rojo <100 | Verde 100–500 | Azul >500",
    ),
    kpi_card_html(
        "Disponibilidad",
        fmt_pct(disp),
        color=disp_color(disp),
        hint="Rojo <90% | Verde 90–95% | Azul >95%",
    ),
]
render_kpi_row(row2, big=True)

st.divider()

# =========================================================
# TOP 10 DE EQUIPOS: TO, DOWNTIME, FALLAS
# (ahora: EQUIPOS en general, sensible a la vista)
# =========================================================
st.subheader("Top 10 de equipos: TO, Downtime y Fallas")

# TO por equipo según vista (Sistema -> Implemento, Tractor -> Tractor, Implemento -> Implemento)
if vista_disp == "Tractor":
    h_rank = horo_sel[horo_sel["TIPO_EQUIPO"] == "TRACTOR"].copy()
elif vista_disp == "Implemento":
    h_rank = horo_sel[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
else:
    # Sistema: usamos implemento como “representante” del sistema para TO neto
    h_rank = horo_sel[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

to_rank = (
    h_rank.groupby("ID_EQUIPO", dropna=True)["TO_HORO"]
    .sum()
    .reset_index()
    .rename(columns={"ID_EQUIPO": "EQUIPO", "TO_HORO": "TO_HR"})
)

# Downtime + fallas por equipo (según vista)
ev_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_rank.empty:
    ev_rank["DT_HR"] = pd.to_numeric(ev_rank["DT_MIN"], errors="coerce") / 60.0
else:
    ev_rank["DT_HR"] = pd.Series(dtype=float)

ev_rank = filter_fallas_for_vista(ev_rank, vista_disp, turnos_sel)

dt_rank = (
    ev_rank.groupby("ID_EQUIPO_AFECTADO", dropna=True)
    .agg(DT_HR=("DT_HR", "sum"), FALLAS=("DT_HR", "size"))
    .reset_index()
    .rename(columns={"ID_EQUIPO_AFECTADO": "EQUIPO"})
)

rank_df = to_rank.merge(dt_rank, on="EQUIPO", how="outer").fillna({"TO_HR": 0.0, "DT_HR": 0.0, "FALLAS": 0})

title_size = 18
caption_style = {"font_size": title_size, "x": 0.5, "xanchor": "center"}

c1, c2, c3 = st.columns(3)

with c1:
    d = rank_df.sort_values("TO_HR", ascending=False).head(10)
    fig = px.bar(d, x="TO_HR", y="EQUIPO", orientation="h", title="Top 10 por Tiempo de Operación (h)")
    fig.update_layout(title=caption_style, xaxis_title="TO (h)", yaxis_title="Equipo", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    d = rank_df.sort_values("DT_HR", ascending=False).head(10)
    fig = px.bar(d, x="DT_HR", y="EQUIPO", orientation="h", title="Top 10 por Downtime (h)")
    fig.update_layout(title=caption_style, xaxis_title="Downtime (h)", yaxis_title="Equipo", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c3:
    d = rank_df.sort_values("FALLAS", ascending=False).head(10)
    fig = px.bar(d, x="FALLAS", y="EQUIPO", orientation="h", title="Top 10 por Cantidad de Fallas")
    fig.update_layout(title=caption_style, xaxis_title="Fallas (n)", yaxis_title="Equipo", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================================================
# EVOLUCIÓN POR MES-AÑO (MTTR y MTBF)
# - sensible a vista: Tractor / Implemento / Sistema
# - sensible a filtros (tractor/implemento/cultivo/turno/proceso)
# =========================================================
st.subheader("Evolución por mes-año (MTTR y MTBF)")

if id_proceso_sel is None:
    st.info("Selecciona un **Proceso** en el filtro lateral para ver su evolución mensual.")
else:
    base = turnos.copy()

    # Rango fechas
    if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
        d1 = pd.to_datetime(date_range[0])
        d2 = pd.to_datetime(date_range[1])
        base = base[(base["FECHA"] >= d1) & (base["FECHA"] <= d2)]

    # Filtros (los mismos de sidebar)
    base = base[base["ID_PROCESO"].astype(str) == str(id_proceso_sel)]
    if cult_sel != "(Todos)":
        base = base[base["CULTIVO"] == cult_sel]
    if trc_sel != "(Todos)":
        base = base[base["ID_TRACTOR"].astype(str) == str(trc_sel)]
    if imp_sel != "(Todos)":
        base = base[base["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]
    if turno_sel != "(Todos)":
        base = base[base["TURNO"].astype(str) == str(turno_sel)]

    st.caption(f"Proceso seleccionado: **{proc_map.get(str(id_proceso_sel), proc_name_sel)}**")

    if base.empty:
        st.info("Con los filtros actuales, no hay turnos para este proceso.")
    else:
        base = base.copy()
        base["MES"] = base["FECHA"].dt.to_period("M").astype(str)  # YYYY-MM
        ids = set(base["ID_TURNO"].astype(str).tolist())

        h = horometros[horometros["ID_TURNO"].astype(str).isin(ids)].copy()
        e = eventos[eventos["ID_TURNO"].astype(str).isin(ids)].copy()

        # fallas + DT
        e = e[e["CATEGORIA_EVENTO"] == "FALLA"].copy()
        e["DT_HR"] = pd.to_numeric(e["DT_MIN"], errors="coerce") / 60.0

        # aplicar filtro de fallas según vista
        e = filter_fallas_for_vista(e, vista_disp, base)

        # Map turno -> mes
        turn_mes = base[["ID_TURNO", "MES"]].copy()
        turn_mes["ID_TURNO"] = turn_mes["ID_TURNO"].astype(str)

        # TO por mes según vista
        h2 = h.merge(turn_mes, on="ID_TURNO", how="left")
        if vista_disp == "Tractor":
            h2 = h2[h2["TIPO_EQUIPO"] == "TRACTOR"].copy()
        else:
            # Implemento y Sistema -> TO neto del implemento
            h2 = h2[h2["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

        to_mes = h2.groupby("MES", dropna=True)["TO_HORO"].sum().reset_index(name="TO_HR")

        # Downtime y fallas por mes
        e2 = e.merge(turn_mes, on="ID_TURNO", how="left")
        dt_mes = (
            e2.groupby("MES", dropna=True)
            .agg(DT_HR=("DT_HR", "sum"), FALLAS=("DT_HR", "size"))
            .reset_index()
        )

        evo = to_mes.merge(dt_mes, on="MES", how="left").fillna({"DT_HR": 0.0, "FALLAS": 0})
        evo["MTTR_HR"] = np.where(evo["FALLAS"] > 0, evo["DT_HR"] / evo["FALLAS"], np.nan)
        evo["MTBF_HR"] = np.where(evo["FALLAS"] > 0, evo["TO_HR"] / evo["FALLAS"], np.nan)
        evo = evo.sort_values("MES", ascending=True)

        cA, cB = st.columns(2)
        with cA:
            fig1 = px.bar(evo, x="MES", y="MTTR_HR", title="MTTR (h/falla) por mes")
            fig1.update_layout(
                title=caption_style,
                xaxis_title="Mes (YYYY-MM)",
                yaxis_title="MTTR (h/falla)",
                margin=dict(l=20, r=20, t=55, b=20),
            )
            st.plotly_chart(fig1, use_container_width=True)

        with cB:
            fig2 = px.bar(evo, x="MES", y="MTBF_HR", title="MTBF (h/falla) por mes")
            fig2.update_layout(
                title=caption_style,
                xaxis_title="Mes (YYYY-MM)",
                yaxis_title="MTBF (h/falla)",
                margin=dict(l=20, r=20, t=55, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================================================
# TOP 10 POR EQUIPO (MTTR / MTBF / DISP) - se mantiene
# =========================================================
st.subheader("Top 10 por Equipo (Confiabilidad)")

# Construcción de top_df sensible a vista
to_equipo = horo_sel.groupby(["TIPO_EQUIPO", "ID_EQUIPO"], dropna=True)["TO_HORO"].sum().reset_index()
to_equipo["TO_HR"] = to_equipo["TO_HORO"]

ev_fallas_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_fallas_rank.empty:
    ev_fallas_rank["DT_HR"] = pd.to_numeric(ev_fallas_rank["DT_MIN"], errors="coerce") / 60.0
else:
    ev_fallas_rank["DT_HR"] = pd.Series(dtype=float)

if vista_disp == "Sistema (TRC+IMP)":
    # sistema: ranking por “sistema/implemento” (TO neto) pero DT viene por implemento (asignación por turno)
    imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids)][["ID_EQUIPO", "TO_HR"]]

    ev_sys = ev_fallas_rank.merge(turnos_sel[["ID_TURNO", "ID_IMPLEMENTO"]], on="ID_TURNO", how="left")
    ev_sys["ID_IMPLEMENTO"] = ev_sys["ID_IMPLEMENTO"].astype(str)
    dt_sys_imp = (
        ev_sys.groupby("ID_IMPLEMENTO", dropna=True)
        .agg(DT_FALLA_HR=("DT_HR", "sum"), FALLAS=("DT_HR", "size"))
        .reset_index()
        .rename(columns={"ID_IMPLEMENTO": "ID_EQUIPO"})
    )

    top_df = to_imp_df.merge(dt_sys_imp, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})

elif vista_disp == "Tractor":
    trc_ids = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()

    to_trc_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "TRACTOR"].copy()
    to_trc_df["ID_EQUIPO"] = to_trc_df["ID_EQUIPO"].astype(str)
    to_trc_df = to_trc_df[to_trc_df["ID_EQUIPO"].isin(trc_ids)][["ID_EQUIPO", "TO_HR"]]

    falla_trc = (
        ev_fallas_rank.groupby("ID_EQUIPO_AFECTADO", dropna=True)
        .agg(DT_FALLA_HR=("DT_HR", "sum"), FALLAS=("DT_HR", "size"))
        .reset_index()
        .rename(columns={"ID_EQUIPO_AFECTADO": "ID_EQUIPO"})
    )
    falla_trc["ID_EQUIPO"] = falla_trc["ID_EQUIPO"].astype(str)
    falla_trc = falla_trc[falla_trc["ID_EQUIPO"].isin(trc_ids)]

    top_df = to_trc_df.merge(falla_trc, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})

else:
    imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids)][["ID_EQUIPO", "TO_HR"]]

    falla_imp = (
        ev_fallas_rank.groupby("ID_EQUIPO_AFECTADO", dropna=True)
        .agg(DT_FALLA_HR=("DT_HR", "sum"), FALLAS=("DT_HR", "size"))
        .reset_index()
        .rename(columns={"ID_EQUIPO_AFECTADO": "ID_EQUIPO"})
    )
    falla_imp["ID_EQUIPO"] = falla_imp["ID_EQUIPO"].astype(str)
    falla_imp = falla_imp[falla_imp["ID_EQUIPO"].isin(imp_ids)]

    top_df = to_imp_df.merge(falla_imp, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})

top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
top_df["DISP"] = np.where(
    (top_df["TO_HR"] + top_df["DT_FALLA_HR"]) > 0,
    top_df["TO_HR"] / (top_df["TO_HR"] + top_df["DT_FALLA_HR"]),
    np.nan
)

colA, colB, colC = st.columns(3)
with colA:
    st.caption("MTTR alto (peor)")
    d = top_df.dropna(subset=["MTTR_HR"]).sort_values("MTTR_HR", ascending=False).head(10)
    fig = px.bar(d, x="MTTR_HR", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with colB:
    st.caption("MTBF bajo (peor)")
    d = top_df.dropna(subset=["MTBF_HR"]).sort_values("MTBF_HR", ascending=True).head(10)
    fig = px.bar(d, x="MTBF_HR", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with colC:
    st.caption("Disponibilidad baja (peor)")
    d = top_df.dropna(subset=["DISP"]).sort_values("DISP", ascending=True).head(10)
    fig = px.bar(d, x="DISP", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

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
