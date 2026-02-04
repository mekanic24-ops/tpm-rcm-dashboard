import zipfile
from pathlib import Path
from typing import Optional, List

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

    # Siempre reservamos espacio para que todas las tarjetas queden alineadas
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

    # CSS DENTRO del componente (evita que se vea como texto y evita “escape”)
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
            width: 300px;
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
            font-size: 52px;
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
            width: 320px;
            min-height: 165px;
          }}
          .kpi-row.big .kpi-value{{
            font-size: 56px;
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

# =========================================================
# LOAD
# =========================================================
tables = load_tables()
turnos = build_enriched_turnos(tables["turnos"], tables["operadores"], tables["lotes"])
horometros = tables["horometros"]
eventos = tables["eventos"]
fallas_cat = tables["fallas_cat"]
cat_proceso = tables["cat_proceso"]

# =========================================================
# SIDEBAR FILTERS (INTERACTIVO)
# =========================================================
st.sidebar.header("Filtros")

min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
)

# Base por fechas
df_f = turnos.copy()
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["FECHA"] >= d1) & (df_f["FECHA"] <= d2)]

# Proceso (label bonito)
proc_map = dict(zip(cat_proceso["ID_PROCESO"].astype(str), cat_proceso["NOMBRE_PROCESO"].astype(str)))
all_proc_ids = sorted(df_f["ID_PROCESO"].dropna().astype(str).unique().tolist())
proc_labels = ["(Todos)"] + [f"{proc_map.get(pid, 'PROCESO')} [{pid}]" for pid in all_proc_ids]

proc_label_sel = st.sidebar.selectbox("Proceso", proc_labels, index=0)
id_proceso_sel = None
if proc_label_sel != "(Todos)":
    id_proceso_sel = proc_label_sel.split("[")[-1].replace("]", "").strip()
    df_f = df_f[df_f["ID_PROCESO"].astype(str) == id_proceso_sel]

# Cultivo (solo Palto / Arandano)
cult_map_label = {"PALTO": "Palto", "ARANDANO": "Arandano"}
cult_vals = sorted([v for v in df_f["CULTIVO"].dropna().unique().tolist() if v in cult_map_label])
cult_opts = ["(Todos)"] + [cult_map_label[v] for v in cult_vals]
cult_sel_label = st.sidebar.selectbox("Cultivo", cult_opts, index=0)

cult_sel = "(Todos)"
if cult_sel_label != "(Todos)":
    inv = {v: k for k, v in cult_map_label.items()}
    cult_sel = inv[cult_sel_label]
    df_f = df_f[df_f["CULTIVO"] == cult_sel]

# Tractor
trc_opts = ["(Todos)"] + sorted(df_f["ID_TRACTOR"].dropna().astype(str).unique().tolist())
trc_sel = st.sidebar.selectbox("Tractor", trc_opts, index=0)
if trc_sel != "(Todos)":
    df_f = df_f[df_f["ID_TRACTOR"].astype(str) == str(trc_sel)]

# Implemento
imp_opts = ["(Todos)"] + sorted(df_f["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
imp_sel = st.sidebar.selectbox("Implemento", imp_opts, index=0)
if imp_sel != "(Todos)":
    df_f = df_f[df_f["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]

# Operador
op_opts = ["(Todos)"] + sorted(df_f["OPERADOR_NOMBRE"].dropna().astype(str).unique().tolist())
op_sel = st.sidebar.selectbox("Operador", op_opts, index=0)
if op_sel != "(Todos)":
    df_f = df_f[df_f["OPERADOR_NOMBRE"].astype(str) == str(op_sel)]

# Turno
turno_opts = ["(Todos)"] + sorted(df_f["TURNO"].dropna().astype(str).unique().tolist())
turno_sel = st.sidebar.selectbox("Turno (D/N)", turno_opts, index=0)
if turno_sel != "(Todos)":
    df_f = df_f[df_f["TURNO"].astype(str) == str(turno_sel)]

vista_disp = st.sidebar.radio(
    "Disponibilidad / MTBF / MTTR basados en:",
    ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
    index=0,
)

# =========================================================
# SELECCIÓN
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

def fallas_de_equipo(cod_equipo: str):
    return ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str) == str(cod_equipo)]

if vista_disp == "Sistema (TRC+IMP)":
    # TO base: IMPLEMENTO (operación neta)
    to_base = to_imp
    dt_base = float(ev_fallas["DT_HR"].sum())
    n_base = int(len(ev_fallas))

elif vista_disp == "Tractor":
    to_base = to_trac
    if trc_sel != "(Todos)":
        ev_b = fallas_de_equipo(trc_sel)
    else:
        trcs = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    dt_base = float(ev_b["DT_HR"].sum()) if not ev_b.empty else 0.0
    n_base = int(len(ev_b))

else:  # Implemento
    to_base = to_imp
    if imp_sel != "(Todos)":
        ev_b = fallas_de_equipo(imp_sel)
    else:
        imps = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]
    dt_base = float(ev_b["DT_HR"].sum()) if not ev_b.empty else 0.0
    n_base = int(len(ev_b))

mttr_hr = (dt_base / n_base) if n_base > 0 else np.nan
mtbf_hr = (to_base / n_base) if n_base > 0 else np.nan
disp = ((to_base - dt_base) / to_base) if (to_base and to_base > 0) else np.nan

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
    kpi_card_html("TO Implemento (h)", fmt_num(to_imp)),
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
# EVOLUCIÓN POR MES-AÑO (según PROCESO elegido)
# =========================================================
st.subheader("Evolución por mes-año (MTTR y MTBF)")

if id_proceso_sel is None:
    st.info("Selecciona un **Proceso** en el filtro lateral para ver su evolución mensual.")
else:
    base = turnos.copy()

    # Rango fechas (mismo filtro)
    if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
        d1 = pd.to_datetime(date_range[0])
        d2 = pd.to_datetime(date_range[1])
        base = base[(base["FECHA"] >= d1) & (base["FECHA"] <= d2)]

    # aplicar mismos filtros seleccionados
    base = base[base["ID_PROCESO"].astype(str) == str(id_proceso_sel)]
    if cult_sel != "(Todos)":
        base = base[base["CULTIVO"] == cult_sel]
    if trc_sel != "(Todos)":
        base = base[base["ID_TRACTOR"].astype(str) == str(trc_sel)]
    if imp_sel != "(Todos)":
        base = base[base["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]
    if op_sel != "(Todos)":
        base = base[base["OPERADOR_NOMBRE"].astype(str) == str(op_sel)]
    if turno_sel != "(Todos)":
        base = base[base["TURNO"].astype(str) == str(turno_sel)]

    proc_name = proc_map.get(str(id_proceso_sel), f"Proceso {id_proceso_sel}")
    st.caption(f"Proceso seleccionado: **{proc_name}**")

    if base.empty:
        st.info("Con los filtros actuales, no hay turnos para este proceso.")
    else:
        base["MES"] = base["FECHA"].dt.to_period("M").astype(str)  # YYYY-MM
        ids = set(base["ID_TURNO"].astype(str).tolist())

        h = horometros[horometros["ID_TURNO"].astype(str).isin(ids)].copy()
        e = eventos[eventos["ID_TURNO"].astype(str).isin(ids)].copy()

        # fallas + DT
        e = e[e["CATEGORIA_EVENTO"] == "FALLA"].copy()
        e["DT_HR"] = pd.to_numeric(e["DT_MIN"], errors="coerce") / 60.0

        # Map turno -> mes
        turn_mes = base[["ID_TURNO", "MES"]].copy()
        turn_mes["ID_TURNO"] = turn_mes["ID_TURNO"].astype(str)

        # TO por mes según vista (Tractor vs Implemento)
        h2 = h.merge(turn_mes, on="ID_TURNO", how="left")
        if vista_disp == "Tractor":
            h2 = h2[h2["TIPO_EQUIPO"] == "TRACTOR"].copy()
        else:
            h2 = h2[h2["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

        to_mes = h2.groupby("MES", dropna=True)["TO_HORO"].sum().reset_index(name="TO_HR")

        # Downtime y fallas por mes
        e2 = e.merge(turn_mes, on="ID_TURNO", how="left")
        dt_mes = e2.groupby("MES", dropna=True).agg(
            DT_HR=("DT_HR", "sum"),
            FALLAS=("DT_HR", "size")
        ).reset_index()

        evo = to_mes.merge(dt_mes, on="MES", how="left").fillna({"DT_HR": 0.0, "FALLAS": 0})
        evo["MTTR_HR"] = np.where(evo["FALLAS"] > 0, evo["DT_HR"] / evo["FALLAS"], np.nan)
        evo["MTBF_HR"] = np.where(evo["FALLAS"] > 0, evo["TO_HR"] / evo["FALLAS"], np.nan)
        evo = evo.sort_values("MES", ascending=True)

        cA, cB = st.columns(2)
        with cA:
            fig1 = px.bar(evo, x="MES", y="MTTR_HR", title="MTTR (h/falla) por mes")
            fig1.update_layout(xaxis_title="Mes (YYYY-MM)", yaxis_title="MTTR (h/falla)", margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig1, use_container_width=True)

        with cB:
            fig2 = px.bar(evo, x="MES", y="MTBF_HR", title="MTBF (h/falla) por mes")
            fig2.update_layout(xaxis_title="Mes (YYYY-MM)", yaxis_title="MTBF (h/falla)", margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================================================
# TOP 10 POR EQUIPO
# =========================================================
st.subheader("Top 10 por Equipo")

to_equipo = horo_sel.groupby(["TIPO_EQUIPO", "ID_EQUIPO"], dropna=True)["TO_HORO"].sum().reset_index()
to_equipo["TO_HR"] = to_equipo["TO_HORO"]

ev_fallas_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_fallas_rank.empty:
    ev_fallas_rank["DT_HR"] = pd.to_numeric(ev_fallas_rank["DT_MIN"], errors="coerce") / 60.0
else:
    ev_fallas_rank["DT_HR"] = pd.Series(dtype=float)

falla_equipo = ev_fallas_rank.groupby("ID_EQUIPO_AFECTADO", dropna=True).agg(
    DT_FALLA_HR=("DT_HR", "sum"),
    FALLAS=("DT_HR", "size")
).reset_index().rename(columns={"ID_EQUIPO_AFECTADO": "ID_EQUIPO"})

def build_top_df_sistema():
    imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids)][["ID_EQUIPO", "TO_HR"]]

    ev_sys = ev_fallas_rank.merge(turnos_sel[["ID_TURNO", "ID_IMPLEMENTO"]], on="ID_TURNO", how="left")
    ev_sys["ID_IMPLEMENTO"] = ev_sys["ID_IMPLEMENTO"].astype(str)
    dt_sys_imp = ev_sys.groupby("ID_IMPLEMENTO", dropna=True).agg(
        DT_FALLA_HR=("DT_HR", "sum"),
        FALLAS=("DT_HR", "size")
    ).reset_index().rename(columns={"ID_IMPLEMENTO": "ID_EQUIPO"})

    top_df = to_imp_df.merge(dt_sys_imp, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})
    top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
    top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
    top_df["DISP"] = np.where(top_df["TO_HR"] > 0, (top_df["TO_HR"] - top_df["DT_FALLA_HR"]) / top_df["TO_HR"], np.nan)
    return top_df

def build_top_df_tractor():
    trc_ids = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()

    to_trc_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "TRACTOR"].copy()
    to_trc_df["ID_EQUIPO"] = to_trc_df["ID_EQUIPO"].astype(str)
    to_trc_df = to_trc_df[to_trc_df["ID_EQUIPO"].isin(trc_ids)][["ID_EQUIPO", "TO_HR"]]

    falla_trc = falla_equipo.copy()
    falla_trc["ID_EQUIPO"] = falla_trc["ID_EQUIPO"].astype(str)
    falla_trc = falla_trc[falla_trc["ID_EQUIPO"].isin(trc_ids)]

    top_df = to_trc_df.merge(falla_trc, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})
    top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
    top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
    top_df["DISP"] = np.where(top_df["TO_HR"] > 0, (top_df["TO_HR"] - top_df["DT_FALLA_HR"]) / top_df["TO_HR"], np.nan)
    return top_df

def build_top_df_implemento():
    imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids)][["ID_EQUIPO", "TO_HR"]]

    falla_imp = falla_equipo.copy()
    falla_imp["ID_EQUIPO"] = falla_imp["ID_EQUIPO"].astype(str)
    falla_imp = falla_imp[falla_imp["ID_EQUIPO"].isin(imp_ids)]

    top_df = to_imp_df.merge(falla_imp, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})
    top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
    top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
    top_df["DISP"] = np.where(top_df["TO_HR"] > 0, (top_df["TO_HR"] - top_df["DT_FALLA_HR"]) / top_df["TO_HR"], np.nan)
    return top_df

if vista_disp == "Sistema (TRC+IMP)":
    top_df = build_top_df_sistema()
elif vista_disp == "Tractor":
    top_df = build_top_df_tractor()
else:
    top_df = build_top_df_implemento()

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
