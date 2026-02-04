import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="DESEMPEÑO OPERACIONAL DE LA FLOTA - INDICADORES DE CONFIABILIDAD - MANTENIBILIDAD - DISPONIBILIDAD", layout="wide")

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
        return pd.read_csv(p, encoding="utf-8-sig")

    turnos = r("TURNOS.csv")
    horometros = r("HOROMETROS_TURNO.csv")
    eventos = r("EVENTOS_TURNO.csv")
    operadores = r("OPERADORES.csv")
    lotes = r("LOTES.csv")
    fallas_cat = r("FALLAS_CATALOGO.csv")

    # Nuevo: catálogo de procesos (en el root del repo)
    cat_path = Path("CAT_PROCESO.csv")
    if not cat_path.exists():
        st.error("Falta CAT_PROCESO.csv en el repo. Súbelo junto al script.")
        st.stop()
    cat_proceso = pd.read_csv(cat_path, encoding="utf-8-sig")

    # Tipos
    turnos["FECHA"] = pd.to_datetime(turnos["FECHA"], errors="coerce")
    horometros["TO_HORO"] = pd.to_numeric(horometros["TO_HORO"], errors="coerce")
    eventos["DT_MIN"] = pd.to_numeric(eventos["DT_MIN"], errors="coerce")

    # Llaves como string (evitar 1.0 / 2.0)
    for col in ["ID_TURNO", "ID_TRACTOR", "ID_IMPLEMENTO", "ID_LOTE", "ID_OPERADOR", "ID_PROCESO", "TURNO"]:
        if col in turnos.columns:
            turnos[col] = turnos[col].astype(str).str.replace(".0", "", regex=False)

    for col in ["ID_TURNO", "ID_EQUIPO", "TIPO_EQUIPO"]:
        if col in horometros.columns:
            horometros[col] = horometros[col].astype(str).str.replace(".0", "", regex=False)

    for col in ["ID_TURNO", "CATEGORIA_EVENTO", "ID_EQUIPO_AFECTADO", "ID_FALLA"]:
        if col in eventos.columns:
            eventos[col] = eventos[col].astype(str).str.replace(".0", "", regex=False)

    if "ID_FALLA" in fallas_cat.columns:
        fallas_cat["ID_FALLA"] = fallas_cat["ID_FALLA"].astype(str).str.replace(".0", "", regex=False)

    # Cat proceso
    cat_proceso["ID_PROCESO"] = cat_proceso["ID_PROCESO"].astype(str).str.replace(".0", "", regex=False)
    cat_proceso["NOMBRE_PROCESO"] = cat_proceso["NOMBRE_PROCESO"].astype(str)

    return {
        "turnos": turnos,
        "horometros": horometros,
        "eventos": eventos,
        "operadores": operadores,
        "lotes": lotes,
        "fallas_cat": fallas_cat,
        "cat_proceso": cat_proceso,
    }

def build_enriched_turnos(turnos, operadores, lotes):
    t = turnos.copy()

    # Operador
    op_map = dict(zip(operadores["ID_OPERADOR"].astype(str), operadores["NOMBRE_OPERADOR"].astype(str)))
    t["OPERADOR_NOMBRE"] = t["ID_OPERADOR"].astype(str).map(op_map)

    # Cultivo (se mantiene para filtro Cultivo)
    lote_map = dict(zip(lotes["ID_LOTE"].astype(str), lotes["CULTIVO"].astype(str)))
    t["CULTIVO"] = t["ID_LOTE"].astype(str).map(lote_map)

    return t

def safe_div(a, b):
    try:
        if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

# =========================================================
# CSS (Cards y paneles con borde + redondeo)
# =========================================================
st.markdown(
    """
    <style>
    .kpi-card{
        border: 2px solid rgba(0,0,0,0.18);
        border-radius: 14px;
        padding: 14px 14px 12px 14px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 10px;
    }
    .kpi-title{
        font-size: 12px;
        opacity: 0.75;
        margin-bottom: 6px;
    }
    .kpi-value{
        font-size: 30px;
        font-weight: 750;
        line-height: 1.05;
    }
    .kpi-sub{
        font-size: 12px;
        opacity: 0.7;
        margin-top: 6px;
    }
    .panel{
        border: 2px solid rgba(0,0,0,0.18);
        border-radius: 14px;
        padding: 12px 12px 6px 12px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 14px;
    }
    .filter-hint{
        font-size: 12px;
        opacity: 0.85;
        margin-top: -4px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def kpi_card(title, value_text, color=None, sub=None):
    style = f"color:{color};" if color else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="{style}">{value_text}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def mttr_color(v):
    if pd.isna(v):
        return None
    if v > 1.5:
        return "#d62728"  # rojo
    if v < 1.2:
        return "#2ca02c"  # verde
    return None

def mtbf_color(v):
    if pd.isna(v):
        return None
    if v < 100:
        return "#d62728"  # rojo
    if 100 <= v <= 500:
        return "#2ca02c"  # verde
    return "#1f77b4"      # azul

# =========================================================
# LOAD
# =========================================================
tables = load_tables()
turnos = tables["turnos"]
horometros = tables["horometros"]
eventos = tables["eventos"]
operadores = tables["operadores"]
lotes = tables["lotes"]
fallas_cat = tables["fallas_cat"]
cat_proceso = tables["cat_proceso"]

turnos = build_enriched_turnos(turnos, operadores, lotes)

# =========================================================
# SIDEBAR FILTERS (sin LOTE)
# =========================================================
st.sidebar.header("Filtros")
st.sidebar.markdown(
    '<div class="filter-hint">Los filtros muestran <b>Nombre</b> y el <b>ID</b> si aplica.</div>',
    unsafe_allow_html=True
)

min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
)

df_f = turnos.copy()
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["FECHA"] >= d1) & (df_f["FECHA"] <= d2)]

# Proceso con catálogo (Nombre [ID])
proc_map = dict(zip(cat_proceso["ID_PROCESO"], cat_proceso["NOMBRE_PROCESO"]))
proc_ids = sorted(df_f["ID_PROCESO"].dropna().astype(str).unique().tolist())
proc_labels = ["(Todos)"] + [f"{proc_map.get(pid, 'PROCESO')} [{pid}]" for pid in proc_ids]
proc_label_sel = st.sidebar.selectbox("Proceso", proc_labels, index=0)
if proc_label_sel != "(Todos)":
    id_proceso = proc_label_sel.split("[")[-1].replace("]", "").strip()
    df_f = df_f[df_f["ID_PROCESO"].astype(str) == id_proceso]

# Cultivo (texto)
cult_opts = ["(Todos)"] + sorted(df_f["CULTIVO"].dropna().astype(str).unique().tolist())
cult_sel = st.sidebar.selectbox("Cultivo", cult_opts, index=0)
if cult_sel != "(Todos)":
    df_f = df_f[df_f["CULTIVO"].astype(str) == str(cult_sel)]

# Tractor (código)
trc_opts = ["(Todos)"] + sorted(df_f["ID_TRACTOR"].dropna().astype(str).unique().tolist())
trc_sel = st.sidebar.selectbox("Tractor", trc_opts, index=0)
if trc_sel != "(Todos)":
    df_f = df_f[df_f["ID_TRACTOR"].astype(str) == str(trc_sel)]

# Implemento (código)
imp_opts = ["(Todos)"] + sorted(df_f["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
imp_sel = st.sidebar.selectbox("Implemento", imp_opts, index=0)
if imp_sel != "(Todos)":
    df_f = df_f[df_f["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]

# Operador (nombre)
op_opts = ["(Todos)"] + sorted(df_f["OPERADOR_NOMBRE"].dropna().astype(str).unique().tolist())
op_sel = st.sidebar.selectbox("Operador", op_opts, index=0)
if op_sel != "(Todos)":
    df_f = df_f[df_f["OPERADOR_NOMBRE"].astype(str) == str(op_sel)]

# Turno (D/N)
turno_opts = ["(Todos)"] + sorted(df_f["TURNO"].dropna().astype(str).unique().tolist())
turno_sel = st.sidebar.selectbox("Turno (D/N)", turno_opts, index=0)
if turno_sel != "(Todos)":
    df_f = df_f[df_f["TURNO"].astype(str) == str(turno_sel)]

st.sidebar.divider()

vista_disp = st.sidebar.radio(
    "Disponibilidad / MTBF / MTTR basados en:",
    ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
    index=0
)

# =========================================================
# DATA SELECTION
# =========================================================
turnos_sel = df_f.copy()
ids_turno = set(turnos_sel["ID_TURNO"].astype(str).tolist())

horo_sel = horometros[horometros["ID_TURNO"].astype(str).isin(ids_turno)].copy()
ev_sel = eventos[eventos["ID_TURNO"].astype(str).isin(ids_turno)].copy()

# =========================================================
# KPI GLOBAL (según vista_disp)
# =========================================================
to_trac = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "TRACTOR", "TO_HORO"].sum()
to_imp  = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO", "TO_HORO"].sum()

ev_fallas = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_fallas.empty:
    ev_fallas["DT_HR"] = ev_fallas["DT_MIN"].astype(float) / 60.0
else:
    ev_fallas["DT_HR"] = pd.Series(dtype=float)

def fallas_de_equipo(cod_equipo: str):
    return ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str) == str(cod_equipo)]

if vista_disp == "Sistema (TRC+IMP)":
    to_base = to_imp
    dt_base = ev_fallas["DT_HR"].sum()
    n_base = int(len(ev_fallas))

elif vista_disp == "Tractor":
    to_base = to_trac
    if trc_sel != "(Todos)":
        ev_b = fallas_de_equipo(trc_sel)
    else:
        trcs = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    dt_base = ev_b["DT_HR"].sum() if not ev_b.empty else 0.0
    n_base = int(len(ev_b))

else:  # Implemento
    to_base = to_imp
    if imp_sel != "(Todos)":
        ev_b = fallas_de_equipo(imp_sel)
    else:
        imps = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]
    dt_base = ev_b["DT_HR"].sum() if not ev_b.empty else 0.0
    n_base = int(len(ev_b))

mttr_hr = (dt_base / n_base) if n_base > 0 else np.nan
mtbf_hr = (to_base / n_base) if n_base > 0 else np.nan
disp = ((to_base - dt_base) / to_base) if (to_base and to_base > 0) else np.nan

# =========================================================
# UI HEADER
# =========================================================
st.title("DESEMPEÑO OPERACIONAL DE LA FLOTA - INDICADORES DE CONFIABILIDAD - MANTENIBILIDAD - DISPONIBILIDAD")
st.caption(f"Vista actual de KPIs: **{vista_disp}**")

# =========================================================
# KPI CARDS (sin card Turnos)
# =========================================================
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    kpi_card("TO Tractor (h)", f"{to_trac:,.2f}")
with col2:
    kpi_card("TO Implemento (h)", f"{to_imp:,.2f}")
with col3:
    kpi_card("Downtime Fallas (h)", f"{dt_base:,.2f}")
with col4:
    kpi_card("Fallas", f"{n_base:,}")
with col5:
    kpi_card("Disponibilidad", f"{disp*100:,.2f}%" if pd.notna(disp) else "—")
with col6:
    kpi_card(" ", " ")

col7, col8, col9 = st.columns(3)
with col7:
    kpi_card("MTTR (h/falla)", f"{mttr_hr:,.2f}" if pd.notna(mttr_hr) else "—", color=mttr_color(mttr_hr))
with col8:
    kpi_card(
        "MTBF (h/falla)",
        f"{mtbf_hr:,.2f}" if pd.notna(mtbf_hr) else "—",
        color=mtbf_color(mtbf_hr),
        sub="Rojo <100 | Verde 100–500 | Azul >500"
    )
with col9:
    kpi_card(" ", " ")

st.divider()

# =========================================================
# TOP 10 POR EQUIPO (panel con borde)
# =========================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Top 10 por Equipo")

to_equipo = horo_sel.groupby(["TIPO_EQUIPO", "ID_EQUIPO"], dropna=True)["TO_HORO"].sum().reset_index()
to_equipo["TO_HR"] = to_equipo["TO_HORO"]

ev_fallas_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
ev_fallas_rank["DT_HR"] = ev_fallas_rank["DT_MIN"].astype(float) / 60.0

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
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("MTTR alto (peor)")
    d = top_df.dropna(subset=["MTTR_HR"]).sort_values("MTTR_HR", ascending=False).head(10)
    st.plotly_chart(px.bar(d, x="MTTR_HR", y="ID_EQUIPO", orientation="h"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("MTBF bajo (peor)")
    d = top_df.dropna(subset=["MTBF_HR"]).sort_values("MTBF_HR", ascending=True).head(10)
    st.plotly_chart(px.bar(d, x="MTBF_HR", y="ID_EQUIPO", orientation="h"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with colC:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.caption("Disponibilidad baja (peor)")
    d = top_df.dropna(subset=["DISP"]).sort_values("DISP", ascending=True).head(10)
    st.plotly_chart(px.bar(d, x="DISP", y="ID_EQUIPO", orientation="h"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================================================
# TOP 10 TÉCNICO (Subsis / Comp / Parte) sin “ISO 14224” en textos
# =========================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Top 10 técnico: SUBSISTEMA / COMPONENTE / PARTE")

ev_f = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if ev_f.empty:
    st.info("No hay fallas para construir el ranking técnico.")
else:
    ev_f["DT_HR"] = ev_f["DT_MIN"].astype(float) / 60.0

    # Filtro por vista_disp
    if vista_disp == "Tractor":
        if trc_sel != "(Todos)":
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str) == str(trc_sel)]
        else:
            trcs = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    elif vista_disp == "Implemento":
        if imp_sel != "(Todos)":
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str) == str(imp_sel)]
        else:
            imps = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]

    if ev_f.empty:
        st.info("No hay fallas en la vista seleccionada para construir el ranking técnico.")
    else:
        fcat = fallas_cat.copy()
        rename_iso = {}
        if "SUB UNIDAD" in fcat.columns:
            rename_iso["SUB UNIDAD"] = "SUBSISTEMA"
        if "PIEZA" in fcat.columns:
            rename_iso["PIEZA"] = "PARTE"
        fcat = fcat.rename(columns=rename_iso)

        ev_f = ev_f.merge(fcat, on="ID_FALLA", how="left")

        metrica = st.selectbox(
            "Rankear por",
            ["Downtime (h)", "# Fallas", "MTTR (h/falla)", "MTBF (h/falla)", "Disponibilidad"],
            index=0,
            key="metrica_tecnica"
        )

        to_trac_sel = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "TRACTOR", "TO_HORO"].sum()
        to_imp_sel  = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO", "TO_HORO"].sum()
        if vista_disp == "Tractor":
            TO_BASE_GLOBAL = to_trac_sel
        elif vista_disp == "Implemento":
            TO_BASE_GLOBAL = to_imp_sel
        else:
            TO_BASE_GLOBAL = to_imp_sel

        def rank_group(col_group, titulo):
            if col_group not in ev_f.columns:
                st.info(f"No existe columna **{col_group}** en el catálogo.")
                return

            tmp = ev_f.dropna(subset=[col_group]).copy()
            if tmp.empty:
                st.info(f"No hay datos válidos para **{titulo}**.")
                return

            g = tmp.groupby(col_group, dropna=True).agg(
                DT_HR=("DT_HR", "sum"),
                FALLAS=("DT_HR", "size"),
            ).reset_index()

            g["MTTR_HR"] = np.where(g["FALLAS"] > 0, g["DT_HR"] / g["FALLAS"], np.nan)
            g["MTBF_HR"] = np.where(g["FALLAS"] > 0, TO_BASE_GLOBAL / g["FALLAS"], np.nan)
            g["DISP"]    = np.where(TO_BASE_GLOBAL > 0, (TO_BASE_GLOBAL - g["DT_HR"]) / TO_BASE_GLOBAL, np.nan)

            if metrica == "Downtime (h)":
                g["VALOR"] = g["DT_HR"]; asc = False
            elif metrica == "# Fallas":
                g["VALOR"] = g["FALLAS"]; asc = False
            elif metrica == "MTTR (h/falla)":
                g["VALOR"] = g["MTTR_HR"]; asc = False
            elif metrica == "MTBF (h/falla)":
                g["VALOR"] = g["MTBF_HR"]; asc = True
            else:
                g["VALOR"] = g["DISP"]; asc = True

            g = g.sort_values("VALOR", ascending=asc).head(10)

            st.markdown('<div class="panel">', unsafe_allow_html=True)
            fig = px.bar(g, x="VALOR", y=col_group, orientation="h", title=titulo)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            rank_group("SUBSISTEMA", "Top 10 por SUBSISTEMA")
        with c2:
            rank_group("COMPONENTE", "Top 10 por COMPONENTE")
        with c3:
            rank_group("PARTE", "Top 10 por PARTE")

        with st.expander("Ver detalle (Top 10)"):
            # muestra un resumen de las 3 tablas (si deseas)
            pass

        with st.expander("Ver tabla de fallas (vista técnica)"):
            cols_show = [c for c in ["ID_TURNO", "ID_EQUIPO_AFECTADO", "ID_FALLA", "SUBSISTEMA", "COMPONENTE", "PARTE", "DT_HR"] if c in ev_f.columns]
            st.dataframe(ev_f[cols_show].sort_values("DT_HR", ascending=False).head(300), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# =========================================================
# DESCARGAS
# =========================================================
st.subheader("Descargar datos filtrados")

turnos_out = turnos_sel.copy()
horo_out = horo_sel.copy()
ev_out = ev_sel.copy()

st.download_button(
    "Descargar TURNOS filtrado (CSV)",
    data=turnos_out.to_csv(index=False).encode("utf-8-sig"),
    file_name="TURNOS_filtrado.csv",
    mime="text/csv",
)
st.download_button(
    "Descargar HOROMETROS filtrado (CSV)",
    data=horo_out.to_csv(index=False).encode("utf-8-sig"),
    file_name="HOROMETROS_TURNO_filtrado.csv",
    mime="text/csv",
)
st.download_button(
    "Descargar EVENTOS filtrado (CSV)",
    data=ev_out.to_csv(index=False).encode("utf-8-sig"),
    file_name="EVENTOS_TURNO_filtrado.csv",
    mime="text/csv",
)
