import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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
        return pd.read_csv(p, encoding="utf-8-sig")

    turnos = r("TURNOS.csv")
    horometros = r("HOROMETROS_TURNO.csv")
    eventos = r("EVENTOS_TURNO.csv")
    operadores = r("OPERADORES.csv")
    lotes = r("LOTES.csv")
    fallas_cat = r("FALLAS_CATALOGO.csv")
    cat_proceso = r("CAT_PROCESO.csv")  # debe venir dentro del ZIP

    # Tipos
    turnos["FECHA"] = pd.to_datetime(turnos["FECHA"], errors="coerce")
    horometros["TO_HORO"] = pd.to_numeric(horometros["TO_HORO"], errors="coerce")
    eventos["DT_MIN"] = pd.to_numeric(eventos["DT_MIN"], errors="coerce")

    # Normalizar IDs (evitar 1.0)
    def norm_str(s):
        return s.astype(str).str.replace(".0", "", regex=False)

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
    cat_proceso["NOMBRE_PROCESO"] = cat_proceso["NOMBRE_PROCESO"].astype(str)

    operadores["ID_OPERADOR"] = norm_str(operadores["ID_OPERADOR"])
    operadores["NOMBRE_OPERADOR"] = operadores["NOMBRE_OPERADOR"].astype(str)

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

def build_enriched_turnos(turnos, operadores, lotes):
    t = turnos.copy()
    op_map = dict(zip(operadores["ID_OPERADOR"], operadores["NOMBRE_OPERADOR"]))
    t["OPERADOR_NOMBRE"] = t["ID_OPERADOR"].map(op_map)

    lote_map = dict(zip(lotes["ID_LOTE"], lotes["CULTIVO"]))
    t["CULTIVO"] = t["ID_LOTE"].map(lote_map)
    return t

def color_span(value_text: str, color_hex: str) -> str:
    return f"<span style='color:{color_hex}; font-weight:800;'>{value_text}</span>"

def mttr_color(v):
    if pd.isna(v): return None
    if v > 1.5: return "#d62728"  # rojo
    if v < 1.2: return "#2ca02c"  # verde
    return None

def mtbf_color(v):
    if pd.isna(v): return None
    if v < 100: return "#d62728"      # rojo
    if 100 <= v <= 500: return "#2ca02c"  # verde
    return "#1f77b4"                  # azul

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
# SIDEBAR (FORM)
# =========================================================
st.sidebar.header("Filtros")

min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
default_date = (min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None)

if "applied" not in st.session_state:
    st.session_state.applied = {
        "date_range": default_date,
        "proc_label": "(Todos)",
        "cultivo": "(Todos)",
        "tractor": "(Todos)",
        "implemento": "(Todos)",
        "operador": "(Todos)",
        "turno": "(Todos)",
        "vista_disp": "Sistema (TRC+IMP)",
    }

proc_map = dict(zip(cat_proceso["ID_PROCESO"], cat_proceso["NOMBRE_PROCESO"]))
all_proc_ids = sorted(turnos["ID_PROCESO"].dropna().astype(str).unique().tolist())
proc_labels = ["(Todos)"] + [f"{proc_map.get(pid, 'PROCESO')} [{pid}]" for pid in all_proc_ids]

cult_opts_all = ["(Todos)"] + sorted(turnos["CULTIVO"].dropna().astype(str).unique().tolist())
trc_opts_all = ["(Todos)"] + sorted(turnos["ID_TRACTOR"].dropna().astype(str).unique().tolist())
imp_opts_all = ["(Todos)"] + sorted(turnos["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
op_opts_all = ["(Todos)"] + sorted(turnos["OPERADOR_NOMBRE"].dropna().astype(str).unique().tolist())
turno_opts_all = ["(Todos)"] + sorted(turnos["TURNO"].dropna().astype(str).unique().tolist())

with st.sidebar.form("filtros_form", clear_on_submit=False):
    date_range = st.date_input("Rango de fechas", value=st.session_state.applied["date_range"])

    proc_label_sel = st.selectbox(
        "Proceso",
        proc_labels,
        index=proc_labels.index(st.session_state.applied["proc_label"])
        if st.session_state.applied["proc_label"] in proc_labels else 0
    )

    cult_sel = st.selectbox(
        "Cultivo",
        cult_opts_all,
        index=cult_opts_all.index(st.session_state.applied["cultivo"])
        if st.session_state.applied["cultivo"] in cult_opts_all else 0
    )

    trc_sel = st.selectbox(
        "Tractor",
        trc_opts_all,
        index=trc_opts_all.index(st.session_state.applied["tractor"])
        if st.session_state.applied["tractor"] in trc_opts_all else 0
    )

    imp_sel = st.selectbox(
        "Implemento",
        imp_opts_all,
        index=imp_opts_all.index(st.session_state.applied["implemento"])
        if st.session_state.applied["implemento"] in imp_opts_all else 0
    )

    op_sel = st.selectbox(
        "Operador",
        op_opts_all,
        index=op_opts_all.index(st.session_state.applied["operador"])
        if st.session_state.applied["operador"] in op_opts_all else 0
    )

    turno_sel = st.selectbox(
        "Turno (D/N)",
        turno_opts_all,
        index=turno_opts_all.index(st.session_state.applied["turno"])
        if st.session_state.applied["turno"] in turno_opts_all else 0
    )

    vista_disp = st.radio(
        "Disponibilidad / MTBF / MTTR basados en:",
        ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
        index=["Sistema (TRC+IMP)", "Tractor", "Implemento"].index(st.session_state.applied["vista_disp"])
    )

    apply_btn = st.form_submit_button("✅ Aplicar filtros")

if apply_btn:
    st.session_state.applied = {
        "date_range": date_range,
        "proc_label": proc_label_sel,
        "cultivo": cult_sel,
        "tractor": trc_sel,
        "implemento": imp_sel,
        "operador": op_sel,
        "turno": turno_sel,
        "vista_disp": vista_disp,
    }

ap = st.session_state.applied

# =========================================================
# FILTRADO
# =========================================================
df_f = turnos.copy()

date_range = ap["date_range"]
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["FECHA"] >= d1) & (df_f["FECHA"] <= d2)]

proc_label_sel = ap["proc_label"]
if proc_label_sel != "(Todos)":
    id_proceso = proc_label_sel.split("[")[-1].replace("]", "").strip()
    df_f = df_f[df_f["ID_PROCESO"].astype(str) == id_proceso]

cult_sel = ap["cultivo"]
if cult_sel != "(Todos)":
    df_f = df_f[df_f["CULTIVO"].astype(str) == str(cult_sel)]

trc_sel = ap["tractor"]
if trc_sel != "(Todos)":
    df_f = df_f[df_f["ID_TRACTOR"].astype(str) == str(trc_sel)]

imp_sel = ap["implemento"]
if imp_sel != "(Todos)":
    df_f = df_f[df_f["ID_IMPLEMENTO"].astype(str) == str(imp_sel)]

op_sel = ap["operador"]
if op_sel != "(Todos)":
    df_f = df_f[df_f["OPERADOR_NOMBRE"].astype(str) == str(op_sel)]

turno_sel = ap["turno"]
if turno_sel != "(Todos)":
    df_f = df_f[df_f["TURNO"].astype(str) == str(turno_sel)]

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
    ev_fallas["DT_HR"] = ev_fallas["DT_MIN"].astype(float) / 60.0
else:
    ev_fallas["DT_HR"] = pd.Series(dtype=float)

def fallas_de_equipo(cod_equipo: str):
    return ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str) == str(cod_equipo)]

vista_disp = ap["vista_disp"]

if vista_disp == "Sistema (TRC+IMP)":
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
# KPIs (SIN CARDS/CSS)
# =========================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("TO Tractor (h)", f"{to_trac:,.2f}")
c2.metric("TO Implemento (h)", f"{to_imp:,.2f}")
c3.metric("Downtime Fallas (h)", f"{dt_base:,.2f}")
c4.metric("Fallas", f"{n_base:,}")
c5.metric("Disponibilidad", f"{disp*100:,.2f}%" if pd.notna(disp) else "—")

c6, c7 = st.columns(2)
mttr_txt = f"{mttr_hr:,.2f}" if pd.notna(mttr_hr) else "—"
mtbf_txt = f"{mtbf_hr:,.2f}" if pd.notna(mtbf_hr) else "—"

mttr_c = mttr_color(mttr_hr)
mtbf_c = mtbf_color(mtbf_hr)

with c6:
    st.markdown("### MTTR (h/falla)")
    if mttr_c:
        st.markdown(color_span(mttr_txt, mttr_c), unsafe_allow_html=True)
    else:
        st.markdown(f"**{mttr_txt}**")

with c7:
    st.markdown("### MTBF (h/falla)")
    if mtbf_c:
        st.markdown(color_span(mtbf_txt, mtbf_c), unsafe_allow_html=True)
    else:
        st.markdown(f"**{mtbf_txt}**")
    st.caption("Rojo <100 | Verde 100–500 | Azul >500")

st.divider()

# =========================================================
# TOP 10 POR EQUIPO (SIN BORDES)
# =========================================================
st.subheader("Top 10 por Equipo")

to_equipo = horo_sel.groupby(["TIPO_EQUIPO", "ID_EQUIPO"], dropna=True)["TO_HORO"].sum().reset_index()
to_equipo["TO_HR"] = to_equipo["TO_HORO"]

ev_fallas_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_fallas_rank.empty:
    ev_fallas_rank["DT_HR"] = ev_fallas_rank["DT_MIN"].astype(float) / 60.0
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
    st.plotly_chart(fig, width="stretch")

with colB:
    st.caption("MTBF bajo (peor)")
    d = top_df.dropna(subset=["MTBF_HR"]).sort_values("MTBF_HR", ascending=True).head(10)
    fig = px.bar(d, x="MTBF_HR", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with colC:
    st.caption("Disponibilidad baja (peor)")
    d = top_df.dropna(subset=["DISP"]).sort_values("DISP", ascending=True).head(10)
    fig = px.bar(d, x="DISP", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# =========================================================
# TOP 10 TÉCNICO (SIN "ISO14224")
# =========================================================
st.subheader("Top 10 técnico: SUBSISTEMA / COMPONENTE / PARTE")

ev_f = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if ev_f.empty:
    st.info("No hay fallas para construir el ranking técnico.")
else:
    ev_f["DT_HR"] = ev_f["DT_MIN"].astype(float) / 60.0

    # filtro por vista
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
        if "SUB UNIDAD" in fcat.columns:
            fcat = fcat.rename(columns={"SUB UNIDAD": "SUBSISTEMA"})
        if "PIEZA" in fcat.columns:
            fcat = fcat.rename(columns={"PIEZA": "PARTE"})

        ev_f = ev_f.merge(fcat, on="ID_FALLA", how="left")

        metrica = st.selectbox(
            "Rankear por",
            ["Downtime (h)", "# Fallas", "MTTR (h/falla)", "MTBF (h/falla)", "Disponibilidad"],
            index=0,
            key="metrica_tecnica"
        )

        # TO base global (para calcular MTBF/DISP por grupo)
        to_trac_sel = float(horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "TRACTOR", "TO_HORO"].sum())
        to_imp_sel = float(horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO", "TO_HORO"].sum())
        TO_BASE_GLOBAL = to_imp_sel if vista_disp == "Sistema (TRC+IMP)" else (to_trac_sel if vista_disp == "Tractor" else to_imp_sel)

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
            g["DISP"] = np.where(TO_BASE_GLOBAL > 0, (TO_BASE_GLOBAL - g["DT_HR"]) / TO_BASE_GLOBAL, np.nan)

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
            fig = px.bar(g, x="VALOR", y=col_group, orientation="h", title=titulo)
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, width="stretch")

        c1, c2, c3 = st.columns(3)
        with c1:
            rank_group("SUBSISTEMA", "Top 10 por SUBSISTEMA")
        with c2:
            rank_group("COMPONENTE", "Top 10 por COMPONENTE")
        with c3:
            rank_group("PARTE", "Top 10 por PARTE")

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
