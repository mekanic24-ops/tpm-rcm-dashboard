import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="TPM | Operación + Confiabilidad", layout="wide")

ZIP_NAME = "TPM_modelo_normalizado_CSV.zip"
DATA_DIR = Path("data_normalizada")

# ----------------------------
# Helpers
# ----------------------------
def ensure_data_unzipped():
    """Unzip the normalized CSV package into a local folder."""
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
    traslados = r("TRASLADOS_EXTERNOS.csv")
    horometros = r("HOROMETROS_TURNO.csv")
    eventos = r("EVENTOS_TURNO.csv")
    operadores = r("OPERADORES.csv")
    lotes = r("LOTES.csv")
    fallas_cat = r("FALLAS_CATALOGO.csv")

    # Tipos
    turnos["FECHA"] = pd.to_datetime(turnos["FECHA"], errors="coerce")

    for c in ["DT_TURNO_MIN", "DEMORA_SALIDA_MIN", "TO_TRACTOR_HR"]:
        if c in turnos.columns:
            turnos[c] = pd.to_numeric(turnos[c], errors="coerce")

    horometros["TO_HORO"] = pd.to_numeric(horometros["TO_HORO"], errors="coerce")
    eventos["DT_MIN"] = pd.to_numeric(eventos["DT_MIN"], errors="coerce")

    # Asegurar strings para llaves
    for col in ["ID_TURNO", "ID_TRACTOR", "ID_IMPLEMENTO", "ID_LOTE", "ID_OPERADOR", "ID_PROCESO"]:
        if col in turnos.columns:
            turnos[col] = turnos[col].astype(str)

    for col in ["ID_TURNO", "ID_EQUIPO", "TIPO_EQUIPO"]:
        if col in horometros.columns:
            horometros[col] = horometros[col].astype(str)

    for col in ["ID_TURNO", "CATEGORIA_EVENTO", "ID_EQUIPO_AFECTADO", "ID_FALLA"]:
        if col in eventos.columns:
            eventos[col] = eventos[col].astype(str)

    return {
        "turnos": turnos,
        "traslados": traslados,
        "horometros": horometros,
        "eventos": eventos,
        "operadores": operadores,
        "lotes": lotes,
        "fallas_cat": fallas_cat,
    }

def safe_div(a, b):
    try:
        if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def build_enriched_turnos(turnos, operadores, lotes):
    # Operador name
    op_map = dict(zip(operadores["ID_OPERADOR"].astype(str), operadores["NOMBRE_OPERADOR"].astype(str)))
    t = turnos.copy()
    t["OPERADOR_NOMBRE"] = t["ID_OPERADOR"].astype(str).map(op_map)

    # Cultivo via lotes
    lote_map = dict(zip(lotes["ID_LOTE"].astype(str), lotes["CULTIVO"].astype(str)))
    t["CULTIVO"] = t["ID_LOTE"].astype(str).map(lote_map)
    return t

def cascade_options(df, col):
    vals = df[col].dropna().astype(str)
    vals = vals[vals != ""]
    return ["(Todos)"] + sorted(vals.unique().tolist())

# ----------------------------
# Load
# ----------------------------
tables = load_tables()
turnos = tables["turnos"]
horometros = tables["horometros"]
eventos = tables["eventos"]
operadores = tables["operadores"]
lotes = tables["lotes"]
fallas_cat = tables["fallas_cat"]

turnos = build_enriched_turnos(turnos, operadores, lotes)

# ----------------------------
# Sidebar Filters (cascading)
# ----------------------------
st.sidebar.header("Filtros")

# Rango de fechas
min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
)

# Filtro base por fechas
df_f = turnos.copy()
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["FECHA"] >= d1) & (df_f["FECHA"] <= d2)]

# Proceso
proc_opts = cascade_options(df_f, "ID_PROCESO")
id_proceso = st.sidebar.selectbox("Proceso (ID_PROCESO)", proc_opts, index=0)
if id_proceso != "(Todos)":
    df_f = df_f[df_f["ID_PROCESO"].astype(str) == str(id_proceso)]

# Cultivo
cult_opts = cascade_options(df_f, "CULTIVO")
cultivo = st.sidebar.selectbox("Cultivo", cult_opts, index=0)
if cultivo != "(Todos)":
    df_f = df_f[df_f["CULTIVO"].astype(str) == str(cultivo)]

# Lote
lote_opts = cascade_options(df_f, "ID_LOTE")
id_lote = st.sidebar.selectbox("Lote (ID_LOTE)", lote_opts, index=0)
if id_lote != "(Todos)":
    df_f = df_f[df_f["ID_LOTE"].astype(str) == str(id_lote)]

# Tractor
trc_opts = cascade_options(df_f, "ID_TRACTOR")
id_trc = st.sidebar.selectbox("Tractor (ID_TRACTOR)", trc_opts, index=0)
if id_trc != "(Todos)":
    df_f = df_f[df_f["ID_TRACTOR"].astype(str) == str(id_trc)]

# Implemento
imp_opts = cascade_options(df_f, "ID_IMPLEMENTO")
id_imp = st.sidebar.selectbox("Implemento (ID_IMPLEMENTO)", imp_opts, index=0)
if id_imp != "(Todos)":
    df_f = df_f[df_f["ID_IMPLEMENTO"].astype(str) == str(id_imp)]

# Operador
op_opts = cascade_options(df_f, "OPERADOR_NOMBRE")
op_name = st.sidebar.selectbox("Operador", op_opts, index=0)
if op_name != "(Todos)":
    df_f = df_f[df_f["OPERADOR_NOMBRE"].astype(str) == str(op_name)]

# Turno D/N
turno_opts = cascade_options(df_f, "TURNO")
turno_dn = st.sidebar.selectbox("Turno (D/N)", turno_opts, index=0)
if turno_dn != "(Todos)":
    df_f = df_f[df_f["TURNO"].astype(str) == str(turno_dn)]

st.sidebar.divider()

# ✅ Nuevo selector de KPIs (Disponibilidad/MTBF/MTTR)
vista_disp = st.sidebar.radio(
    "Disponibilidad / MTBF / MTTR basados en:",
    ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
    index=0
)

# ----------------------------
# Derivados para cálculo
# ----------------------------
turnos_sel = df_f.copy()
ids_turno = set(turnos_sel["ID_TURNO"].astype(str).tolist())

horo_sel = horometros[horometros["ID_TURNO"].astype(str).isin(ids_turno)].copy()
ev_sel = eventos[eventos["ID_TURNO"].astype(str).isin(ids_turno)].copy()

# ----------------------------
# KPI global (según vista_disp)
# ----------------------------
to_trac = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "TRACTOR", "TO_HORO"].sum()
to_imp  = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO", "TO_HORO"].sum()

ev_fallas = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_fallas.empty:
    ev_fallas["DT_HR"] = ev_fallas["DT_MIN"].astype(float) / 60.0
else:
    ev_fallas["DT_HR"] = pd.Series(dtype=float)

def fallas_de_equipo(cod_equipo: str):
    return ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str) == str(cod_equipo)]

# Base según vista
if vista_disp == "Sistema (TRC+IMP)":
    # TO base productiva: implemento (neto)
    to_base = to_imp
    dt_base = ev_fallas["DT_HR"].sum()
    n_base  = int(len(ev_fallas))

elif vista_disp == "Tractor":
    to_base = to_trac
    if id_trc != "(Todos)":
        ev_b = fallas_de_equipo(id_trc)
    else:
        trcs = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    dt_base = ev_b["DT_HR"].sum() if not ev_b.empty else 0.0
    n_base  = int(len(ev_b))

else:  # Implemento
    to_base = to_imp
    if id_imp != "(Todos)":
        ev_b = fallas_de_equipo(id_imp)
    else:
        imps = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
        ev_b = ev_fallas[ev_fallas["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]
    dt_base = ev_b["DT_HR"].sum() if not ev_b.empty else 0.0
    n_base  = int(len(ev_b))

# KPIs
mttr_hr = (dt_base / n_base) if n_base > 0 else np.nan
mtbf_hr = (to_base / n_base) if n_base > 0 else np.nan
disp    = ((to_base - dt_base) / to_base) if (to_base and to_base > 0) else np.nan

# Para referencia (sistema total dentro del filtro)
dt_fallas_sistema_hr = ev_fallas["DT_HR"].sum() if not ev_fallas.empty else 0.0
n_fallas_sistema = int(len(ev_fallas))

# ----------------------------
# UI
# ----------------------------
st.title("TPM | Operación Agrícola + Confiabilidad (Modelo Normalizado)")
st.caption(f"Vista actual de KPIs: **{vista_disp}**")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Turnos", f"{len(turnos_sel):,}")
c2.metric("TO Tractor (h)", f"{to_trac:,.2f}")
c3.metric("TO Implemento (h)", f"{to_imp:,.2f}")
c4.metric("Downtime Fallas (h)", f"{dt_base:,.2f}")
c5.metric("Fallas", f"{n_base:,}")
c6.metric("Disponibilidad", f"{disp*100:,.2f}%" if pd.notna(disp) else "—")

c7, c8 = st.columns(2)
c7.metric("MTTR (h/falla)", f"{mttr_hr:,.2f}" if pd.notna(mttr_hr) else "—")
c8.metric("MTBF (h/falla)", f"{mtbf_hr:,.2f}" if pd.notna(mtbf_hr) else "—")

with st.expander("Referencia (Sistema completo dentro del filtro)"):
    st.write(f"- Downtime sistema (h): **{dt_fallas_sistema_hr:,.2f}**")
    st.write(f"- Fallas sistema: **{n_fallas_sistema:,}**")
    st.write(f"- TO base sistema (implemento, h): **{to_imp:,.2f}**")

st.divider()

# ----------------------------
# Tendencias por fecha
# ----------------------------
st.subheader("Tendencia por fecha")

# TO por día (tractor/implemento)
horo_day = horo_sel.merge(turnos_sel[["ID_TURNO", "FECHA"]], on="ID_TURNO", how="left")
to_day = horo_day.groupby(["FECHA", "TIPO_EQUIPO"], dropna=True)["TO_HORO"].sum().reset_index()

fig_to = px.line(to_day, x="FECHA", y="TO_HORO", color="TIPO_EQUIPO", markers=True)
st.plotly_chart(fig_to, use_container_width=True)

# Fallas por día (siempre sistema total; si quieres que siga vista_disp también, te lo adapto)
ev_day = ev_sel.merge(turnos_sel[["ID_TURNO", "FECHA"]], on="ID_TURNO", how="left")
f_day = ev_day[ev_day["CATEGORIA_EVENTO"] == "FALLA"].groupby("FECHA").size().reset_index(name="Fallas")
if not f_day.empty:
    fig_fd = px.bar(f_day, x="FECHA", y="Fallas")
    st.plotly_chart(fig_fd, use_container_width=True)
else:
    st.info("No hay fallas en el rango/selección.")

st.divider()

# ----------------------------
# Top 10 por Equipo (coherente con vista_disp)
# ----------------------------
st.subheader("Top 10 por Equipo (según vista de KPIs)")

# Base horómetros por equipo
to_equipo = horo_sel.groupby(["TIPO_EQUIPO", "ID_EQUIPO"], dropna=True)["TO_HORO"].sum().reset_index()
to_equipo["TO_HR"] = to_equipo["TO_HORO"]

# Base fallas por equipo afectado
ev_fallas_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
ev_fallas_rank["DT_HR"] = ev_fallas_rank["DT_MIN"].astype(float) / 60.0
falla_equipo = ev_fallas_rank.groupby("ID_EQUIPO_AFECTADO", dropna=True).agg(
    DT_FALLA_HR=("DT_HR", "sum"),
    FALLAS=("DT_HR", "size")
).reset_index().rename(columns={"ID_EQUIPO_AFECTADO": "ID_EQUIPO"})

def build_top_df_sistema():
    # Implementos presentes en turnos filtrados
    imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    # TO por implemento
    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids)][["ID_EQUIPO", "TO_HR"]]

    # Downtime del sistema por implemento:
    # sumamos TODAS las fallas del turno (tractor o implemento) agrupadas por implemento del turno
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
    top_df["TIPO_RANKING"] = "IMPLEMENTO (Sistema)"
    return top_df

def build_top_df_tractor():
    trc_ids = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()

    # TO por tractor
    to_trc_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "TRACTOR"].copy()
    to_trc_df["ID_EQUIPO"] = to_trc_df["ID_EQUIPO"].astype(str)
    to_trc_df = to_trc_df[to_trc_df["ID_EQUIPO"].isin(trc_ids)][["ID_EQUIPO", "TO_HR"]]

    # DT y #fallas solo de tractor
    falla_trc = falla_equipo.copy()
    falla_trc["ID_EQUIPO"] = falla_trc["ID_EQUIPO"].astype(str)
    falla_trc = falla_trc[falla_trc["ID_EQUIPO"].isin(trc_ids)]

    top_df = to_trc_df.merge(falla_trc, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})
    top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
    top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
    top_df["DISP"] = np.where(top_df["TO_HR"] > 0, (top_df["TO_HR"] - top_df["DT_FALLA_HR"]) / top_df["TO_HR"], np.nan)
    top_df["TIPO_RANKING"] = "TRACTOR"
    return top_df

def build_top_df_implemento():
    imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    # TO por implemento
    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids)][["ID_EQUIPO", "TO_HR"]]

    # DT y #fallas solo de implemento
    falla_imp = falla_equipo.copy()
    falla_imp["ID_EQUIPO"] = falla_imp["ID_EQUIPO"].astype(str)
    falla_imp = falla_imp[falla_imp["ID_EQUIPO"].isin(imp_ids)]

    top_df = to_imp_df.merge(falla_imp, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})
    top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
    top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
    top_df["DISP"] = np.where(top_df["TO_HR"] > 0, (top_df["TO_HR"] - top_df["DT_FALLA_HR"]) / top_df["TO_HR"], np.nan)
    top_df["TIPO_RANKING"] = "IMPLEMENTO"
    return top_df

if vista_disp == "Sistema (TRC+IMP)":
    top_df = build_top_df_sistema()
elif vista_disp == "Tractor":
    top_df = build_top_df_tractor()
else:
    top_df = build_top_df_implemento()

if top_df.empty:
    st.info("No hay datos suficientes para construir Top 10 por Equipo con los filtros actuales.")
else:
    st.caption(f"Ranking basado en: **{top_df['TIPO_RANKING'].iloc[0]}**")

    colA, colB, colC = st.columns(3)

    with colA:
        st.caption("MTTR alto (peor)")
        d = top_df.dropna(subset=["MTTR_HR"]).sort_values("MTTR_HR", ascending=False).head(10)
        st.plotly_chart(px.bar(d, x="MTTR_HR", y="ID_EQUIPO", orientation="h"), use_container_width=True)

    with colB:
        st.caption("MTBF bajo (peor)")
        d = top_df.dropna(subset=["MTBF_HR"]).sort_values("MTBF_HR", ascending=True).head(10)
        st.plotly_chart(px.bar(d, x="MTBF_HR", y="ID_EQUIPO", orientation="h"), use_container_width=True)

    with colC:
        st.caption("Disponibilidad baja (peor)")
        d = top_df.dropna(subset=["DISP"]).sort_values("DISP", ascending=True).head(10)
        st.plotly_chart(px.bar(d, x="DISP", y="ID_EQUIPO", orientation="h"), use_container_width=True)

    with st.expander("Ver tabla Top 10 (detalle)"):
        st.dataframe(
            top_df.sort_values(["DISP", "MTTR_HR"], ascending=[True, False]).head(50),
            use_container_width=True
        )

st.divider()

# ----------------------------
# Top 10 técnico ISO 14224 (con selector de métrica) + coherente con vista_disp
# ----------------------------
st.subheader("Top 10 técnico (ISO 14224): SUBSISTEMA / COMPONENTE / PARTE")

ev_f = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if ev_f.empty:
    st.info("No hay fallas para construir el ranking técnico.")
else:
    ev_f["DT_HR"] = ev_f["DT_MIN"].astype(float) / 60.0

    # Filtro por vista_disp
    if vista_disp == "Tractor":
        if id_trc != "(Todos)":
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str) == str(id_trc)]
        else:
            trcs = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str).isin(trcs)]
    elif vista_disp == "Implemento":
        if id_imp != "(Todos)":
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str) == str(id_imp)]
        else:
            imps = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
            ev_f = ev_f[ev_f["ID_EQUIPO_AFECTADO"].astype(str).isin(imps)]
    # Sistema: no filtra

    if ev_f.empty:
        st.info("No hay fallas en la vista seleccionada para construir el ranking técnico.")
    else:
        # Join con catálogo de fallas (ISO 14224)
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
            key="metrica_tecnica_iso"
        )

        st.caption(f"Vista técnica basada en fallas de: **{vista_disp}**")

        # TO base global según vista (para MTBF/Disponibilidad técnico)
        to_trac_sel = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "TRACTOR", "TO_HORO"].sum()
        to_imp_sel  = horo_sel.loc[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO", "TO_HORO"].sum()

        if vista_disp == "Tractor":
            TO_BASE_GLOBAL = to_trac_sel
        elif vista_disp == "Implemento":
            TO_BASE_GLOBAL = to_imp_sel
        else:
            TO_BASE_GLOBAL = to_imp_sel  # Sistema: base productiva

        def rank_iso(col_group, titulo):
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
                g["VALOR"] = g["DT_HR"]
                asc = False
            elif metrica == "# Fallas":
                g["VALOR"] = g["FALLAS"]
                asc = False
            elif metrica == "MTTR (h/falla)":
                g["VALOR"] = g["MTTR_HR"]
                asc = False
            elif metrica == "MTBF (h/falla)":
                g["VALOR"] = g["MTBF_HR"]
                asc = True
            else:  # Disponibilidad
                g["VALOR"] = g["DISP"]
                asc = True

            g = g.sort_values("VALOR", ascending=asc).head(10)

            fig = px.bar(g, x="VALOR", y=col_group, orientation="h", title=titulo)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"Ver detalle - {titulo}"):
                show_cols = [col_group, "DT_HR", "FALLAS", "MTTR_HR", "MTBF_HR", "DISP"]
                st.dataframe(g[show_cols], use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            rank_iso("SUBSISTEMA", "Top 10 por SUBSISTEMA (ISO 14224)")
        with c2:
            rank_iso("COMPONENTE", "Top 10 por COMPONENTE (ISO 14224)")
        with c3:
            rank_iso("PARTE", "Top 10 por PARTE (ISO 14224)")

        with st.expander("Ver tabla de fallas (vista técnica)"):
            cols_show = [c for c in ["ID_TURNO", "ID_EQUIPO_AFECTADO", "ID_FALLA", "SUBSISTEMA", "COMPONENTE", "PARTE", "DT_HR"] if c in ev_f.columns]
            st.dataframe(ev_f[cols_show].sort_values("DT_HR", ascending=False).head(300), use_container_width=True)

st.divider()

# ----------------------------
# Descargas (datos filtrados)
# ----------------------------
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
