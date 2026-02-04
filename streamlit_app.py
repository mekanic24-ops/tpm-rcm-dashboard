import zipfile
from pathlib import Path
from typing import Optional, List, Tuple

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
    cat_proceso = r("CAT_PROCESO.csv")

    # Tipos base
    turnos["FECHA"] = pd.to_datetime(turnos["FECHA"], errors="coerce")
    horometros["TO_HORO"] = pd.to_numeric(horometros["TO_HORO"], errors="coerce")
    eventos["DT_MIN"] = pd.to_numeric(eventos["DT_MIN"], errors="coerce")

    def norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).str.replace(".0", "", regex=False).str.strip()

    for col in ["ID_TURNO", "ID_TRACTOR", "ID_IMPLEMENTO", "ID_LOTE", "ID_OPERADOR", "ID_PROCESO", "TURNO"]:
        if col in turnos.columns:
            turnos[col] = norm_str(turnos[col])

    for col in ["ID_TURNO", "ID_EQUIPO", "TIPO_EQUIPO"]:
        if col in horometros.columns:
            horometros[col] = norm_str(horometros[col])

    for col in ["ID_TURNO", "CATEGORIA_EVENTO", "ID_EQUIPO_AFECTADO", "ID_FALLA"]:
        if col in eventos.columns:
            eventos[col] = norm_str(eventos[col])

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

def norm_turno(x) -> str:
    s = str(x).strip().upper()
    if s in ["D", "DIA", "DÍA", "DAY"]:
        return "D"
    if s in ["N", "NOCHE", "NIGHT"]:
        return "N"
    return s

def build_enriched_turnos(turnos, operadores, lotes):
    t = turnos.copy()
    op_map = dict(zip(operadores["ID_OPERADOR"], operadores["NOMBRE_OPERADOR"]))
    t["OPERADOR_NOMBRE"] = t["ID_OPERADOR"].map(op_map)

    lote_map = dict(zip(lotes["ID_LOTE"], lotes["CULTIVO"]))
    t["CULTIVO"] = t["ID_LOTE"].map(lote_map)
    t["CULTIVO"] = t["CULTIVO"].apply(normalize_cultivo)

    if "TURNO" in t.columns:
        t["TURNO_NORM"] = t["TURNO"].apply(norm_turno)
    return t

def mttr_color_3(v):
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

def parse_date_range(dr) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if isinstance(dr, tuple) and len(dr) == 2 and all(dr):
        return pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    return None, None

def apply_filters(df: pd.DataFrame,
                  d1: Optional[pd.Timestamp],
                  d2: Optional[pd.Timestamp],
                  cultivo: str,
                  tractor: str,
                  implemento: str,
                  turno: str,
                  id_proceso: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if d1 is not None and d2 is not None:
        out = out[(out["FECHA"] >= d1) & (out["FECHA"] <= d2)]
    if cultivo != "(Todos)":
        out = out[out["CULTIVO"] == cultivo]
    if tractor != "(Todos)":
        out = out[out["ID_TRACTOR"].astype(str) == str(tractor)]
    if implemento != "(Todos)":
        out = out[out["ID_IMPLEMENTO"].astype(str) == str(implemento)]
    if turno != "(Todos)":
        if "TURNO_NORM" in out.columns:
            out = out[out["TURNO_NORM"] == turno]
        else:
            out = out[out["TURNO"].apply(norm_turno) == turno]
    if id_proceso is not None:
        out = out[out["ID_PROCESO"].astype(str) == str(id_proceso)]
    return out

# =========================================================
# LOAD
# =========================================================
tables = load_tables()
turnos = build_enriched_turnos(tables["turnos"], tables["operadores"], tables["lotes"])
horometros = tables["horometros"]
eventos = tables["eventos"]
cat_proceso = tables["cat_proceso"]

proc_map = dict(zip(cat_proceso["ID_PROCESO"].astype(str), cat_proceso["NOMBRE_PROCESO"].astype(str)))
name_to_id = {v.strip().upper(): k for k, v in proc_map.items()}

# =========================================================
# SESSION STATE (para no perder selección al cambiar otro filtro)
# =========================================================
if "sel_proceso_name" not in st.session_state:
    st.session_state.sel_proceso_name = "(Todos)"
if "sel_turno_btn" not in st.session_state:
    st.session_state.sel_turno_btn = "(Todos)"
if "sel_cultivo_btn" not in st.session_state:
    st.session_state.sel_cultivo_btn = "(Todos)"

# =========================================================
# SIDEBAR FILTERS (INTERACTIVO + PERSISTENTE)
# =========================================================
st.sidebar.header("Filtros")

min_d = turnos["FECHA"].min()
max_d = turnos["FECHA"].max()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
)

d1, d2 = parse_date_range(date_range)

# --- Cultivo como botones ---
cult_choice = st.sidebar.radio(
    "Cultivo",
    ["(Todos)", "Palto", "Arandano"],
    index=["(Todos)", "Palto", "Arandano"].index(st.session_state.sel_cultivo_btn),
    key="sel_cultivo_btn"
)
cultivo_sel = "(Todos)"
if cult_choice == "Palto":
    cultivo_sel = "PALTO"
elif cult_choice == "Arandano":
    cultivo_sel = "ARANDANO"

# Tractor
df_pre = apply_filters(turnos, d1, d2, cultivo_sel, "(Todos)", "(Todos)", "(Todos)", None)
trc_opts = ["(Todos)"] + sorted(df_pre["ID_TRACTOR"].dropna().astype(str).unique().tolist())
trc_sel = st.sidebar.selectbox("Tractor", trc_opts, index=0)

# Implemento
df_pre2 = apply_filters(turnos, d1, d2, cultivo_sel, trc_sel, "(Todos)", "(Todos)", None)
imp_opts = ["(Todos)"] + sorted(df_pre2["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist())
imp_sel = st.sidebar.selectbox("Implemento", imp_opts, index=0)

# Turno como botones (Día, Noche)
turno_btn = st.sidebar.radio(
    "Turno",
    ["(Todos)", "Día", "Noche"],
    index=["(Todos)", "Día", "Noche"].index(st.session_state.sel_turno_btn),
    key="sel_turno_btn",
    horizontal=True
)
turno_sel = "(Todos)"
if turno_btn != "(Todos)":
    turno_sel = "D" if turno_btn == "Día" else "N"

# --- Proceso ordenado por TO Implemento (y que no se reinicie) ---
df_for_proc = apply_filters(turnos, d1, d2, cultivo_sel, trc_sel, imp_sel, turno_sel, None)
ids_turno_proc = set(df_for_proc["ID_TURNO"].astype(str).tolist())

h_proc = horometros[
    (horometros["ID_TURNO"].astype(str).isin(ids_turno_proc)) &
    (horometros["TIPO_EQUIPO"].astype(str) == "IMPLEMENTO")
].copy()

to_turno_imp = h_proc.groupby("ID_TURNO", dropna=True)["TO_HORO"].sum().reset_index()
to_turno_imp["ID_TURNO"] = to_turno_imp["ID_TURNO"].astype(str)

tmp_proc = df_for_proc[["ID_TURNO", "ID_PROCESO"]].copy()
tmp_proc["ID_TURNO"] = tmp_proc["ID_TURNO"].astype(str)
tmp_proc["ID_PROCESO"] = tmp_proc["ID_PROCESO"].astype(str)

proc_to = tmp_proc.merge(to_turno_imp, on="ID_TURNO", how="left").fillna({"TO_HORO": 0.0})
proc_to = proc_to.groupby("ID_PROCESO", dropna=True)["TO_HORO"].sum().reset_index(name="TO_IMP_HR")
proc_to = cat_proceso[["ID_PROCESO", "NOMBRE_PROCESO"]].astype(str).merge(
    proc_to, on="ID_PROCESO", how="left"
).fillna({"TO_IMP_HR": 0.0})
proc_to["NOMBRE_PROCESO"] = proc_to["NOMBRE_PROCESO"].astype(str).str.strip()
proc_to = proc_to.sort_values("TO_IMP_HR", ascending=False)

proc_names_sorted = ["(Todos)"] + proc_to["NOMBRE_PROCESO"].tolist()

# asegurar que el valor guardado exista; si no, volver a (Todos)
if st.session_state.sel_proceso_name not in proc_names_sorted:
    st.session_state.sel_proceso_name = "(Todos)"

proc_name_sel = st.sidebar.radio(
    "Proceso (ordenado por TO implemento)",
    proc_names_sorted,
    index=proc_names_sorted.index(st.session_state.sel_proceso_name),
    key="sel_proceso_name"
)

id_proceso_sel = None
if proc_name_sel != "(Todos)":
    id_proceso_sel = name_to_id.get(proc_name_sel.strip().upper())

vista_disp = st.sidebar.radio(
    "Disponibilidad / MTBF / MTTR basados en:",
    ["Sistema (TRC+IMP)", "Tractor", "Implemento"],
    index=0,
)

# =========================================================
# APLICAR FILTROS FINAL
# =========================================================
df_f = apply_filters(turnos, d1, d2, cultivo_sel, trc_sel, imp_sel, turno_sel, id_proceso_sel)

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
    to_base = to_imp  # siempre implemento como base
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
# TOP 10 EQUIPOS (Implemento): TO, Downtime, Fallas
# =========================================================
st.subheader("Top 10 de equipos (Implemento): TO, Downtime y Fallas")

# TO implemento por equipo
to_imp_eq = horo_sel[horo_sel["TIPO_EQUIPO"] == "IMPLEMENTO"].groupby("ID_EQUIPO", dropna=True)["TO_HORO"].sum().reset_index(name="TO_HR")
to_imp_eq["ID_EQUIPO"] = to_imp_eq["ID_EQUIPO"].astype(str)

# Fallas y DT por equipo (solo implementos)
ev_rank = ev_sel[ev_sel["CATEGORIA_EVENTO"] == "FALLA"].copy()
if not ev_rank.empty:
    ev_rank["DT_HR"] = pd.to_numeric(ev_rank["DT_MIN"], errors="coerce") / 60.0
else:
    ev_rank["DT_HR"] = pd.Series(dtype=float)

imp_ids = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()
ev_rank_imp = ev_rank[ev_rank["ID_EQUIPO_AFECTADO"].astype(str).isin(imp_ids)].copy()

dt_fallas_imp = ev_rank_imp.groupby("ID_EQUIPO_AFECTADO", dropna=True).agg(
    DT_HR=("DT_HR", "sum"),
    FALLAS=("DT_HR", "size")
).reset_index().rename(columns={"ID_EQUIPO_AFECTADO": "ID_EQUIPO"})
dt_fallas_imp["ID_EQUIPO"] = dt_fallas_imp["ID_EQUIPO"].astype(str)

# Merge para tener una base completa
base_imp = to_imp_eq.merge(dt_fallas_imp, on="ID_EQUIPO", how="left").fillna({"DT_HR": 0.0, "FALLAS": 0})

c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Top 10 por TO Implemento (h)")
    d = base_imp.sort_values("TO_HR", ascending=False).head(10).sort_values("TO_HR", ascending=True)
    fig = px.bar(d, x="TO_HR", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="TO (h)", yaxis_title="Equipo")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.caption("Top 10 por Downtime (h)")
    d = base_imp.sort_values("DT_HR", ascending=False).head(10).sort_values("DT_HR", ascending=True)
    fig = px.bar(d, x="DT_HR", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Downtime (h)", yaxis_title="Equipo")
    st.plotly_chart(fig, use_container_width=True)

with c3:
    st.caption("Top 10 por cantidad de fallas")
    d = base_imp.sort_values("FALLAS", ascending=False).head(10).sort_values("FALLAS", ascending=True)
    fig = px.bar(d, x="FALLAS", y="ID_EQUIPO", orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Fallas (n)", yaxis_title="Equipo")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =========================================================
# EVOLUCIÓN POR MES-AÑO (según PROCESO elegido)
# =========================================================
st.subheader("Evolución por mes-año (MTTR y MTBF)")

if id_proceso_sel is None:
    st.info("Selecciona un **Proceso** en el filtro lateral para ver su evolución mensual.")
else:
    base = apply_filters(turnos, d1, d2, cultivo_sel, trc_sel, imp_sel, turno_sel, id_proceso_sel)

    proc_name = proc_map.get(str(id_proceso_sel), f"Proceso {id_proceso_sel}")
    st.caption(f"Proceso seleccionado: **{proc_name}**")

    if base.empty:
        st.info("Con los filtros actuales, no hay turnos para este proceso.")
    else:
        base["MES"] = base["FECHA"].dt.to_period("M").astype(str)
        ids = set(base["ID_TURNO"].astype(str).tolist())

        h = horometros[horometros["ID_TURNO"].astype(str).isin(ids)].copy()
        e = eventos[eventos["ID_TURNO"].astype(str).isin(ids)].copy()

        e = e[e["CATEGORIA_EVENTO"] == "FALLA"].copy()
        e["DT_HR"] = pd.to_numeric(e["DT_MIN"], errors="coerce") / 60.0

        turn_mes = base[["ID_TURNO", "MES"]].copy()
        turn_mes["ID_TURNO"] = turn_mes["ID_TURNO"].astype(str)

        h2 = h.merge(turn_mes, on="ID_TURNO", how="left")
        if vista_disp == "Tractor":
            h2 = h2[h2["TIPO_EQUIPO"] == "TRACTOR"].copy()
        else:
            h2 = h2[h2["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()

        to_mes = h2.groupby("MES", dropna=True)["TO_HORO"].sum().reset_index(name="TO_HR")

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
# TOP 10 POR EQUIPO (MTTR, MTBF, DISP) - según vista
# =========================================================
st.subheader("Top 10 por Equipo (MTTR alto, MTBF bajo, Disponibilidad baja)")

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
    imp_ids2 = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids2)][["ID_EQUIPO", "TO_HR"]]

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
    trc_ids2 = turnos_sel["ID_TRACTOR"].dropna().astype(str).unique().tolist()

    to_trc_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "TRACTOR"].copy()
    to_trc_df["ID_EQUIPO"] = to_trc_df["ID_EQUIPO"].astype(str)
    to_trc_df = to_trc_df[to_trc_df["ID_EQUIPO"].isin(trc_ids2)][["ID_EQUIPO", "TO_HR"]]

    falla_trc = falla_equipo.copy()
    falla_trc["ID_EQUIPO"] = falla_trc["ID_EQUIPO"].astype(str)
    falla_trc = falla_trc[falla_trc["ID_EQUIPO"].isin(trc_ids2)]

    top_df = to_trc_df.merge(falla_trc, on="ID_EQUIPO", how="left").fillna({"DT_FALLA_HR": 0.0, "FALLAS": 0})
    top_df["MTTR_HR"] = np.where(top_df["FALLAS"] > 0, top_df["DT_FALLA_HR"] / top_df["FALLAS"], np.nan)
    top_df["MTBF_HR"] = np.where(top_df["FALLAS"] > 0, top_df["TO_HR"] / top_df["FALLAS"], np.nan)
    top_df["DISP"] = np.where(top_df["TO_HR"] > 0, (top_df["TO_HR"] - top_df["DT_FALLA_HR"]) / top_df["TO_HR"], np.nan)
    return top_df

def build_top_df_implemento():
    imp_ids2 = turnos_sel["ID_IMPLEMENTO"].dropna().astype(str).unique().tolist()

    to_imp_df = to_equipo[to_equipo["TIPO_EQUIPO"] == "IMPLEMENTO"].copy()
    to_imp_df["ID_EQUIPO"] = to_imp_df["ID_EQUIPO"].astype(str)
    to_imp_df = to_imp_df[to_imp_df["ID_EQUIPO"].isin(imp_ids2)][["ID_EQUIPO", "TO_HR"]]

    falla_imp = falla_equipo.copy()
    falla_imp["ID_EQUIPO"] = falla_imp["ID_EQUIPO"].astype(str)
    falla_imp = falla_imp[falla_imp["ID_EQUIPO"].isin(imp_ids2)]

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
