"""PathSnap evaluator UI (runs on Laptop 2).

Run (from the project root, so .streamlit/config.toml is picked up):
    streamlit run ui/app.py
Set the API address in the sidebar (e.g. http://192.168.1.10:8000) or with
the PATHSNAP_API env var.

The UI only receives predictions and metrics from the API. Ground truth never
leaves the API server.
"""
import os
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.viz import showcase_map  # noqa: E402

st.set_page_config(page_title="PathSnap · Route Reconstruction", page_icon="🛰️", layout="wide")

# --------------------------------------------------------------------------- styles
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1730 0%, #0b1020 100%);
                            border-right: 1px solid #1f2a44; }
#MainMenu, footer { visibility: hidden; }

.ps-hero { display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;
           padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
           background: radial-gradient(120% 140% at 0% 0%, rgba(168,85,247,.28) 0%, rgba(56,189,248,.10) 45%, rgba(18,26,47,.9) 75%);
           border: 1px solid #26314f; box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.ps-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; margin:0;
            background: linear-gradient(90deg,#f5f3ff 0%,#c084fc 45%,#38bdf8 100%);
            -webkit-background-clip:text; background-clip:text; color: transparent; }
.ps-sub { color:#94a3b8; margin:.25rem 0 0 0; font-size:.98rem; }
.ps-chip { display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .8rem; border-radius:999px;
           font-size:.82rem; font-weight:600; border:1px solid #26314f; background:#0f172a; color:#cbd5e1; }
.ps-dot { width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
.ps-ok  { background:#22c55e; box-shadow:0 0 10px #22c55e; }
.ps-bad { background:#ef4444; box-shadow:0 0 10px #ef4444; }

.ps-label { color:#94a3b8; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin:.2rem 0 .5rem; }
.ps-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .9rem; margin-bottom: 1rem; }
.ps-card { padding: 1rem 1.1rem; border-radius: 16px; background: #121a2f; border: 1px solid #1f2a44;
           position: relative; overflow: hidden; }
.ps-card::before { content:""; position:absolute; inset:0 0 auto 0; height:3px;
                   background: linear-gradient(90deg,#a855f7,#38bdf8); opacity:.9; }
.ps-card .k { color:#94a3b8; font-size:.8rem; font-weight:600; }
.ps-card .v { font-size:1.9rem; font-weight:800; letter-spacing:-.02em; color:#f8fafc; margin-top:.15rem; }
.ps-card .d { font-size:.8rem; font-weight:600; margin-top:.15rem; }
.ps-up { color:#4ade80; } .ps-down { color:#f87171; } .ps-flat { color:#94a3b8; }

.ps-strip { display:flex; flex-wrap:wrap; gap:.55rem; margin: .2rem 0 1rem; }
.ps-pill { padding:.45rem .8rem; border-radius:12px; background:#0f172a; border:1px solid #1f2a44;
           font-size:.85rem; color:#cbd5e1; }
.ps-pill b { color:#f8fafc; }

.ps-empty { text-align:center; padding: 3.5rem 1rem; border:1px dashed #26314f; border-radius:18px;
            color:#94a3b8; background: rgba(18,26,47,.5); }
.ps-empty h3 { color:#e2e8f0; margin-bottom:.3rem; }
.ps-foot { color:#64748b; font-size:.78rem; margin-top:1.2rem; text-align:center; }

div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg,#9333ea,#6366f1); border:0; border-radius:12px;
    font-weight:700; padding:.6rem 1.2rem; box-shadow: 0 6px 20px rgba(147,51,234,.35); }
div[data-testid="stButton"] button[kind="primary"]:hover { filter: brightness(1.1); }
[data-testid="stButtonGroup"] > div { flex-wrap: wrap !important; overflow: visible !important; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { background:#121a2f; border-radius:10px 10px 0 0; padding:.5rem 1rem; }
</style>
""")

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Connection")
    api = st.text_input("API URL", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")
    st.button("Re-check connection", use_container_width=True)
    st.markdown("### 🗺️ Map")
    basemap = st.radio("Basemap", ["dark", "street", "satellite"], horizontal=True,
                       format_func=str.title)
    show_clf = st.toggle("Show classifier-only route", value=False)
    st.markdown("### 🧠 Pipeline")
    st.caption("1. GPS → local metric projection  \n"
               "2. Candidate road arcs within 50 m  \n"
               "3. ML classifier scores each candidate  \n"
               "4. HMM + Viterbi picks a connected route  \n"
               "5. Accuracy computed on the API server")
    st.caption("🔒 Ground-truth routes never leave the API server.")


@st.cache_data(ttl=10, show_spinner=False)
def check(api_url):
    t0 = time.perf_counter()
    health = requests.get(f"{api_url}/health", timeout=5).json()
    trips = requests.get(f"{api_url}/trips", timeout=10).json()["trips"]
    return health, trips, (time.perf_counter() - t0) * 1000


try:
    health, trips, latency = check(api)
    status = (f'<span class="ps-chip"><span class="ps-dot ps-ok"></span>API online · '
              f'{health.get("model", "?").replace(".joblib", "")} · {latency:.0f} ms</span>')
except Exception as e:  # noqa: BLE001
    health, trips, err = None, [], e
    status = '<span class="ps-chip"><span class="ps-dot ps-bad"></span>API offline</span>'

st.html(f"""
<div class="ps-hero">
  <div>
    <p class="ps-title">🛰️ PathSnap</p>
    <p class="ps-sub">Map-matching &amp; route reconstruction from noisy GPS · ML candidate scoring + HMM/Viterbi decoding</p>
  </div>
  <div>{status}</div>
</div>""")

if health is None:
    st.error(f"Cannot reach the API at `{api}`.")
    st.markdown("- Both laptops on the **same network** (phone hotspot recommended)\n"
                "- API running on Laptop 1 (`run_api.bat`)\n"
                "- Port **8000** allowed in Laptop 1's firewall (`setup_firewall.bat`)\n"
                "- Correct IP in the sidebar (the **Wi-Fi** adapter's IPv4)")
    with st.expander("Error details"):
        st.code(str(err))
    st.stop()

# --------------------------------------------------------------------------- trip picker
st.html('<div class="ps-label">Choose an evaluation trip</div>')
trip_id = st.pills("Evaluation trip", trips, selection_mode="single",
                   default=st.session_state.get("last_trip", trips[0] if trips else None),
                   label_visibility="collapsed")
btn_col, hint_col = st.columns([2, 3], vertical_alignment="center")
with btn_col:
    go = st.button("Reconstruct route ⚡", type="primary", use_container_width=True, disabled=not trip_id)
with hint_col:
    if trip_id:
        st.caption(f"Selected trip **{trip_id}** · {len(trips)} trips available on the API")

log = st.session_state.setdefault("log", {})

if go and trip_id:
    with st.status(f"Reconstructing trip **{trip_id}** …", expanded=False) as s:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
        except Exception as e:  # noqa: BLE001
            s.update(label="Request failed", state="error")
            st.error(f"Request failed: {e}")
            st.stop()
        if r.status_code != 200:
            s.update(label=f"API error {r.status_code}", state="error")
            st.error(r.text)
            st.stop()
        res = r.json()
        res["_roundtrip_s"] = time.perf_counter() - t0
        s.update(label=f"Trip {trip_id} reconstructed in {res['_roundtrip_s']:.2f}s", state="complete")
    st.session_state["res"] = res
    st.session_state["last_trip"] = trip_id
    if res.get("metrics"):
        h, c = res["metrics"]["hmm"], res["metrics"]["classifier"]
        log[trip_id] = {"trip": trip_id, "GPS points": res["n_points"],
                        "Top-1 (HMM)": h["top1_accuracy"], "Top-1 (classifier)": c["top1_accuracy"],
                        "Continuity (HMM)": h["route_continuity"], "Route F1 (HMM)": h["route_f1"],
                        "Inference (s)": res["inference_seconds"]}
    st.toast(f"Trip {trip_id} done", icon="✅")

res = st.session_state.get("res")


def card(label, value, delta=None, delta_label="vs classifier"):
    d = ""
    if delta is not None:
        cls = "ps-up" if delta > 0.0005 else "ps-down" if delta < -0.0005 else "ps-flat"
        arrow = "▲" if delta > 0.0005 else "▼" if delta < -0.0005 else "■"
        d = f'<div class="d {cls}">{arrow} {delta * 100:+.1f} pts {delta_label}</div>'
    return f'<div class="ps-card"><div class="k">{label}</div><div class="v">{value}</div>{d}</div>'


if not res:
    st.html("""<div class="ps-empty"><h3>Pick a trip and press <b>Reconstruct ⚡</b></h3>
               <p>The selected trip is sent to the API, which runs candidate generation, the classifier and
               HMM/Viterbi decoding, then returns the predicted route and its accuracy.</p></div>""")
else:
    sm = res.get("summary", {})
    st.html(f'<div class="ps-label">Trip {res["trip_id"]} · results</div>')
    if res.get("metrics"):
        h, c = res["metrics"]["hmm"], res["metrics"]["classifier"]
        st.html('<div class="ps-grid">'
                + card("Top-1 accuracy", f"{h['top1_accuracy']:.1%}", h["top1_accuracy"] - c["top1_accuracy"])
                + card("Route continuity", f"{h['route_continuity']:.1%}", h["route_continuity"] - c["route_continuity"])
                + card("Route F1 (length-weighted)", f"{h['route_f1']:.1%}", h["route_f1"] - c["route_f1"])
                + card("Inference time", f"{res['inference_seconds']:.2f}s")
                + "</div>")
    else:
        st.info("No reference route on the server for this trip, so accuracy is not available.")

    dur = sm.get("duration_s", 0)
    pills = [("GPS points", f"{res['n_points']:,}"),
             ("Duration", f"{int(dur // 60)} min {int(dur % 60)} s"),
             ("GPS path", f"{sm.get('gps_distance_km', 0):.2f} km"),
             ("Predicted route", f"{sm.get('predicted_length_km', 0):.2f} km"),
             ("Road segments", f"{sm.get('predicted_segments', 0):,}"),
             ("Candidates / point", f"{sm.get('candidates_per_point', 0):.1f}"),
             ("Points changed by HMM", f"{sm.get('hmm_changed_points', 0):.1%}"),
             ("Model", res.get("model", "?").replace(".joblib", ""))]
    st.html('<div class="ps-strip">' + "".join(f'<span class="ps-pill">{k}: <b>{v}</b></span>' for k, v in pills)
            + "</div>")

    t_map, t_cmp, t_log, t_raw = st.tabs(["🗺️ Route map", "📊 Classifier vs HMM", "🧾 Evaluation log", "🔧 Raw response"])
    with t_map:
        fmap = showcase_map(res["gps"], res["predicted_route"],
                            res["classifier_route"] if show_clf else None, basemap=basemap)
        components.html(fmap.get_root().render(), height=640)
        st.caption("🟣 Animated line: predicted route (HMM) · 🔵 dots: GPS observations"
                   + (" · 🟠 dashed: classifier only" if show_clf else "")
                   + " · zoom with +/− or ⛶ fullscreen · layer button (top-right) switches basemaps.")
    with t_cmp:
        if res.get("metrics"):
            rows = []
            for kind, name in [("classifier", "Classifier only"), ("hmm", "Classifier + HMM")]:
                m = res["metrics"][kind]
                for key, lab in [("top1_accuracy", "Top-1 accuracy"), ("route_continuity", "Route continuity"),
                                 ("route_precision", "Route precision"), ("route_recall", "Route recall"),
                                 ("route_f1", "Route F1")]:
                    rows.append({"Metric": lab, "Method": name, "Value": m[key]})
            df = pd.DataFrame(rows)
            methods = ["Classifier only", "Classifier + HMM"]
            metric_order = list(dict.fromkeys(df["Metric"]))
            base = alt.Chart(df).encode(
                y=alt.Y("Metric:N", sort=metric_order, title=None,
                        axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12, labelPadding=8)),
                yOffset=alt.YOffset("Method:N", sort=methods),
                x=alt.X("Value:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                        axis=alt.Axis(format="%", labelColor="#94a3b8", gridColor="#1f2a44")),
                color=alt.Color("Method:N", sort=methods,
                                scale=alt.Scale(domain=methods, range=["#64748b", "#a855f7"]),
                                legend=alt.Legend(orient="top", title=None, labelColor="#cbd5e1")),
                tooltip=["Metric", "Method", alt.Tooltip("Value:Q", format=".2%")])
            bars = base.mark_bar(cornerRadiusEnd=5, height=14)
            labels = base.mark_text(align="left", dx=4, color="#e2e8f0", fontSize=11).encode(
                text=alt.Text("Value:Q", format=".1%"))
            chart = ((bars + labels).properties(height=330)
                     .configure_view(stroke=None).configure(background="transparent"))
            st.altair_chart(chart, use_container_width=True)
            st.caption("The HMM uses road-network connectivity to turn independent per-point guesses "
                       "into a coherent, connected route.")
        else:
            st.info("Metrics are not available for this trip.")
    with t_log:
        if log:
            ldf = pd.DataFrame(list(log.values())).sort_values("trip")
            pct = ["Top-1 (HMM)", "Top-1 (classifier)", "Continuity (HMM)", "Route F1 (HMM)"]
            a, b, c3 = st.columns(3)
            a.metric("Trips evaluated", f"{len(ldf)} / {len(trips)}")
            b.metric("Mean Top-1 (HMM)", f"{ldf['Top-1 (HMM)'].mean():.1%}",
                     f"{(ldf['Top-1 (HMM)'] - ldf['Top-1 (classifier)']).mean() * 100:+.1f} pts vs classifier")
            c3.metric("Mean route F1 (HMM)", f"{ldf['Route F1 (HMM)'].mean():.1%}")
            st.dataframe(ldf, hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.ProgressColumn(k, format="percent", min_value=0, max_value=1)
                                        for k in pct}
                         | {"Inference (s)": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.info("Trips you reconstruct in this session will be listed here.")
    with t_raw:
        st.json({k: v for k, v in res.items() if k not in ("gps", "predicted_arcs", "predicted_route", "classifier_route")})

st.html('<div class="ps-foot">PathSnap · UE24CS352A Machine Learning · predictions served by the FastAPI backend · '
        'ground truth stays server-side</div>')
