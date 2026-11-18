"""PathSnap evaluator UI (runs on Laptop 2), light logistics-style layout.

Run (from the project root):
    streamlit run ui/app.py
Set the API address in the "⋮" menu (e.g. http://192.168.1.10:8000) or with the
PATHSNAP_API env var. The previous dark UI is kept as ui/app_classic.py.

The UI only receives predictions and metrics from the API. Ground truth never
leaves the API server.
"""
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.viz import idle_map, light_map  # noqa: E402

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="collapsed")

MAP_H = 790
ss = st.session_state
ss.setdefault("api", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000").rstrip("/"))
ss.setdefault("results", {})
ss.setdefault("basemap", "light")
ss.setdefault("show_clf", False)


# --------------------------------------------------------------------------- helpers
def fmt_dur(s):
    s = int(s or 0)
    h, m = divmod(s // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m {s % 60}s"


def fmt_ll(lat, lon):
    return f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'} {abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"


def pct(x):
    return f"{x * 100:.1f}%"


@st.cache_data(ttl=15, show_spinner=False)
def fetch(api):
    t0 = time.perf_counter()
    health = requests.get(f"{api}/health", timeout=5).json()
    try:   # richer list (GPS-only stats); older API versions only have /trips
        trips = requests.get(f"{api}/trips/summary", timeout=20).json()["trips"]
    except Exception:  # noqa: BLE001
        trips = [{"trip_id": t} for t in requests.get(f"{api}/trips", timeout=10).json()["trips"]]
    return health, trips, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- styles
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input { font-family: 'DM Sans', system-ui, sans-serif !important; }
.stApp { background: #e7e8ea; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
.block-container { padding: 14px 16px 6px 16px !important; max-width: 100% !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

.st-key-panel { background: #fff; border-radius: 24px; padding: 20px 18px 16px 18px;
                box-shadow: 0 6px 30px rgba(0,0,0,.06); height: """ + str(MAP_H) + """px !important;
                max-height: """ + str(MAP_H) + """px !important; overflow-y: auto !important;
                overflow-x: hidden; gap: .55rem; flex-wrap: nowrap !important; }
.st-key-panel > div { flex-shrink: 0; }
.st-key-panel::-webkit-scrollbar { width: 6px; } .st-key-panel::-webkit-scrollbar-thumb { background:#e4e4e7; border-radius: 6px; }
.st-key-mapwrap iframe { border-radius: 24px; box-shadow: 0 6px 30px rgba(0,0,0,.06); }

.ps-brand { display:flex; align-items:center; gap:12px; }
.ps-logo { width:40px; height:40px; border-radius:50%; border:2px solid #1f1f1f; display:grid; place-items:center; }
.ps-logo span { width:14px; height:14px; border-radius:50%; background:#1f1f1f; box-shadow: 0 0 0 4px #fff, 0 0 0 6px #1f1f1f; }
.ps-h { font-size: 24px; font-weight: 600; color:#111; letter-spacing:-.01em; }
.ps-sub { font-size: 12.5px; color:#8a8f98; margin-top:-2px; }

/* popover "⋮" button */
.st-key-menu button { border-radius: 14px !important; border: 1px solid #d4d4d8 !important; background:#fff !important;
                      height: 44px; width: 44px; min-width: 44px; padding: 0 !important; }
.st-key-menu button p { font-size: 20px; }

/* segmented control (On the way / Delivered style) */
[data-testid="stButtonGroup"] { width: 100%; }
[data-testid="stButtonGroup"] > div { width: 100%; display: grid !important; grid-template-columns: 1fr 1fr; gap: 6px; }
[data-testid="stButtonGroup"] button { border: 0 !important; border-radius: 16px !important; height: 52px;
                                       background: transparent !important; font-weight: 500; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: #1f1f1f !important; color: #fff !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] p { color:#fff !important; }
[data-testid="stButtonGroup"] button p { font-size: 15px; }

/* trip cards (whole card is a button) */
[class*="st-key-trip_"] button { width: 100%; min-height: 76px; justify-content: flex-start !important; text-align: left;
    background: #f5f6f7 !important; border: 1px solid transparent !important; border-radius: 18px !important;
    padding: 14px 18px !important; position: relative; transition: border-color .15s, background .15s; }
[class*="st-key-trip_"] button:hover { border-color: #d4d4d8 !important; background: #f0f1f3 !important; }
[class*="st-key-trip_"] button > div { width: 100%; justify-content: flex-start !important; }
[class*="st-key-trip_"] button p { font-size: 17px !important; font-weight: 500; color: #111; margin: 0; text-align:left; }
[class*="st-key-trip_"] button p::after { display: block; font-size: 12.5px; font-weight: 400; color: #6b7280; margin-top: 4px; }
[class*="st-key-trip_"] button::after { position: absolute; right: 16px; top: 16px; font-size: 10.5px; font-weight: 700;
    letter-spacing: .04em; padding: 4px 9px; border-radius: 7px; }

/* selected (expanded) card */
.st-key-selcard { border: 1.5px solid #1f1f1f; border-radius: 20px; padding: 16px 18px 12px 18px; gap: .6rem; background:#fff; }
.ps-row { display:flex; align-items:center; gap:10px; }
.ps-ttl { font-size: 18px; font-weight: 600; color:#111; }
.ps-chip { font-size: 10.5px; font-weight: 700; letter-spacing: .04em; padding: 4px 9px; border-radius: 7px; margin-left:auto; }
.ps-chip.lime { background:#d9f99d; color:#1f1f1f; } .ps-chip.lav { background:#e9d5ff; color:#3b0764; }
.ps-chip.grey { background:#eef0f2; color:#52525b; }
.ps-bar { height: 8px; background:#eef0f2; border-radius: 8px; overflow:hidden; }
.ps-bar > div { height:100%; background:#a3e635; border-radius:8px; }
.ps-barlab { display:flex; justify-content:space-between; font-size:12px; color:#6b7280; margin-top:5px; }
.ps-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 4px 10px; }
.ps-k { font-size: 12px; color:#8a8f98; } .ps-v { font-size: 13.5px; color:#111; font-weight:500; margin-top:1px; }
.ps-cmp { display:grid; grid-template-columns: 110px 1fr 52px; gap: 6px 10px; align-items:center; font-size:12.5px; }
.ps-cmp .lab { color:#52525b; } .ps-cmp .num { text-align:right; color:#111; font-weight:600; }
.ps-cmp .bars { display:flex; flex-direction:column; gap:3px; }
.ps-cmp .b { height:6px; border-radius:6px; }
.ps-legend { font-size: 11.5px; color:#6b7280; display:flex; gap:14px; }
.ps-dot { display:inline-block; width:9px; height:9px; border-radius:3px; margin-right:5px; vertical-align:middle; }
.ps-summary { background:#f5f6f7; border-radius: 16px; padding: 10px 14px; font-size: 13px; color:#3f3f46; }
.ps-summary b { color:#111; }

.st-key-selcard button[kind="primary"], .st-key-selcard [data-testid="stBaseButton-primary"] {
    background:#1f1f1f !important; border:0 !important; border-radius:14px !important; height:46px; }
.st-key-selcard button[kind="primary"] p { color:#fff !important; font-weight:600; }
.st-key-selcard [data-testid="stExpander"] details { border: 0 !important; }
.st-key-selcard [data-testid="stExpander"] summary { padding-left: 0 !important; font-size: 13.5px; }
.ps-foot { font-size: 11px; color:#a1a1aa; text-align:center; margin-top: 4px; }
</style>
""")

# --------------------------------------------------------------------------- data
try:
    health, trips, latency = fetch(ss.api)
    online = True
except Exception as e:  # noqa: BLE001
    health, trips, latency, online, err = {}, [], 0, False, e

ids = [t["trip_id"] for t in trips]
info = {t["trip_id"]: t for t in trips}
if ids and ss.get("sel") not in ids:
    ss.sel = ids[0]
results = ss.results
model = (health.get("model") or "?").replace(".joblib", "")


def select(tid):
    ss.sel = tid


def reconstruct(tid):
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{ss.api}/predict", json={"trip_id": tid}, timeout=600)
    except Exception as e:  # noqa: BLE001
        ss.error = f"Request failed: {e}"
        return
    if r.status_code != 200:
        ss.error = f"API error {r.status_code}: {r.text[:300]}"
        return
    res = r.json()
    res["_roundtrip_s"] = time.perf_counter() - t0
    results[tid] = res
    ss.error = None


# dynamic per-card subtitle + status chip (whole card is one button)
card_css = []
for tid in ids:
    t = info[tid]
    sub = (f"{t['n_points']:,} GPS points · {fmt_dur(t['duration_s'])} · {t['gps_distance_km']:.1f} km"
           if "n_points" in t else "GPS trip")
    if tid in results and results[tid].get("metrics"):
        chip, bg, fg = f"TOP-1 {results[tid]['metrics']['hmm']['top1_accuracy'] * 100:.1f}%", "#d9f99d", "#1f1f1f"
    elif tid in results:
        chip, bg, fg = "MATCHED", "#d9f99d", "#1f1f1f"
    else:
        chip, bg, fg = "READY", "#eef0f2", "#52525b"
    card_css.append(f'.st-key-trip_{tid} button p::after{{content:"{sub}";}}'
                    f'.st-key-trip_{tid} button::after{{content:"{chip}";background:{bg};color:{fg};}}')
st.html("<style>" + "".join(card_css) + "</style>")

# --------------------------------------------------------------------------- layout
left, right = st.columns([1, 2.45], gap="small")

with left:
    with st.container(key="panel"):
        h1, h2 = st.columns([5, 1], vertical_alignment="center")
        with h1:
            st.html(f"""<div class="ps-brand"><div class="ps-logo"><span></span></div>
              <div><div class="ps-h">Trips</div>
              <div class="ps-sub">{'● ' if online else '○ '}{'API online · ' + model + f' · {latency:.0f} ms' if online else 'API offline'}</div></div></div>""")
        with h2:
            with st.container(key="menu"):
                with st.popover("⋮"):
                    st.markdown("**Connection**")
                    new_api = st.text_input("API URL", ss.api)
                    if new_api.rstrip("/") != ss.api:
                        ss.api = new_api.rstrip("/")
                        fetch.clear()
                        st.rerun()
                    if st.button("Re-check connection", use_container_width=True):
                        fetch.clear()
                        st.rerun()
                    st.markdown("**Map**")
                    ss.basemap = st.radio("Basemap", ["light", "street", "satellite"], horizontal=True,
                                          format_func=str.title, index=["light", "street", "satellite"].index(ss.basemap))
                    ss.show_clf = st.toggle("Show classifier-only route", value=ss.show_clf)
                    st.caption("🔒 Ground-truth routes never leave the API server.")

        if not online:
            st.error(f"Cannot reach the API at `{ss.api}`")
            st.caption("Same hotspot? API running on Laptop 1 (`run_api.bat`)? Port 8000 allowed "
                       "(`setup_firewall.bat`)? Wi-Fi IP set in the ⋮ menu?")
            with st.expander("Error details"):
                st.code(str(err))
        else:
            view = st.segmented_control("View", ["All trips", "Reconstructed"], default="All trips",
                                        label_visibility="collapsed", key="view") or "All trips"
            done = [t for t in ids if t in results and results[t].get("metrics")]
            if done:
                mean = sum(results[t]["metrics"]["hmm"]["top1_accuracy"] for t in done) / len(done)
                gain = sum(results[t]["metrics"]["hmm"]["top1_accuracy"] - results[t]["metrics"]["classifier"]["top1_accuracy"]
                           for t in done) / len(done)
                st.html(f'<div class="ps-summary">Reconstructed <b>{len(done)}/{len(ids)}</b> · mean Top-1 '
                        f'<b>{pct(mean)}</b> · HMM <b>{gain * 100:+.1f} pts</b> vs classifier</div>')
            shown = ids if view == "All trips" else [t for t in ids if t in results]
            if not shown:
                st.caption("No trips reconstructed yet.")

            for tid in shown:
                if tid != ss.sel:
                    st.button(f"Trip {tid}", key=f"trip_{tid}", on_click=select, args=(tid,),
                              use_container_width=True)
                    continue
                t, res = info[tid], results.get(tid)
                with st.container(key="selcard"):
                    if res and res.get("metrics"):
                        chip = '<span class="ps-chip lime">RECONSTRUCTED</span>'
                    elif res:
                        chip = '<span class="ps-chip lime">MATCHED</span>'
                    else:
                        chip = '<span class="ps-chip grey">READY</span>'
                    st.html(f'<div class="ps-row"><span class="ps-ttl">Trip {tid}</span>{chip}</div>')
                    if res and res.get("metrics"):
                        h = res["metrics"]["hmm"]
                        st.html(f'<div><div class="ps-bar"><div style="width:{h["top1_accuracy"] * 100:.1f}%"></div></div>'
                                f'<div class="ps-barlab"><span>Top-1 accuracy (HMM)</span><b style="color:#111">'
                                f'{pct(h["top1_accuracy"])}</b></div></div>')
                    n_pts = res["n_points"] if res else t.get("n_points")
                    dur = res["summary"]["duration_s"] if res and "summary" in res else t.get("duration_s")
                    dist = res["summary"]["gps_distance_km"] if res and "summary" in res else t.get("gps_distance_km")
                    cells = [("GPS points", f"{n_pts:,}" if n_pts else "–"),
                             ("Duration", fmt_dur(dur) if dur else "–"),
                             ("GPS path", f"{dist:.1f} km" if dist else "–"),
                             ("Model", model)]
                    st.html('<div class="ps-grid">' + "".join(
                        f'<div><div class="ps-k">{k}</div><div class="ps-v">{v}</div></div>' for k, v in cells) + "</div>")
                    st.button("Run again" if res else "Reconstruct route", type="primary", key="go",
                              on_click=reconstruct, args=(tid,), use_container_width=True)
                    if ss.get("error"):
                        st.error(ss.error)
                    if res and res.get("metrics"):
                        with st.expander("Route details and comparison"):
                            c, h = res["metrics"]["classifier"], res["metrics"]["hmm"]
                            rows = ""
                            for key, lab in [("top1_accuracy", "Top-1 accuracy"), ("route_continuity", "Continuity"),
                                             ("route_precision", "Route precision"), ("route_recall", "Route recall"),
                                             ("route_f1", "Route F1")]:
                                rows += (f'<div class="lab">{lab}</div><div class="bars">'
                                         f'<div class="b" style="width:{c[key] * 100:.1f}%;background:#c4c4cc"></div>'
                                         f'<div class="b" style="width:{h[key] * 100:.1f}%;background:#1f1f1f"></div></div>'
                                         f'<div class="num">{pct(h[key])}</div>')
                            sm = res.get("summary", {})
                            st.html('<div class="ps-legend"><span><span class="ps-dot" style="background:#c4c4cc"></span>'
                                    'Classifier only</span><span><span class="ps-dot" style="background:#1f1f1f"></span>'
                                    'Classifier + HMM</span></div><div class="ps-cmp" style="margin-top:8px">' + rows + "</div>")
                            st.html('<div class="ps-grid" style="margin-top:10px">' + "".join(
                                f'<div><div class="ps-k">{k}</div><div class="ps-v">{v}</div></div>' for k, v in [
                                    ("Predicted", f"{sm.get('predicted_length_km', 0):.1f} km"),
                                    ("Segments", f"{sm.get('predicted_segments', 0):,}"),
                                    ("Changed by HMM", pct(sm.get("hmm_changed_points", 0))),
                                    ("Inference", f"{res['inference_seconds']:.2f} s")]) + "</div>")
        st.html('<div class="ps-foot">PathSnap · ML candidate scoring + HMM/Viterbi · UE24CS352A</div>')

with right:
    with st.container(key="mapwrap"):
        if not online or not ids:
            st.html(f'<div style="height:{MAP_H}px;border-radius:24px;background:#dfe1e4;display:grid;place-items:center;'
                    f'color:#6b7280;font-size:15px">Waiting for the API…</div>')
        else:
            tid = ss.sel
            t, res = info[tid], results.get(tid)
            title = (f'<div class="ps-title"><svg width="26" height="34" viewBox="0 0 24 32"><path d="M12 0C5.4 0 0 5.4 0 '
                     f'12c0 9 12 20 12 20s12-11 12-20C24 5.4 18.6 0 12 0z" fill="#2b2b2b"/><circle cx="12" cy="12" r="4.5" '
                     f'fill="#fff"/></svg><div><div class="t">Trip {tid}</div>'
                     f'<div class="s">{"Reconstructed route" if res else "Not reconstructed yet"}</div></div></div>')
            start = res["gps"][0] if res else t.get("start")
            end = res["gps"][-1] if res else t.get("end")
            cols = [("From", fmt_ll(*start) if start else "–", "start of GPS trace"),
                    ("To", fmt_ll(*end) if end else "–", "end of GPS trace")]
            if res and res.get("metrics"):
                h, c = res["metrics"]["hmm"], res["metrics"]["classifier"]
                d = (h["top1_accuracy"] - c["top1_accuracy"]) * 100
                cols += [("Top-1 accuracy", pct(h["top1_accuracy"]), f"{d:+.1f} pts vs classifier"),
                         ("Route continuity", pct(h["route_continuity"]), f"classifier {pct(c['route_continuity'])}"),
                         ("Route F1", pct(h["route_f1"]), "length-weighted"),
                         ("Inference", f"{res['inference_seconds']:.2f} s", f"{res['n_points']:,} points")]
            chip = ('<span class="ps-chip dark">RECONSTRUCTED</span>' if res
                    else '<span class="ps-chip lav">READY</span>')
            card = (f'<div class="ps-card"><div class="top"><span class="id">Trip #{tid}</span>{chip}'
                    f'<span class="right">Model · {model}</span></div><div class="cols">'
                    + "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div><div class="u">{u}</div></div>'
                              for k, v, u in cols) + "</div></div>")
            if res:
                fmap = light_map(res["gps"], res["predicted_route"],
                                 res["classifier_route"] if ss.show_clf else None,
                                 basemap=ss.basemap, title_html=title, card_html=card)
            elif start and end:
                fmap = idle_map(start, end, basemap=ss.basemap, title_html=title, card_html=card)
            else:
                fmap = None
            if fmap is not None:
                components.html(fmap.get_root().render(), height=MAP_H)
            else:
                st.html(f'<div style="height:{MAP_H}px;border-radius:24px;background:#dfe1e4;display:grid;'
                        f'place-items:center;color:#6b7280">Press <b>&nbsp;Reconstruct route&nbsp;</b> to see trip {tid}</div>')
