"""PathSnap evaluator UI (runs on Laptop 2).

Run:
    streamlit run ui/app.py
Set the API address in the sidebar (e.g. http://192.168.1.10:8000) or with
the PATHSNAP_API env var.
"""
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.viz import route_map  # noqa: E402

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - Route Reconstruction")

api = st.sidebar.text_input("API URL", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")

if st.sidebar.button("Test connection"):
    try:
        t0 = time.perf_counter()
        h = requests.get(f"{api}/health", timeout=5).json()
        st.sidebar.success(f"Connected - model: {h.get('model')} "
                           f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
    except Exception as e:
        st.sidebar.error(f"Not reachable: {e}")

try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach API at {api}.\n\nChecklist: both laptops on the same network "
             f"(phone hotspot recommended), API running on Laptop 1, port 8000 allowed in "
             f"Laptop 1's firewall, correct IP.\n\nError: {e}")
    st.stop()

trip_id = st.selectbox("Evaluation trip", trips)
if st.button("Reconstruct route", type="primary"):
    with st.spinner("Waiting for API..."):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()
    if r.status_code != 200:
        st.error(r.text)
        st.stop()
    res = r.json()
    st.session_state["res"] = res

res = st.session_state.get("res")
if res:
    st.subheader(f"Trip {res['trip_id']} - {res['n_points']} GPS points, "
                 f"{res['inference_seconds']:.2f}s inference")
    if not res["metrics"]:
        st.info("No reference route on the server for this trip, so accuracy is not available.")
    else:
        cols = st.columns(2)
        for col, kind, label in [(cols[0], "hmm", "Classifier + HMM (final)"),
                                 (cols[1], "classifier", "Classifier only")]:
            m = res["metrics"][kind]
            col.markdown(f"**{label}**")
            col.metric("Top-1 accuracy", f"{m['top1_accuracy']:.2%}")
            col.metric("Route continuity", f"{m['route_continuity']:.2%}")
            col.metric("Route F1 (length-weighted)", f"{m['route_f1']:.2%}")
    fmap = route_map(res["gps"], {
        "Predicted route (HMM)": (res["predicted_route"], "#d62728"),
        "Classifier only": (res["classifier_route"], "#ff7f0e"),
    })
    components.html(fmap.get_root().render(), height=650)
