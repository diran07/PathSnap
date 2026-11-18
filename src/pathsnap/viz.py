"""Folium map helpers (shared by offline scripts and the UI)."""
import folium
import numpy as np


def arcs_to_lines(nodes_lonlat, arcs, arc_ids):
    """Consecutive distinct arcs -> list of [[lat, lon], [lat, lon]] segments."""
    out, last = [], None
    for a in arc_ids:
        if a < 0 or a == last:
            continue
        u, v = arcs[a]
        out.append([[nodes_lonlat[u, 1], nodes_lonlat[u, 0]], [nodes_lonlat[v, 1], nodes_lonlat[v, 0]]])
        last = a
    return out


def route_map(gps_latlon, layers, network_lines=None, every=1):
    """gps_latlon: (n,2) lat/lon. layers: dict name -> (segments, colour)."""
    gps_latlon = np.asarray(gps_latlon)
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if network_lines:
        fg = folium.FeatureGroup("Road network (nearby)", show=False)
        for seg in network_lines:
            folium.PolyLine(seg, color="#999999", weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup("GPS observations")
    for lat, lon in gps_latlon[::every]:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, opacity=0.7).add_to(fg)
    fg.add_to(m)
    for name, (segs, colour) in layers.items():
        fg = folium.FeatureGroup(name)
        for seg in segs:
            folium.PolyLine(seg, color=colour, weight=5, opacity=0.8).add_to(fg)
        fg.add_to(m)
    folium.Marker(gps_latlon[0].tolist(), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps_latlon[-1].tolist(), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([gps_latlon.min(0).tolist(), gps_latlon.max(0).tolist()])
    return m
