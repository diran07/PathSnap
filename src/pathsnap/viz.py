"""Folium map helpers (shared by offline scripts and the UI)."""
import folium
import numpy as np
from folium import plugins

ESRI_SAT = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics")


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


def route_map(gps_latlon, layers, network_lines=None, every=None, max_points=1500):
    """gps_latlon: (n,2) lat/lon. layers: dict name -> (segments, colour).

    GPS markers are thinned to at most `max_points` (display only) so long trips
    stay responsive in the browser; predictions still cover every point.
    """
    gps_latlon = np.asarray(gps_latlon)
    if every is None:
        every = max(1, int(np.ceil(len(gps_latlon) / max_points)))
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if network_lines:
        fg = folium.FeatureGroup("Road network (nearby)", show=False)
        for seg in network_lines:
            folium.PolyLine(seg, color="#999999", weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)
    label = "GPS observations" if every == 1 else f"GPS observations (every {every}th shown)"
    fg = folium.FeatureGroup(label)
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


def _chain(segments):
    """Join consecutive segments into continuous polylines (fewer, smoother lines)."""
    lines, cur = [], []
    for a, b in segments:
        if cur and cur[-1] == a:
            cur.append(b)
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [a, b]
    if len(cur) > 1:
        lines.append(cur)
    return lines


def showcase_map(gps_latlon, predicted, classifier=None, max_points=1200, basemap="dark"):
    """Presentation map for the evaluator UI.

    Layers: animated predicted route, optional classifier-only route, thinned GPS
    points; switchable dark / street / satellite basemaps, fullscreen.
    """
    gps_latlon = np.asarray(gps_latlon)
    every = max(1, int(np.ceil(len(gps_latlon) / max_points)))
    # scrollWheelZoom off: scrolling the page over the map would otherwise zoom the map out
    # (use the +/- buttons, double-click or fullscreen to zoom)
    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True, scrollWheelZoom=False)
    # "Dark" = OpenStreetMap tiles with a CSS colour inversion: no extra tile server
    # (third-party dark tile servers can be blocked or require keys on some networks).
    m.get_root().header.add_child(folium.Element(
        "<style>.ps-dark-tiles{filter:invert(1) hue-rotate(200deg) brightness(.92) contrast(.88) saturate(.6);}"
        ".leaflet-container{background:#0b1020;}</style>"))
    tiles = {
        "dark": folium.TileLayer("OpenStreetMap", name="Dark", class_name="ps-dark-tiles",
                                 show=basemap == "dark"),
        "street": folium.TileLayer("OpenStreetMap", name="Street", show=basemap == "street"),
        "satellite": folium.TileLayer(ESRI_SAT[0], attr=ESRI_SAT[1], name="Satellite", max_zoom=19,
                                      show=basemap == "satellite"),
    }
    for layer in tiles.values():   # only the chosen basemap is visible; others via layer control
        layer.add_to(m)

    fg = folium.FeatureGroup("GPS observations" + ("" if every == 1 else f" (1 in {every})"))
    for lat, lon in gps_latlon[::every]:
        folium.CircleMarker([lat, lon], radius=2.2, color="#38bdf8", weight=0,
                            fill=True, fill_opacity=0.75).add_to(fg)
    fg.add_to(m)

    if classifier:
        fg = folium.FeatureGroup("Classifier only", show=False)
        for line in _chain(classifier):
            folium.PolyLine(line, color="#f59e0b", weight=3, opacity=0.85, dash_array="6 6").add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup("Predicted route (HMM)")
    for line in _chain(predicted):
        folium.PolyLine(line, color="#a855f7", weight=9, opacity=0.25).add_to(fg)   # glow
        plugins.AntPath(line, color="#c084fc", pulse_color="#f5f3ff", weight=4,
                        delay=900, dash_array=[14, 22], opacity=0.95).add_to(fg)
    fg.add_to(m)

    for (lat, lon), name, colour in [(gps_latlon[0], "Start", "green"), (gps_latlon[-1], "End", "red")]:
        folium.Marker([lat, lon], tooltip=name,
                      icon=folium.Icon(color=colour, icon="flag" if name == "End" else "play",
                                       prefix="fa")).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    # (no MiniMap plugin: on load it re-synchronised the main map and overrode the zoom)
    folium.LayerControl(collapsed=True).add_to(m)
    # fit to the bulk of the trip: a single GPS glitch should not zoom the map out to a continent
    lo = np.percentile(gps_latlon, 1, axis=0)
    hi = np.percentile(gps_latlon, 99, axis=0)
    bounds = [lo.tolist(), hi.tolist()]
    m.fit_bounds(bounds)
    # When embedded in an iframe (Streamlit), the frame is resized after the map is created,
    # so the initial fit is computed for a tiny map and ends up zoomed far out. Re-fit
    # whenever the container size changes, until the user pans or zooms themselves.
    name = m.get_name()
    m.get_root().html.add_child(folium.Element(f"""
<script>
window.addEventListener('load', function() {{
  var map = {name}, bounds = {bounds}, userMoved = false;
  function refit() {{
    if (userMoved) return;
    var s = map.getSize();
    if (s.x < 50 || s.y < 50) return;
    map.invalidateSize();
    map.fitBounds(bounds, {{padding: [24, 24], animate: false}});
  }}
  map.on('dragstart', function() {{ userMoved = true; }});
  map.getContainer().addEventListener('dblclick', function() {{ userMoved = true; }});
  if (window.ResizeObserver) new ResizeObserver(refit).observe(map.getContainer());
  [100, 500, 1500].forEach(function(t) {{ setTimeout(refit, t); }});
}});
</script>"""))
    return m
