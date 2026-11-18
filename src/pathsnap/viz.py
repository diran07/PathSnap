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


# --------------------------------------------------------------------------- light "logistics" style
_LIGHT_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body { font-family: 'DM Sans', system-ui, sans-serif; }
  .ps-light-tiles { filter: grayscale(1) brightness(1.06) contrast(.82); }
  .leaflet-container { background: #e9eaec; font-family: 'DM Sans', system-ui, sans-serif; }
  .leaflet-control-zoom, .leaflet-control-layers, .leaflet-control-zoom-fullscreen {
      border: 0 !important; border-radius: 14px !important; box-shadow: 0 4px 18px rgba(0,0,0,.10) !important; }
  .leaflet-control-zoom a { width: 38px !important; height: 38px !important; line-height: 38px !important; }
  .ps-title { position: absolute; top: 18px; left: 22px; z-index: 900; display: flex; align-items: center;
              gap: 10px; pointer-events: none; }
  .ps-title .t { font-size: 34px; font-weight: 600; letter-spacing: -.02em; color: #2b2b2b;
                 text-shadow: 0 0 10px #f2f2f2, 0 0 4px #f2f2f2; }
  .ps-title .s { font-size: 13px; color: #6b7280; margin-top: 2px; }
  .ps-card { position: absolute; left: 16px; right: 16px; bottom: 16px; z-index: 900; background: #fff;
             border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.12); overflow: hidden; }
  .ps-card .top { display: flex; align-items: center; gap: 14px; padding: 16px 22px;
                  border-bottom: 1px solid #eef0f2; }
  .ps-card .id { font-size: 19px; font-weight: 500; color: #111; }
  .ps-card .right { margin-left: auto; font-size: 13px; color: #111; border: 1px solid #d4d4d8;
                    border-radius: 12px; padding: 7px 14px; font-weight: 600; }
  .ps-card .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                   gap: 6px 18px; padding: 14px 22px 18px; }
  .ps-card .k { font-size: 12px; color: #8a8f98; }
  .ps-card .v { font-size: 15px; color: #111; font-weight: 500; margin-top: 2px; }
  .ps-card .u { font-size: 12px; color: #8a8f98; }
  .ps-chip { font-size: 11px; font-weight: 700; letter-spacing: .04em; padding: 4px 9px; border-radius: 7px; }
  .ps-chip.dark { background: #1f1f1f; color: #d9f99d; }
  .ps-chip.lime { background: #d9f99d; color: #1f1f1f; }
  .ps-chip.lav  { background: #e9d5ff; color: #3b0764; }
</style>"""

_PIN_SVG = ('<svg width="30" height="40" viewBox="0 0 24 32"><path d="M12 0C5.4 0 0 5.4 0 12c0 9 12 20 12 20s12-11 '
            '12-20C24 5.4 18.6 0 12 0z" fill="#2b2b2b"/><circle cx="12" cy="12" r="4.5" fill="#fff"/></svg>')


def _light_base(basemap):
    m = folium.Map(tiles=None, zoom_control=False, prefer_canvas=True, scrollWheelZoom=False)
    m.get_root().header.add_child(folium.Element(_LIGHT_CSS))
    layers = {
        "light": folium.TileLayer("OpenStreetMap", name="Light", class_name="ps-light-tiles",
                                  show=basemap == "light"),
        "street": folium.TileLayer("OpenStreetMap", name="Street", show=basemap == "street"),
        "satellite": folium.TileLayer(ESRI_SAT[0], attr=ESRI_SAT[1], name="Satellite", max_zoom=19,
                                      show=basemap == "satellite"),
    }
    for layer in layers.values():
        layer.add_to(m)
    return m


def _finish_light(m, bounds, title_html, card_html):
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    plugins.Fullscreen(position="topright").add_to(m)
    name = m.get_name()
    m.get_root().html.add_child(folium.Element(f"""
{title_html or ''}
{card_html or ''}
<script>
window.addEventListener('load', function() {{
  var map = {name}, bounds = {bounds}, userMoved = false;
  L.control.zoom({{position: 'topright'}}).addTo(map);
  var card = document.querySelector('.ps-card');
  function refit() {{
    if (userMoved) return;
    var s = map.getSize();
    if (s.x < 50 || s.y < 50) return;
    map.invalidateSize();
    var bottom = card ? card.offsetHeight + 40 : 24;   // keep the route clear of the floating card
    map.fitBounds(bounds, {{paddingTopLeft: [40, 90], paddingBottomRight: [70, bottom], animate: false}});
  }}
  map.on('dragstart', function() {{ userMoved = true; }});
  map.getContainer().addEventListener('dblclick', function() {{ userMoved = true; }});
  if (window.ResizeObserver) new ResizeObserver(refit).observe(map.getContainer());
  [100, 500, 1500].forEach(function(t) {{ setTimeout(refit, t); }});
}});
</script>"""))
    return m


def light_map(gps_latlon, predicted, classifier=None, basemap="light", title_html=None, card_html=None,
              max_points=1500):
    """Minimal light map (logistics-app style): dotted GPS trace, solid dark predicted route,
    position-style start marker, pin at the end, title and floating detail card."""
    gps_latlon = np.asarray(gps_latlon)
    m = _light_base(basemap)
    every = max(1, int(np.ceil(len(gps_latlon) / max_points)))
    trace = gps_latlon[::every].tolist()

    fg = folium.FeatureGroup("GPS trace (observed)")
    folium.PolyLine(trace, color="#3f3f46", weight=3, opacity=0.75, dash_array="2 9",
                    line_cap="round").add_to(fg)
    fg.add_to(m)
    if classifier:
        fg = folium.FeatureGroup("Classifier only")
        for line in _chain(classifier):
            folium.PolyLine(line, color="#a855f7", weight=3, opacity=0.8, dash_array="7 7").add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup("Predicted route (HMM)")
    for line in _chain(predicted):
        folium.PolyLine(line, color="#ffffff", weight=8, opacity=0.9).add_to(fg)   # casing
        folium.PolyLine(line, color="#1f1f1f", weight=4, opacity=1).add_to(fg)
    fg.add_to(m)

    folium.CircleMarker(gps_latlon[0].tolist(), radius=14, color="#1f1f1f", weight=0, fill=True,
                        fill_opacity=0.15, tooltip="Start").add_to(m)
    folium.CircleMarker(gps_latlon[0].tolist(), radius=6, color="#ffffff", weight=2, fill=True,
                        fill_color="#1f1f1f", fill_opacity=1, tooltip="Start").add_to(m)
    folium.Marker(gps_latlon[-1].tolist(), tooltip="End",
                  icon=folium.DivIcon(html=_PIN_SVG, icon_size=(30, 40), icon_anchor=(15, 40))).add_to(m)
    lo = np.percentile(gps_latlon, 1, axis=0)
    hi = np.percentile(gps_latlon, 99, axis=0)
    return _finish_light(m, [lo.tolist(), hi.tolist()], title_html, card_html)


def idle_map(start_latlon, end_latlon, basemap="light", title_html=None, card_html=None):
    """Map shown before a trip is reconstructed: start dot and end pin only."""
    m = _light_base(basemap)
    folium.CircleMarker(start_latlon, radius=6, color="#ffffff", weight=2, fill=True,
                        fill_color="#1f1f1f", fill_opacity=1, tooltip="Start").add_to(m)
    folium.Marker(end_latlon, tooltip="End",
                  icon=folium.DivIcon(html=_PIN_SVG, icon_size=(30, 40), icon_anchor=(15, 40))).add_to(m)
    pts = np.array([start_latlon, end_latlon])
    lo, hi = pts.min(0) - 0.01, pts.max(0) + 0.01
    return _finish_light(m, [lo.tolist(), hi.tolist()], title_html, card_html)
