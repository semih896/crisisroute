
from __future__ import annotations

import base64
import heapq
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import plotly.graph_objects as go
import streamlit as st


node_id = str
edge_id = Tuple[node_id, node_id]


@dataclass
class edge:
    u: node_id
    v: node_id
    length: float
    risk: float = 0.0
    road_type: str = "street"
    name: str = ""

    @property
    def cost(self) -> float:
        return self.length + self.risk * 2

    @property
    def id(self) -> edge_id:
        return make_edge_id(self.u, self.v)


def make_edge_id(a: node_id, b: node_id) -> edge_id:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


# small demo graph. it stays here as a safe fallback for offline demo.
demo_nodes: Dict[node_id, Tuple[float, float]] = {
    "Aid Center": (0.8, 3.55),
    "Bridge A": (2.05, 3.55),
    "Main Road": (4.05, 3.35),
    "School Shelter": (5.55, 4.55),
    "Hospital": (7.45, 3.40),
    "Stadium": (5.75, 1.55),
    "Park Shelter": (2.35, 1.55),
    "University Dorm": (1.05, 0.30),
    "Fire Station": (3.75, 0.45),
    "Water Point": (7.35, 0.65),
    "Old Town": (8.75, 1.90),
}

demo_edges: List[edge] = [
    edge("Aid Center", "Bridge A", 2, 0, "bridge", "Rescue Bridge"),
    edge("Aid Center", "Park Shelter", 3, 1, "street", "Depot Street"),
    edge("Aid Center", "University Dorm", 4, 1, "service", "Dorm Service Road"),
    edge("Bridge A", "Main Road", 2, 1, "main", "North Avenue"),
    edge("Bridge A", "School Shelter", 4, 0, "trail", "School Footpath"),
    edge("Bridge A", "Park Shelter", 2, 1, "service", "Park Access Road"),
    edge("Main Road", "School Shelter", 2, 1, "street", "School Street"),
    edge("Main Road", "Hospital", 3, 2, "main", "Hospital Road"),
    edge("Main Road", "Stadium", 3, 0, "main", "Central Boulevard"),
    edge("School Shelter", "Hospital", 3, 1, "street", "Clinic Street"),
    edge("Hospital", "Old Town", 2, 2, "main", "Old Town Road"),
    edge("Stadium", "Old Town", 3, 1, "street", "Market Street"),
    edge("Stadium", "Water Point", 2, 0, "street", "Water Line"),
    edge("Park Shelter", "Stadium", 4, 1, "trail", "Park Trail"),
    edge("Park Shelter", "University Dorm", 2, 0, "trail", "Green Path"),
    edge("University Dorm", "Fire Station", 3, 1, "street", "South Campus Road"),
    edge("Fire Station", "Water Point", 3, 1, "street", "Fireline Road"),
    edge("Fire Station", "Stadium", 3, 1, "service", "Service Link"),
    edge("Water Point", "Old Town", 3, 0, "main", "East Ring Road"),
    edge("Hospital", "Water Point", 4, 2, "service", "Supply Road"),
]

demo_priorities = {
    "School Shelter": 8,
    "Hospital": 9,
    "Stadium": 5,
    "Park Shelter": 4,
    "University Dorm": 6,
    "Fire Station": 3,
    "Water Point": 7,
    "Old Town": 2,
}


def init_state() -> None:
    defaults = {
        "mode": "demo",
        "nodes": demo_nodes.copy(),
        "edges": list(demo_edges),
        "priorities": demo_priorities.copy(),
        "closed_edges": set(),
        "current_node": "Aid Center",
        "destination_node": "Hospital",
        "current_label": "Aid Center",
        "destination_label": "Hospital",
        "route": [],
        "route_cost": math.inf,
        "visited": [],
        "event_log": ["system ready. demo city map loaded."],
        "routing_style": "manual",
        "osm_graph_loaded": False,
        "road_suggestions": [],
        "vision_result": None,
        "last_reason": "initial state",
        "road_names": [],
        "loaded_place": "",
        "show_real_basemap": True,
        "click_sets": "start",
        "last_clicked_point": None,
        "processed_click_key": "",
        "map_click_action": "start",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log(msg: str) -> None:
    st.session_state.event_log.insert(0, msg)
    st.session_state.event_log = st.session_state.event_log[:12]


def build_adj(edges: Sequence[edge], closed: Set[edge_id]) -> Dict[node_id, List[Tuple[node_id, float]]]:
    adj = {n: [] for n in st.session_state.nodes}
    for e in edges:
        if e.id in closed:
            continue
        if e.u not in adj or e.v not in adj:
            continue
        adj[e.u].append((e.v, e.cost))
        adj[e.v].append((e.u, e.cost))
    return adj


def dijkstra(start: node_id, goal: node_id, edges: Sequence[edge], closed: Set[edge_id]):
    adj = build_adj(edges, closed)
    dist = {n: math.inf for n in adj}
    prev: Dict[node_id, Optional[node_id]] = {n: None for n in adj}
    visited: List[node_id] = []

    if start not in adj or goal not in adj:
        return math.inf, [], visited

    dist[start] = 0.0
    pq: List[Tuple[float, node_id]] = [(0.0, start)]
    done: Set[node_id] = set()

    while pq:
        d, cur = heapq.heappop(pq)
        if cur in done:
            continue

        done.add(cur)
        visited.append(cur)

        if cur == goal:
            break

        for nb, w in adj[cur]:
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = cur
                heapq.heappush(pq, (nd, nb))

    if dist[goal] == math.inf:
        return math.inf, [], visited

    path: List[node_id] = []
    cur: Optional[node_id] = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[goal], path, visited


def choose_emergency_target(start: node_id):
    q: List[Tuple[int, float, node_id, List[node_id], List[node_id]]] = []

    for target, pr in st.session_state.priorities.items():
        if pr <= 0 or target == start:
            continue
        dist, path, visited = dijkstra(start, target, st.session_state.edges, st.session_state.closed_edges)
        if path:
            heapq.heappush(q, (-pr, dist, target, path, visited))

    if not q:
        return None, math.inf, [], []

    _, dist, target, path, visited = heapq.heappop(q)
    return target, dist, path, visited


def recalc(reason: str = "route updated") -> None:
    start = st.session_state.current_node

    if st.session_state.routing_style == "emergency":
        target, dist, path, visited = choose_emergency_target(start)
        st.session_state.destination_node = target or ""
    else:
        target = st.session_state.destination_node
        dist, path, visited = dijkstra(start, target, st.session_state.edges, st.session_state.closed_edges)

    st.session_state.route_cost = dist
    st.session_state.route = path
    st.session_state.visited = visited
    st.session_state.last_reason = reason

    if path:
        log(f"{reason}: {' -> '.join(map(str, path))}, cost={dist:.1f}")
    else:
        log(f"{reason}: no available route")


def step_route() -> None:
    if not st.session_state.route:
        recalc("planning before movement")
        return

    if len(st.session_state.route) <= 1:
        log("unit is already at destination")
        return

    next_node = st.session_state.route[1]
    st.session_state.current_node = next_node
    st.session_state.route = st.session_state.route[1:]
    log(f"live location changed: {next_node}")

    if next_node == st.session_state.destination_node:
        log(f"destination reached: {next_node}")


def make_road_name_list(edges: Sequence[edge]) -> List[str]:
    names = sorted({e.name for e in edges if e.name and e.name != "unnamed road"})
    return names[:400]


def close_roads_by_name(names: Sequence[str]) -> int:
    wanted = [clean_text(x) for x in names if x]
    count = 0
    for e in st.session_state.edges:
        e_name = clean_text(e.name)
        if not e_name:
            continue
        if any(w in e_name or e_name in w for w in wanted):
            if e.id not in st.session_state.closed_edges:
                st.session_state.closed_edges.add(e.id)
                count += 1
    return count


def clean_text(x: object) -> str:
    if isinstance(x, list):
        x = " ".join(map(str, x))
    return re.sub(r"\s+", " ", str(x).lower()).strip()


def load_demo() -> None:
    st.session_state.mode = "demo"
    st.session_state.nodes = demo_nodes.copy()
    st.session_state.edges = list(demo_edges)
    st.session_state.priorities = demo_priorities.copy()
    st.session_state.closed_edges = set()
    st.session_state.current_node = "Aid Center"
    st.session_state.destination_node = "Hospital"
    st.session_state.current_label = "Aid Center"
    st.session_state.destination_label = "Hospital"
    st.session_state.osm_graph_loaded = False
    st.session_state.road_names = make_road_name_list(demo_edges)
    log("demo city map loaded")
    recalc("demo map loaded")


def load_osm_graph(place: str, dist_m: int, network_type: str) -> None:
    try:
        import osmnx as ox
    except Exception as exc:
        st.error("osmnx is not installed. install requirements first.")
        st.code("pip install -r requirements.txt")
        raise exc

    with st.spinner("loading real road graph from OpenStreetMap..."):
        G = ox.graph_from_address(place, dist=dist_m, network_type=network_type, simplify=True)

        # osmnx changed this helper in newer versions.
        # v2.x: ox.truncate.largest_component
        # older versions: ox.utils_graph.get_largest_component
        try:
            G = ox.truncate.largest_component(G, strongly=False)
        except AttributeError:
            try:
                G = ox.utils_graph.get_largest_component(G, strongly=False)
            except AttributeError:
                import networkx as nx
                largest_nodes = max(nx.weakly_connected_components(G), key=len)
                G = G.subgraph(largest_nodes).copy()

    nodes: Dict[node_id, Tuple[float, float]] = {}
    for n, data in G.nodes(data=True):
        nodes[str(n)] = (float(data["x"]), float(data["y"]))

    edges: List[edge] = []
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 1.0))
        name = data.get("name", "unnamed road")
        if isinstance(name, list):
            name = name[0] if name else "unnamed road"

        highway = data.get("highway", "street")
        if isinstance(highway, list):
            highway = highway[0]

        road_type = "street"
        if highway in ["motorway", "trunk", "primary", "secondary"]:
            road_type = "main"
        elif highway in ["service", "unclassified"]:
            road_type = "service"
        elif highway in ["footway", "path", "pedestrian", "steps"]:
            road_type = "trail"

        edges.append(edge(str(u), str(v), length, 0, road_type, str(name)))

    if not nodes or not edges:
        st.error("no graph found for this area. try another place or increase distance.")
        return

    # use graph center as default start and destination candidates
    some_nodes = list(nodes.keys())
    st.session_state.nodes = nodes
    st.session_state.edges = edges
    st.session_state.closed_edges = set()
    st.session_state.priorities = {}
    st.session_state.current_node = some_nodes[0]
    st.session_state.destination_node = some_nodes[min(10, len(some_nodes) - 1)]
    st.session_state.current_label = f"nearest node in {place}"
    st.session_state.destination_label = "select destination"
    st.session_state.osm_graph_loaded = True
    st.session_state.mode = "osm"
    st.session_state.loaded_place = place
    st.session_state.road_names = make_road_name_list(edges)
    log(f"real map loaded: {place}, nodes={len(nodes)}, roads={len(edges)}")
    recalc("real road graph loaded")


def dms_to_decimal(deg: str, minute: str, sec: str, hemi: str) -> float:
    val = float(deg) + float(minute) / 60 + float(sec) / 3600
    if hemi.upper() in ["S", "W"]:
        val *= -1
    return val


def parse_lat_lon(text_value: str) -> Optional[Tuple[float, float]]:
    # accepts:
    # 37.065, 37.378
    # 37.065 37.378
    # 37°02'10.24"N 37°19'04.74"E
    raw = text_value.strip()

    dms = re.findall(
        r"(\d+(?:\.\d+)?)\s*°\s*(\d+(?:\.\d+)?)\s*['’]\s*(\d+(?:\.\d+)?)\s*(?:\"|”)?\s*([NSEW])",
        raw,
        flags=re.IGNORECASE,
    )
    if len(dms) >= 2:
        first = dms_to_decimal(*dms[0])
        second = dms_to_decimal(*dms[1])

        first_hemi = dms[0][3].upper()
        second_hemi = dms[1][3].upper()

        if first_hemi in ["N", "S"] and second_hemi in ["E", "W"]:
            return first, second
        if first_hemi in ["E", "W"] and second_hemi in ["N", "S"]:
            return second, first

    parts = re.split(r"[,\s]+", raw)
    parts = [p for p in parts if p]
    if len(parts) != 2:
        return None

    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError:
        return None

    if -90 <= a <= 90 and -180 <= b <= 180:
        return a, b
    return None


def geocode_or_coord(query: str) -> Optional[Tuple[float, float]]:
    coords = parse_lat_lon(query)
    if coords:
        return coords

    try:
        import osmnx as ox
        lat, lon = ox.geocode(query)
        return float(lat), float(lon)
    except Exception:
        return None


def nearest_node_from_latlon(lat: float, lon: float) -> node_id:
    best_node = ""
    best_dist = math.inf

    for n, (x, y) in st.session_state.nodes.items():
        # OSM stores x=longitude, y=latitude
        d = (x - lon) ** 2 + (y - lat) ** 2
        if d < best_dist:
            best_node = n
            best_dist = d

    if not best_node:
        raise ValueError("nearest node not found")
    return best_node


def set_osm_start_dest(start_query: str, dest_query: str) -> None:
    if st.session_state.mode != "osm":
        st.warning("load a real map first.")
        return

    start_coord = geocode_or_coord(start_query)
    dest_coord = geocode_or_coord(dest_query)

    if not start_coord:
        st.warning("start location could not be found. try coordinates like 37.065, 37.378")
        return

    if not dest_coord:
        st.warning("destination could not be found. try coordinates like 37.065, 37.378")
        return

    st.session_state.current_node = nearest_node_from_latlon(*start_coord)
    st.session_state.destination_node = nearest_node_from_latlon(*dest_coord)
    st.session_state.current_label = start_query.strip()
    st.session_state.destination_label = dest_query.strip()
    st.session_state.routing_style = "manual"

    log("start and destination matched to nearest graph nodes")
    recalc("real location route planned")


def analyze_image_with_gemini(uploaded_file, model: str, area_note: str, api_key: str):
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        st.error("google-genai is not installed.")
        st.code("pip install google-genai")
        raise exc

    if not api_key:
        st.error("Gemini API key is missing. add it in the sidebar or set GEMINI_API_KEY.")
        return None

    image_bytes = uploaded_file.getvalue()
    mime = uploaded_file.type or "image/jpeg"

    road_hint = ", ".join(st.session_state.road_names[:80])
    prompt = f"""
You are helping an emergency routing dashboard.

Look at the uploaded aerial/drone/satellite image.
Extract only useful routing information. Return valid JSON only.

Area hint:
{area_note}

Known road names from the loaded map:
{road_hint}

JSON schema:
{{
  "closed_roads": [
    {{"name": "road name if visible or likely", "confidence": 0.0, "reason": "short reason"}}
  ],
  "risk_areas": [
    {{"type": "collapsed_building/flood/fire/debris/unknown", "confidence": 0.0, "reason": "short reason"}}
  ],
  "useful_places": [
    {{"type": "building/park/water/open_area/shelter", "name": "short label", "confidence": 0.0}}
  ],
  "short_summary": "one short paragraph"
}}

Rules:
- If a road name is not clear, put the closest known road name only if there is a visual clue.
- Do not invent many roads. Prefer fewer but more confident suggestions.
- Confidence must be between 0 and 1.
- Output must be JSON only.
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    raw = response.text or ""
    return parse_json_safely(raw)

def parse_json_safely(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"error": "model did not return json", "raw": raw}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {"error": "json parse failed", "raw": raw}


def road_style(t: str):
    if t == "main":
        return "#64748b", 4
    if t == "bridge":
        return "#475569", 5
    if t == "service":
        return "#94a3b8", 2
    if t == "trail":
        return "#b8860b", 2
    return "#94a3b8", 3




def nearest_node_from_xy(lon: float, lat: float) -> node_id:
    best = ""
    best_d = math.inf
    for n, (x, y) in st.session_state.nodes.items():
        d = (x - lon) ** 2 + (y - lat) ** 2
        if d < best_d:
            best = n
            best_d = d
    if not best:
        raise ValueError("nearest graph point not found")
    return best


def draw_clickable_osm_graph():
    try:
        import folium
        from streamlit_folium import st_folium
    except Exception:
        st.error("streamlit-folium is not installed. install requirements first.")
        st.code("pip install streamlit-folium folium")
        return None

    nodes: Dict[node_id, Tuple[float, float]] = st.session_state.nodes
    edges: List[edge] = st.session_state.edges
    center_lat, center_lon, zoom = map_center_and_zoom()
    active_edges = route_edge_ids()
    closed = st.session_state.closed_edges

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(zoom),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # graph layer
    for e in edges:
        if e.u not in nodes or e.v not in nodes:
            continue

        lon0, lat0 = nodes[e.u]
        lon1, lat1 = nodes[e.v]

        is_active = e.id in active_edges
        is_closed = e.id in closed

        if is_closed:
            color = "#991b1b"
            weight = 5
        elif is_active:
            color = "#16a34a"
            weight = 6
        else:
            color, weight = road_style(e.road_type)
            weight = max(2, min(weight, 4))

        folium.PolyLine(
            locations=[(lat0, lon0), (lat1, lon1)],
            color=color,
            weight=weight,
            opacity=0.88 if is_active or is_closed else 0.45,
            tooltip=e.name if e.name else None,
        ).add_to(m)

    # start and destination markers
    if st.session_state.current_node in nodes:
        lon, lat = nodes[st.session_state.current_node]
        folium.Marker(
            [lat, lon],
            popup=f"START<br>{st.session_state.current_label}",
            tooltip="START",
            icon=folium.Icon(color="blue", icon="play"),
        ).add_to(m)

    if st.session_state.destination_node in nodes:
        lon, lat = nodes[st.session_state.destination_node]
        folium.Marker(
            [lat, lon],
            popup=f"DESTINATION<br>{st.session_state.destination_label}",
            tooltip="DESTINATION",
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(m)

    # clicked point marker
    if st.session_state.last_clicked_point:
        lat, lon = st.session_state.last_clicked_point
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#111827",
            fill=True,
            fill_color="#facc15",
            fill_opacity=0.9,
            tooltip="last clicked point",
        ).add_to(m)

    return st_folium(
        m,
        height=650,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="main_osm_click_map",
    )

def route_edge_ids() -> Set[edge_id]:
    if len(st.session_state.route) < 2:
        return set()
    return {make_edge_id(a, b) for a, b in zip(st.session_state.route, st.session_state.route[1:])}


def map_center_and_zoom() -> Tuple[float, float, float]:
    nodes = st.session_state.nodes
    if not nodes:
        return 37.066, 37.377, 15

    lons = [xy[0] for xy in nodes.values()]
    lats = [xy[1] for xy in nodes.values()]
    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    spread = max(max(lons) - min(lons), max(lats) - min(lats))
    if spread < 0.006:
        zoom = 18
    elif spread < 0.015:
        zoom = 17
    elif spread < 0.035:
        zoom = 16
    else:
        zoom = 15

    return center_lat, center_lon, zoom


def draw_real_map_with_tiles() -> go.Figure:
    nodes: Dict[node_id, Tuple[float, float]] = st.session_state.nodes
    edges: List[edge] = st.session_state.edges
    closed = st.session_state.closed_edges
    active_edges = route_edge_ids()

    center_lat, center_lon, zoom = map_center_and_zoom()
    fig = go.Figure()

    # Draw normal roads first. They sit directly on the OpenStreetMap tile layer.
    for e in edges:
        if e.u not in nodes or e.v not in nodes:
            continue

        lon0, lat0 = nodes[e.u]
        lon1, lat1 = nodes[e.v]

        is_closed = e.id in closed
        is_active = e.id in active_edges

        if is_closed:
            color, width = "#991b1b", 5
        elif is_active:
            color, width = "#16a34a", 6
        else:
            color, width = road_style(e.road_type)
            width = max(width, 2)

        fig.add_trace(go.Scattermapbox(
            lon=[lon0, lon1],
            lat=[lat0, lat1],
            mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=(
                f"{e.name}<br>{e.u} - {e.v}<br>"
                f"type={e.road_type}<br>cost={e.cost:.1f}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Start and destination markers. User sees real place labels, not OSM node ids.
    marker_lons, marker_lats, marker_text, marker_color, marker_size = [], [], [], [], []

    if st.session_state.current_node in nodes:
        lon, lat = nodes[st.session_state.current_node]
        marker_lons.append(lon)
        marker_lats.append(lat)
        marker_text.append("START")
        marker_color.append("#1d4ed8")
        marker_size.append(16)

    if st.session_state.destination_node in nodes:
        lon, lat = nodes[st.session_state.destination_node]
        marker_lons.append(lon)
        marker_lats.append(lat)
        marker_text.append("DESTINATION")
        marker_color.append("#dc2626")
        marker_size.append(16)

    if marker_lons:
        fig.add_trace(go.Scattermapbox(
            lon=marker_lons,
            lat=marker_lats,
            mode="markers+text",
            marker=dict(size=marker_size, color=marker_color),
            text=marker_text,
            textposition="top center",
            textfont=dict(size=13, color="#0f172a"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    # Show a few important route road names as small labels.
    used_names: Set[str] = set()
    label_lons, label_lats, label_text = [], [], []

    for e in edges:
        if len(label_text) >= 15:
            break
        if e.id not in active_edges and e.road_type != "main":
            continue
        if not e.name or e.name == "unnamed road" or e.name in used_names:
            continue
        if e.u in nodes and e.v in nodes:
            lon0, lat0 = nodes[e.u]
            lon1, lat1 = nodes[e.v]
            label_lons.append((lon0 + lon1) / 2)
            label_lats.append((lat0 + lat1) / 2)
            label_text.append(e.name[:28])
            used_names.add(e.name)

    if label_text:
        fig.add_trace(go.Scattermapbox(
            lon=label_lons,
            lat=label_lats,
            mode="text",
            text=label_text,
            textfont=dict(size=10, color="#111827"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=45, b=10),
        title=dict(
            text="real map + algorithm graph overlay",
            x=0.02,
            font=dict(size=20, color="#0f172a"),
        ),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
    )
    return fig


def draw_map() -> go.Figure:
    if st.session_state.mode == "osm" and st.session_state.show_real_basemap:
        return draw_real_map_with_tiles()

    nodes: Dict[node_id, Tuple[float, float]] = st.session_state.nodes
    edges: List[edge] = st.session_state.edges
    closed = st.session_state.closed_edges
    route = st.session_state.route

    fig = go.Figure()

    # simple background details for demo mode
    if st.session_state.mode == "demo":
        fig.add_shape(type="rect", x0=1.55, y0=3.05, x1=2.28, y1=4.10,
                      line=dict(width=0), fillcolor="rgba(37,99,235,0.16)", layer="below")
        fig.add_shape(type="rect", x0=1.45, y0=0.82, x1=3.22, y1=2.18,
                      line=dict(width=0), fillcolor="rgba(34,197,94,0.14)", layer="below")
        fig.add_shape(type="circle", x0=5.15, y0=0.95, x1=6.35, y1=2.18,
                      line=dict(width=0), fillcolor="rgba(148,163,184,0.18)", layer="below")

    # roads
    for e in edges:
        if e.u not in nodes or e.v not in nodes:
            continue

        x0, y0 = nodes[e.u]
        x1, y1 = nodes[e.v]

        is_closed = e.id in closed
        color, width = road_style(e.road_type)
        if is_closed:
            color = "#991b1b"
            width = max(width, 4)

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=f"{e.name}<br>{e.u} - {e.v}<br>cost={e.cost:.1f}<extra></extra>",
            showlegend=False,
        ))

    # route overlay
    if len(route) > 1:
        for a, b in zip(route, route[1:]):
            if a in nodes and b in nodes:
                x0, y0 = nodes[a]
                x1, y1 = nodes[b]
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode="lines",
                    line=dict(color="#16a34a", width=6),
                    hovertemplate=f"active route: {a} -> {b}<extra></extra>",
                    showlegend=False,
                ))

    # nodes are intentionally small
    xs, ys, text, colors = [], [], [], []
    for n, (x, y) in nodes.items():
        xs.append(x)
        ys.append(y)
        if st.session_state.mode == "demo":
            text.append(str(n))
        else:
            text.append("")
        if n == st.session_state.current_node:
            colors.append("#1d4ed8")
        elif n == st.session_state.destination_node:
            colors.append("#dc2626")
        else:
            colors.append("#0f172a" if st.session_state.mode == "osm" else "#475569")

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text" if st.session_state.mode == "demo" else "markers",
        marker=dict(size=8 if st.session_state.mode == "osm" else 10,
                    color=colors, line=dict(width=1, color="white")),
        text=text,
        textposition="bottom center",
        textfont=dict(size=9, color="#0f172a"),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # important labels for real maps. numeric OSM node ids are not useful for users,
    # so only start, destination and a few road names are labeled.
    if st.session_state.mode == "osm":
        label_points = []
        if st.session_state.current_node in nodes:
            x, y = nodes[st.session_state.current_node]
            label_points.append((x, y, "START", "#1d4ed8"))
        if st.session_state.destination_node in nodes:
            x, y = nodes[st.session_state.destination_node]
            label_points.append((x, y, "DESTINATION", "#dc2626"))

        for x, y, label, color in label_points:
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=18, color=color, line=dict(width=2, color="white")),
                text=[label],
                textposition="top center",
                textfont=dict(size=12, color=color),
                hovertemplate=f"{label}<extra></extra>",
                showlegend=False,
            ))

        used_names = set()
        label_added = 0
        for e in edges:
            if label_added >= 18:
                break
            if not e.name or e.name == "unnamed road" or e.name in used_names:
                continue
            if e.road_type not in {"main", "street"}:
                continue
            if e.u in nodes and e.v in nodes:
                x0, y0 = nodes[e.u]
                x1, y1 = nodes[e.v]
                fig.add_trace(go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    mode="text",
                    text=[e.name[:28]],
                    textfont=dict(size=9, color="#334155"),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                used_names.add(e.name)
                label_added += 1

    title = "real OpenStreetMap road graph" if st.session_state.mode == "osm" else "demo city response map"
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=45, b=10),
        title=dict(text=title, x=0.02, font=dict(size=20, color="#0f172a")),
        plot_bgcolor="#eef2f7",
        paper_bgcolor="#ffffff",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def css() -> None:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg,#eef3f8 0%,#f8fafc 45%,#eef2f7 100%) !important;
        color:#0f172a !important;
    }

    .block-container {padding-top:1rem; max-width:1500px}

    .stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp span,
    .stApp div[data-testid="stMarkdownContainer"],
    .stApp div[data-testid="stMetricLabel"],
    .stApp div[data-testid="stMetricValue"] {
        color:#0f172a !important;
    }

    div[data-testid="stMetric"] {
        background:#ffffff !important;
        border:1px solid #cbd5e1 !important;
        border-radius:16px !important;
        padding:13px !important;
        box-shadow:0 4px 14px rgba(15,23,42,.08) !important;
    }

    .card {
        background:#ffffff !important;
        border:1px solid #cbd5e1 !important;
        border-radius:16px !important;
        padding:15px 17px !important;
        margin-bottom:13px !important;
        box-shadow:0 4px 14px rgba(15,23,42,.08) !important;
    }

    .card * {color:#0f172a !important}

    .routebox {
        background:#dbeafe !important;
        color:#172554 !important;
        border:1px solid #93c5fd !important;
        border-radius:999px !important;
        padding:6px 10px !important;
        display:inline-block !important;
        margin:3px !important;
        font-size:13px !important;
        font-weight:700 !important;
    }

    section[data-testid="stSidebar"] {
        background:#1e293b !important;
        border-right:1px solid #334155 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color:#f8fafc !important;
    }

    .stTextInput input,.stTextArea textarea {
        background:#ffffff !important;
        color:#0f172a !important;
        border:1px solid #94a3b8 !important;
    }

    div[data-baseweb="select"] > div {
        background:#ffffff !important;
        color:#0f172a !important;
        border:1px solid #94a3b8 !important;
    }

    .stButton>button {
        background:#0f172a !important;
        color:white !important;
        border:1px solid #0f172a !important;
        border-radius:10px !important;
        font-weight:650 !important;
    }

    .stButton>button * {color:white !important}
    .stButton>button:hover {background:#1e293b !important;border-color:#1e293b !important}

    div[data-testid="stExpander"] {
        background:#ffffff !important;
        border:1px solid #cbd5e1 !important;
        border-radius:14px !important;
    }


    pre, code {
        background:#f8fafc !important;
        color:#0f172a !important;
        border:1px solid #cbd5e1 !important;
        border-radius:10px !important;
    }

    .minirow {
        background:#f8fafc !important;
        border:1px solid #cbd5e1 !important;
        border-radius:12px !important;
        padding:9px 11px !important;
        margin:7px 0 !important;
        color:#0f172a !important;
    }

    .minirow * {color:#0f172a !important}

    .legendbox {
        background:#ffffff;
        border:1px solid #cbd5e1;
        border-radius:14px;
        padding:10px 12px;
        margin:8px 0 14px 0;
        box-shadow:0 2px 8px rgba(15,23,42,.05);
        display:flex;
        gap:16px;
        flex-wrap:wrap;
        align-items:center;
    }

    .legendbox span {
        color:#0f172a !important;
        font-size:14px;
        font-weight:600;
    }

    .l {
        display:inline-block;
        width:34px;
        height:5px;
        border-radius:99px;
        margin-right:6px;
        vertical-align:middle;
    }

    .green {background:#16a34a}
    .red {background:#991b1b}
    .gray {background:#64748b}

    .dot {
        display:inline-block;
        width:12px;
        height:12px;
        border-radius:50%;
        margin-right:6px;
        vertical-align:middle;
    }

    .blue {background:#1d4ed8}
    .redDot {background:#dc2626}

    div[data-testid="stExpander"] * {color:#0f172a !important}
    </style>
    """, unsafe_allow_html=True)



def route_road_names(limit: int = 8) -> List[str]:
    if len(st.session_state.route) < 2:
        return []

    names: List[str] = []
    for a, b in zip(st.session_state.route, st.session_state.route[1:]):
        eid = make_edge_id(a, b)
        for e in st.session_state.edges:
            if e.id == eid and e.name and e.name != "unnamed road":
                if e.name not in names:
                    names.append(e.name)
                break

    return names[:limit]


def route_pills() -> str:
    if not st.session_state.route:
        return "<span class='routebox'>no active route</span>"

    if st.session_state.mode == "osm":
        road_count = max(len(st.session_state.route) - 1, 0)
        start_label = st.session_state.current_label or "start"
        dest_label = st.session_state.destination_label or "destination"
        return (
            f"<span class='routebox'>{start_label}</span>"
            f"<span class='routebox'>{road_count} graph steps</span>"
            f"<span class='routebox'>{dest_label}</span>"
        )

    return "".join(f"<span class='routebox'>{x}</span>" for x in st.session_state.route[:12])



def format_confidence(value) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return "?"


def render_vision_result() -> None:
    result = st.session_state.vision_result
    if not result:
        return

    if isinstance(result, dict) and result.get("error"):
        st.error(result.get("error"))
        with st.expander("raw model output"):
            st.code(str(result.get("raw", "")))
        return

    summary = result.get("short_summary", "") if isinstance(result, dict) else ""
    if summary:
        st.write(summary)

    closed = result.get("closed_roads", []) if isinstance(result, dict) else []
    risks = result.get("risk_areas", []) if isinstance(result, dict) else []
    places = result.get("useful_places", []) if isinstance(result, dict) else []

    if closed:
        st.markdown("**suggested closed roads**")
        for item in closed:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "unknown road")
            conf = format_confidence(item.get("confidence"))
            reason = item.get("reason", "")
            st.markdown(
                f"""
                <div class="minirow">
                    <b>{name}</b><br>
                    confidence: {conf}<br>
                    <span>{reason}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if risks:
        st.markdown("**risk areas**")
        for item in risks[:4]:
            if not isinstance(item, dict):
                continue
            st.write(f"- {item.get('type', 'unknown')}  confidence: {format_confidence(item.get('confidence'))}")

    if places:
        st.markdown("**useful places**")
        shown = []
        for item in places[:6]:
            if isinstance(item, dict):
                shown.append(f"{item.get('type', 'place')}: {item.get('name', '')}")
        if shown:
            st.write(", ".join(shown))

    with st.expander("raw JSON"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def map_legend() -> None:
    st.markdown(
        """
        <div class="legendbox">
            <span><i class="l green"></i> active route</span>
            <span><i class="l red"></i> closed road</span>
            <span><i class="l gray"></i> road graph</span>
            <span><i class="dot blue"></i> start</span>
            <span><i class="dot redDot"></i> destination</span>
        </div>
        """,
        unsafe_allow_html=True,
    )



def main() -> None:
    st.set_page_config(page_title="CrisisRoute v4", layout="wide", page_icon="🗺️")
    init_state()
    css()

    st.title("CrisisRoute v4")
    st.caption("emergency routing with dijkstra, priority queue, real map graph and Gemini vision analysis")

    cost = "∞" if math.isinf(st.session_state.route_cost) else f"{st.session_state.route_cost:.1f}"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("current", st.session_state.current_label if st.session_state.mode == "osm" else st.session_state.current_node)
    c2.metric("destination", st.session_state.destination_label if st.session_state.mode == "osm" else (st.session_state.destination_node or "not selected"))
    c3.metric("route cost", cost)
    c4.metric("closed roads", len(st.session_state.closed_edges))

    with st.sidebar:
        st.header("map source")

        if st.button("load demo map", use_container_width=True):
            load_demo()
            st.rerun()

        st.divider()
        st.subheader("real map")
        place = st.text_input("map center", value="Gaziantep University, Gaziantep, Turkey")
        dist_m = st.slider("map radius meter", 300, 2500, 900, 100)
        network_type = st.selectbox("network type", ["drive", "walk", "all"], index=0)

        if st.button("load real road graph", use_container_width=True):
            load_osm_graph(place, dist_m, network_type)
            st.rerun()

        if st.session_state.mode == "osm":
            st.session_state.show_real_basemap = st.checkbox(
                "show real map under graph",
                value=st.session_state.show_real_basemap,
                help="uses OpenStreetMap tiles as the background and draws the algorithm graph on top",
            )

        st.divider()
        st.subheader("routing")
        st.session_state.routing_style = st.radio(
            "route mode",
            ["manual", "emergency"],
            index=0 if st.session_state.routing_style == "manual" else 1,
        )

        if st.session_state.mode == "demo":
            names = list(st.session_state.nodes.keys())
            st.session_state.current_node = st.selectbox(
                "from", names,
                index=names.index(st.session_state.current_node) if st.session_state.current_node in names else 0,
            )
            st.session_state.destination_node = st.selectbox(
                "to", names,
                index=names.index(st.session_state.destination_node) if st.session_state.destination_node in names else 1,
            )
        else:
            st.caption("address search may fail for small campus buildings. coordinates are safer.")
            start_q = st.text_input("from address or lat,lon", value=place)
            dest_q = st.text_input("to address or lat,lon", value="37.0660, 37.3770")
            if st.button("match locations to graph", use_container_width=True):
                set_osm_start_dest(start_q, dest_q)
                st.rerun()

        if st.button("calculate route", use_container_width=True):
            recalc("manual calculation")
            st.rerun()

        s1, s2 = st.columns(2)
        with s1:
            if st.button("step", use_container_width=True):
                step_route()
                st.rerun()
        with s2:
            if st.button("random closure", use_container_width=True):
                open_edges = [e for e in st.session_state.edges if e.id not in st.session_state.closed_edges]
                if open_edges:
                    e = random.choice(open_edges)
                    st.session_state.closed_edges.add(e.id)
                    log(f"road closed: {e.name}")
                    recalc("live replan after closure")
                    st.rerun()

    left, right = st.columns([1.55, 1])

    with left:
        if st.session_state.mode == "osm" and st.session_state.show_real_basemap:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("map selection")
            st.write("choose what your next map click means. click directly on the map to set start or destination.")

            st.session_state.map_click_action = st.radio(
                "map click action",
                ["start", "destination"],
                horizontal=True,
                index=0 if st.session_state.map_click_action == "start" else 1,
            )

            click_data = draw_clickable_osm_graph()

            if click_data and click_data.get("last_clicked"):
                lat = float(click_data["last_clicked"]["lat"])
                lon = float(click_data["last_clicked"]["lng"])
                click_key = f"{lat:.6f},{lon:.6f},{st.session_state.map_click_action}"

                if click_key != st.session_state.processed_click_key:
                    st.session_state.processed_click_key = click_key
                    st.session_state.last_clicked_point = (lat, lon)
                    nearest = nearest_node_from_xy(lon, lat)
                    label = f"{lat:.6f}, {lon:.6f}"

                    if st.session_state.map_click_action == "start":
                        st.session_state.current_node = nearest
                        st.session_state.current_label = label
                        log(f"start selected on map: {label}")
                    else:
                        st.session_state.destination_node = nearest
                        st.session_state.destination_label = label
                        log(f"destination selected on map: {label}")

                    st.session_state.routing_style = "manual"
                    recalc("route updated from map click")
                    st.rerun()

            map_legend()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            box = st.empty()
            box.plotly_chart(draw_map(), use_container_width=True)
            map_legend()

        m1, m2 = st.columns(2)
        with m1:
            if st.button("move one step", use_container_width=True):
                step_route()
                st.rerun()
        with m2:
            if st.button("finish route", use_container_width=True):
                if not st.session_state.route:
                    recalc("route planned before movement")
                safety = 0
                while len(st.session_state.route) > 1 and safety < 500:
                    step_route()
                    safety += 1
                st.rerun()

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("active route")
        st.markdown(route_pills(), unsafe_allow_html=True)
        roads = route_road_names()
        if roads:
            st.write("**route roads:** " + ", ".join(roads))
        st.write(f"last update: {st.session_state.last_reason}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("gemini vision / aerial image")
        st.write("upload a drone or satellite-like image. Gemini suggests blocked roads and useful areas.")

        api_key = st.text_input(
            "Gemini API key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="you can also set GEMINI_API_KEY as environment variable",
        )
        model = st.text_input("vision model", value=os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"))
        area_note = st.text_area("area note", value=st.session_state.loaded_place or "current crisis area")
        img = st.file_uploader("aerial / drone image", type=["png", "jpg", "jpeg", "webp"])

        if img is not None:
            st.image(img, caption="uploaded image", use_container_width=True)

        if st.button("analyze image", use_container_width=True):
            if img is None:
                st.warning("upload an image first.")
            else:
                result = analyze_image_with_gemini(img, model, area_note, api_key)
                st.session_state.vision_result = result
                if result:
                    closed = result.get("closed_roads", [])
                    st.session_state.road_suggestions = closed
                    log("vision analysis completed")
                st.rerun()

        if st.session_state.vision_result:
            render_vision_result()

            names = []
            for item in st.session_state.road_suggestions:
                if isinstance(item, dict) and item.get("name"):
                    label = f"{item.get('name')}  confidence={item.get('confidence', '?')}"
                    names.append((label, item.get("name")))

            if names:
                choices = st.multiselect(
                    "apply suggested closures",
                    options=[x[0] for x in names],
                    default=[x[0] for x in names if isinstance(x[1], str)],
                )
                if st.button("apply selected closures", use_container_width=True):
                    chosen_names = [name for label, name in names if label in choices]
                    cnt = close_roads_by_name(chosen_names)
                    log(f"vision suggestions applied. closed edge count={cnt}")
                    recalc("live replan after vision update")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("manual road closing")
        road_options = st.session_state.road_names[:120]
        selected = st.multiselect("close road by name", road_options)
        if st.button("apply manual road closing", use_container_width=True):
            cnt = close_roads_by_name(selected)
            log(f"manual road closing applied. closed edge count={cnt}")
            recalc("live replan after manual road closing")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("advanced log and explanation"):
        st.markdown("**event log**")
        for item in st.session_state.event_log:
            st.write("• " + item)

        st.markdown(
            """
            - real map mode loads an OpenStreetMap road network and converts it into a graph.
            - dijkstra finds the route on the current graph.
            - priority queue is still available in emergency mode for selecting the most urgent target.
            - Gemini vision reads an uploaded aerial image and suggests blocked roads or useful areas.
            - the user confirms the AI suggestions before graph changes are applied.
            - after every road closure, the route is recalculated.
            """
        )


if __name__ == "__main__":
    main()
