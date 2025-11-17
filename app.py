# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from shapely.geometry import Point
from datetime import datetime
import pytz
import warnings
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
warnings.filterwarnings("ignore")

st.set_page_config(page_title="대전 안전경로 탐색", layout="wide")

# ----------------------------------------------------
# 0. CSV → GeoDataFrame
# ----------------------------------------------------
def load_point_csv(path):
    import chardet
    with open(path, 'rb') as f:
        enc = chardet.detect(f.read(50000))['encoding']
    # st.write(f"[INFO] {path} 인코딩 감지 → {enc}")

    df = pd.read_csv(path, encoding=enc)

    cols = df.columns

    # 위경도
    lat = next((c for c in cols if "lat" in c.lower() or "위도" in c), None)
    lon = next((c for c in cols if "lon" in c.lower() or "경도" in c), None)
    if lat and lon:
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon], df[lat]),
            crs="EPSG:4326"
        )

    # TM 좌표
    x = next((c for c in cols if c.lower() in ['x','tm_x','tmy_x']), None)
    y = next((c for c in cols if c.lower() in ['y','tm_y','tmy_y']), None)
    if x and y:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[x], df[y]),
            crs="EPSG:5181"
        )
        return gdf.to_crs(4326)

    raise ValueError(f"좌표 열을 찾을 수 없음 → {cols}")

# ----------------------------------------------------
# 1. 데이터 & 그래프 캐싱 로드
# ----------------------------------------------------
@st.cache_resource
def load_graph_and_scores():
    lamps_gdf = load_point_csv("대전광역시_가로등 현황_20221201.csv")
    cctv_gdf  = load_point_csv("daejeon_traffic_cctv.csv")
    child_gdf = load_point_csv("daejeon_child_zone.csv")
    acc_gdf   = load_point_csv("대전광역시 유성구_교통사고 현황_20211231.csv")

    G = ox.graph_from_place("Daejeon, South Korea", network_type="walk")
    edges = ox.graph_to_gdfs(G, nodes=False)

    def safe_score_points(src, edges, buf_m=30):
        e = edges.to_crs(5181).copy()
        p = src.to_crs(5181)

        e["buf"] = e.geometry.buffer(buf_m)
        joined = gpd.sjoin(p, e.set_geometry("buf"), predicate="within", how="left")

        if "index_right" in joined.columns:
            idx = joined["index_right"]
        elif isinstance(joined.index, pd.MultiIndex):
            idx = joined.index.get_level_values(-1)
        else:
            idx = pd.Index([])

        cnt = idx.value_counts().reindex(e.index, fill_value=0)

        lengths = e.geometry.length.clip(lower=1.0)
        density = cnt / (lengths / 100.0)
        p99 = max(np.percentile(density, 99), 1e-6)

        return (density.clip(0, p99) / p99).fillna(0.0)

    edges_5181 = edges.to_crs(5181).copy()
    edges_5181["lamp"]  = safe_score_points(lamps_gdf, edges, 30)
    edges_5181["cctv"]  = safe_score_points(cctv_gdf,  edges, 30)
    edges_5181["child"] = safe_score_points(child_gdf, edges, 50)

    # 유성구 사고
    try:
        yuseong = ox.geocode_to_gdf("Yuseong-gu, Daejeon, South Korea").to_crs(4326)
        yuseong_poly = yuseong.unary_union
    except:
        yuseong_poly = acc_gdf.unary_union.buffer(0.003)

    e = edges_5181.copy()
    e["buf"] = e.geometry.buffer(50)
    joined = gpd.sjoin(acc_gdf.to_crs(5181), e.set_geometry("buf"), predicate="within", how="left")

    if "index_right" in joined.columns:
        idx = joined["index_right"]
    elif isinstance(joined.index, pd.MultiIndex):
        idx = joined.index.get_level_values(-1)
    else:
        idx = pd.Index([])

    acc_cnt = idx.value_counts().reindex(e.index, fill_value=0)
    lengths = e.geometry.length.clip(lower=1.0)
    acc_dens = acc_cnt / (lengths / 100.0)

    inside = e.to_crs(4326).geometry.intersects(yuseong_poly)
    ys_vals = acc_dens[inside]

    if len(ys_vals) > 0:
        p99 = max(np.percentile(ys_vals, 99), 1e-6)
        neutral = np.median(ys_vals[ys_vals > 0]) if (ys_vals > 0).any() else 0
    else:
        p99, neutral = 1, 0

    edges_5181["acc"] = acc_dens.clip(0, p99) / p99
    edges_5181.loc[~inside, "acc"] = neutral / p99

    now = datetime.now(pytz.timezone("Asia/Seoul"))
    night = (now.hour >= 18 or now.hour < 6)

    if night:
        wL, wC, wZ, wA = 1.5, 1.2, 2.0, 1.3
    else:
        wL, wC, wZ, wA = 0.7, 1.0, 1.5, 0.8

    edges_5181["safe"] = (
        wL*edges_5181["lamp"] +
        wC*edges_5181["cctv"] +
        wZ*edges_5181["child"]
    )
    edges_5181["risk"] = (1 + wA*edges_5181["acc"]) / (1 + edges_5181["safe"])

    edges_final = edges_5181.to_crs(4326)

    # 그래프에 cost 등록
    for (u, v, k), r in zip(edges_final.index, edges_final["risk"]):
        G[u][v][k]["cost"] = float(r)

    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    nodes_proj = nodes.to_crs(5181)

    return G, nodes, nodes_proj

G, nodes, nodes_proj = load_graph_and_scores()

# ----------------------------------------------------
# 2. 지오코딩 + 최근접 노드
# ----------------------------------------------------
geocode = Nominatim(user_agent="safe_route").geocode

def is_latlon(s):
    if "," not in s:
        return False
    a,b = s.split(",",1)
    try:
        float(a); float(b)
        return True
    except:
        return False

def geocode_robust(q):
    q = q.strip()
    if is_latlon(q):
        a,b = q.split(",",1)
        return (float(a), float(b))

    loc = geocode(q)
    if loc: return (loc.latitude, loc.longitude)

    loc = geocode(f"{q}, Daejeon, South Korea")
    if loc: return (loc.latitude, loc.longitude)

    gdf = ox.geocode_to_gdf(f"{q}, Daejeon, South Korea")
    if len(gdf):
        c = gdf.geometry.iloc[0].centroid
        return (float(c.y), float(c.x))

    return (36.351, 127.385)

def find_nearest_node(lat, lon):
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(5181).iloc[0]
    dx = nodes_proj.geometry.x - pt.x
    dy = nodes_proj.geometry.y - pt.y
    dist2 = dx*dx + dy*dy
    return dist2.idxmin()

# ----------------------------------------------------
# 3. Streamlit UI
# ----------------------------------------------------
st.title("🛡️ 대전 안전경로 탐색기")
st.write("가로등, CCTV, 어린이보호구역, 유성구 사고 데이터를 이용한 시간대별 안전 경로 탐색")

col1, col2 = st.columns(2)
with col1:
    orig_in = st.text_input("출발지 주소 또는 'lat,lon'", "대전광역시청")
with col2:
    dest_in = st.text_input("도착지 주소 또는 'lat,lon'", "충남대학교")

if st.button("경로 찾기"):
    with st.spinner("경로 탐색 중..."):
        orig_latlon = geocode_robust(orig_in)
        dest_latlon = geocode_robust(dest_in)

        orig_node = find_nearest_node(orig_latlon[0], orig_latlon[1])
        dest_node = find_nearest_node(dest_latlon[0], dest_latlon[1])

        route = nx.shortest_path(G, orig_node, dest_node, weight="cost")

        path_nodes = [G.nodes[n] for n in route]
        latlons = [(d['y'], d['x']) for d in path_nodes]

        m = folium.Map(location=[path_nodes[0]['y'], path_nodes[0]['x']], zoom_start=14)
        folium.PolyLine(latlons, weight=6, opacity=0.7).add_to(m)
        folium.Marker(orig_latlon, popup="출발지").add_to(m)
        folium.Marker(dest_latlon, popup="도착지").add_to(m)

        st_folium(m, width=900, height=600)
