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
import zipfile
import os
import requests
import chardet

warnings.filterwarnings("ignore")
st.set_page_config(page_title="대전 안전경로 탐색", layout="wide")


# ----------------------------------------------------
# 0. CSV → GeoDataFrame (사고 / CCTV 공용)
# ----------------------------------------------------
def load_point_csv(path: str) -> gpd.GeoDataFrame:
    with open(path, "rb") as f:
        enc = chardet.detect(f.read(50000))["encoding"]
    st.write(f"[INFO] {path} 인코딩 감지 → {enc}")

    df = pd.read_csv(path, encoding=enc)
    cols = df.columns

    # 위경도 열
    lat = next((c for c in cols if "lat" in c.lower() or "위도" in c), None)
    lon = next((c for c in cols if "lon" in c.lower() or "경도" in c), None)
    if lat and lon:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon], df[lat]),
            crs="EPSG:4326",
        )
        return gdf

    # TM 좌표 열
    x = next((c for c in cols if c.lower() in ["x", "tm_x", "tmy_x"]), None)
    y = next((c for c in cols if c.lower() in ["y", "tm_y", "tmy_y"]), None)
    if x and y:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[x], df[y]),
            crs="EPSG:5181",
        )
        return gdf.to_crs(4326)

    raise ValueError(f"좌표 열을 찾을 수 없음 → {cols}")


def clean_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """geometry NaN/Inf 제거 + CRS 4326 보정."""
    if gdf.crs is None:
        # 혹시 모르니 기본값 4326 가정
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[
        np.isfinite(gdf.geometry.x) &
        np.isfinite(gdf.geometry.y)
    ].copy()
    return gdf


# ----------------------------------------------------
# 1. CCTV 없는 사고 지점 찾기
#    (sjoin_nearest + 좌표 필터)
# ----------------------------------------------------
def find_accidents_without_cctv(
    acc_gdf: gpd.GeoDataFrame,
    cctv_gdf: gpd.GeoDataFrame,
    radius_m: float = 50,
) -> gpd.GeoDataFrame:
    """
    사고 지점 중, 반경 radius_m 내에 CCTV가 1개도 없는 지점만 반환 (EPSG:4326).
    """
    acc_gdf = clean_points(acc_gdf)
    cctv_gdf = clean_points(cctv_gdf)

    # 5181로 변환해서 거리 계산 (단위: m)
    acc_5181 = acc_gdf.to_crs(5181)
    cctv_5181 = cctv_gdf.to_crs(5181)

    if len(cctv_5181) == 0:
        print("[WARN] CCTV 포인트가 0개입니다. 모든 사고 지점을 'CCTV 없음'으로 간주합니다.")
        return acc_gdf.copy()

    # 사고 기준으로 가장 가까운 CCTV까지 거리 계산
    joined = gpd.sjoin_nearest(
        acc_5181,
        cctv_5181,
        how="left",
        distance_col="dist",
        max_distance=radius_m,
    )

    # dist가 NaN이면 주변 radius_m 안에 CCTV가 없는 사고
    no_cctv_mask = joined["dist"].isna()
    acc_no_cctv_5181 = acc_5181.loc[no_cctv_mask].copy()
    acc_no_cctv = acc_no_cctv_5181.to_crs(4326)

    print(f"[INFO] 반경 {radius_m}m 이내 CCTV 없는 사고 지점 수: {len(acc_no_cctv)}")
    return acc_no_cctv


@st.cache_resource
def load_accident_layers(radius_m: float = 50) -> gpd.GeoDataFrame:
    """
    - 사고 전체(accident_yuseong.csv)
    - CCTV 반경 radius_m 안에 없는 사고
    둘을 합집합으로 묶어 '빨간 점' 레이어로 반환.
    """
    ACC_PATH = "accident_yuseong.csv"
    CCTV_PATH = "cctv_daejeon.csv"

    acc_gdf_raw = load_point_csv(ACC_PATH)
    cctv_gdf_raw = load_point_csv(CCTV_PATH)

    acc_gdf = clean_points(acc_gdf_raw)
    cctv_gdf = clean_points(cctv_gdf_raw)

    st.write(f"[INFO] 사고 지점 전체 (정제 전): {len(acc_gdf_raw)} → (정제 후): {len(acc_gdf)}")
    st.write(f"[INFO] CCTV 전체 (정제 전): {len(cctv_gdf_raw)} → (정제 후): {len(cctv_gdf)}")

    acc_no_cctv = find_accidents_without_cctv(acc_gdf, cctv_gdf, radius_m=radius_m)

    # 🚩 합집합: 사고 전체 + CCTV 없는 사고 지점
    danger_points = pd.concat([acc_gdf, acc_no_cctv]).drop_duplicates(subset="geometry")
    danger_points = gpd.GeoDataFrame(danger_points, geometry="geometry", crs="EPSG:4326")

    # 🚨 좌표 유효성(∞/NaN) 최종 필터
    danger_points = danger_points[danger_points.geometry.notna()].copy()
    danger_points = danger_points[
        np.isfinite(danger_points.geometry.x) &
        np.isfinite(danger_points.geometry.y)
    ].copy()

    st.write(f"[INFO] 빨간 점(합집합) 총 개수 (유효 좌표만): {len(danger_points)}")

    return danger_points


# ----------------------------------------------------
# 2. 그래프 로드 + 시간대별 cost 계산
# ----------------------------------------------------
@st.cache_resource
def load_graph_and_scores():
    zip_path = "daejeon_safe_graph.zip"
    extract_dir = "graphdata"

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    graph_path = os.path.join(extract_dir, "daejeon_safe_graph.graphml")
    G = ox.load_graphml(graph_path)

    now = datetime.now(pytz.timezone("Asia/Seoul"))
    night = (now.hour >= 18 or now.hour < 6)

    if night:
        # 밤: 밝기 / CCTV / 보호구역 / 사고 가중치 ↑
        wL, wC, wZ, wA = 1.5, 1.2, 2.0, 1.3
    else:
        # 낮: 보호구역 중심
        wL, wC, wZ, wA = 0.7, 1.0, 1.5, 0.8

    for u, v, k, data in G.edges(keys=True, data=True):
        lamp = float(data.get("lamp", 0.0))
        cctv = float(data.get("cctv", 0.0))
        child = float(data.get("child", 0.0))
        acc = float(data.get("acc", 0.0))

        safe = wL * lamp + wC * cctv + wZ * child
        risk = (1 + wA * acc) / (1 + safe)

        data["cost"] = float(risk)

    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    nodes_proj = nodes.to_crs(5181)

    return G, nodes, nodes_proj


G, nodes, nodes_proj = load_graph_and_scores()
danger_points = load_accident_layers(radius_m=50)


# ----------------------------------------------------
# 3. 지오코딩 + 최근접 노드
# ----------------------------------------------------
def geocode_kakao(q: str):
    try:
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {st.secrets['KAKAO_REST_KEY']}"}
        params = {"query": q, "size": 1}
        r = requests.get(url, headers=headers, params=params, timeout=3)
        r.raise_for_status()
        docs = r.json().get("documents", [])
        if not docs:
            return None, None, None
        d = docs[0]
        return float(d["y"]), float(d["x"]), d["place_name"]
    except Exception:
        return None, None, None


geocode = Nominatim(user_agent="safe_route_daejeon", timeout=3).geocode


def is_latlon(s: str) -> bool:
    if "," not in s:
        return False
    a, b = s.split(",", 1)
    try:
        float(a)
        float(b)
        return True
    except Exception:
        return False


def geocode_robust(q: str):
    q = q.strip()

    # 1) "36.35,127.38" 형태
    if is_latlon(q):
        a, b = q.split(",", 1)
        return float(a), float(b)

    # 2) Kakao
    lat, lon, _ = geocode_kakao(q)
    if lat is not None:
        return lat, lon

    # 3) Nominatim
    try:
        loc = geocode(q)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass

    # 4) OSM + 대전 보정
    try:
        gdf = ox.geocode_to_gdf(f"{q}, Daejeon, South Korea")
        if len(gdf):
            c = gdf.geometry.iloc[0].centroid
            return float(c.y), float(c.x)
    except Exception:
        pass

    # 5) 완전 실패 시 대전 중심
    return 36.351, 127.385


def find_nearest_node(lat: float, lon: float):
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(5181).iloc[0]
    dx = nodes_proj.geometry.x - pt.x
    dy = nodes_proj.geometry.y - pt.y
    dist2 = dx * dx + dy * dy
    return dist2.idxmin()


# ----------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------
st.title("🛡️ 대전 안전경로 탐색기")
st.write("가로등·CCTV·어린이보호구역·유성구 사고 데이터를 기반으로 시간대별 안전 경로를 탐색합니다.")
st.write("🔴 지도 위 빨간 점 = **모든 사고 지점 + 반경 50m 내에 CCTV가 없는 사고 지점** (합집합)")

if "route_result" not in st.session_state:
    st.session_state["route_result"] = None

col1, col2 = st.columns(2)
with col1:
    orig_in = st.text_input("출발지 (주소 또는 위도,경도)", "대전광역시청")
with col2:
    dest_in = st.text_input("도착지 (주소 또는 위도,경도)", "충남대학교")

if st.button("✅ 안전 경로 찾기"):
    with st.spinner("경로 탐색 중입니다..."):
        try:
            orig_latlon = geocode_robust(orig_in)
            dest_latlon = geocode_robust(dest_in)

            orig_node = find_nearest_node(*orig_latlon)
            dest_node = find_nearest_node(*dest_latlon)

            route = nx.shortest_path(G, orig_node, dest_node, weight="cost")
            latlons = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

            st.session_state["route_result"] = {
                "latlons": latlons,
                "orig": orig_latlon,
                "dest": dest_latlon,
            }
        except nx.NetworkXNoPath:
            st.error("출발지와 도착지 사이에 도보 경로를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"경로 탐색 중 오류가 발생했습니다: {e}")


# ----------------------------------------------------
# 5. 지도 표시 (경로 + 빨간 점)
# ----------------------------------------------------
if st.session_state["route_result"] is not None:
    data = st.session_state["route_result"]
    latlons = data["latlons"]
    orig_latlon = data["orig"]
    dest_latlon = data["dest"]

    center_lat, center_lon = latlons[0]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 경로 (파란 선)
    folium.PolyLine(latlons, weight=6, opacity=0.7, color="blue").add_to(m)

    # 출발 / 도착
    folium.Marker(orig_latlon, popup="출발지").add_to(m)
    folium.Marker(dest_latlon, popup="도착지").add_to(m)

    # 빨간 점 (사고 전체 + CCTV 없는 사고, 유효 좌표만)
    for pt in danger_points.geometry:
        folium.CircleMarker(
            location=[pt.y, pt.x],
            radius=4,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
        ).add_to(m)

    st_folium(m, width=900, height=600)
else:
    st.info("출발지와 도착지를 입력하고 **[✅ 안전 경로 찾기]** 버튼을 눌러 주세요.")
