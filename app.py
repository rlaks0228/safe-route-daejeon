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


warnings.filterwarnings("ignore")

st.set_page_config(page_title="대전 안전경로 탐색", layout="wide")


# ----------------------------------------------------
# 1. 그래프 로드 (ZIP → GraphML) + 시간대별 cost 계산
# ----------------------------------------------------
@st.cache_resource
def load_graph_and_scores():
    # 1) zip 압축 해제
    zip_path = "daejeon_safe_graph.zip"
    extract_dir = "graphdata"

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # 2) graphml 불러오기
    graph_path = os.path.join(extract_dir, "daejeon_safe_graph.graphml")
    G = ox.load_graphml(graph_path)

    # 3) 시간대별 가중치로 cost 계산
    now = datetime.now(pytz.timezone("Asia/Seoul"))
    night = (now.hour >= 18 or now.hour < 6)

    if night:
        # 밤: 밝기 / CCTV / 어린이보호구역 / 사고 가중치 ↑
        wL, wC, wZ, wA = 1.5, 1.2, 2.0, 1.3
    else:
        # 낮: 사고보다는 보호구역 중심
        wL, wC, wZ, wA = 0.7, 1.0, 1.5, 0.8

    for u, v, k, data in G.edges(keys=True, data=True):
        lamp  = float(data.get("lamp", 0.0))
        cctv  = float(data.get("cctv", 0.0))
        child = float(data.get("child", 0.0))
        acc   = float(data.get("acc", 0.0))

        safe = wL * lamp + wC * cctv + wZ * child
        risk = (1 + wA * acc) / (1 + safe)

        data["cost"] = float(risk)

    # 4) 최근접 노드 계산용 노드 GeoDataFrame
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    nodes_proj = nodes.to_crs(5181)

    return G, nodes, nodes_proj


G, nodes, nodes_proj = load_graph_and_scores()


# ----------------------------------------------------
# 2. 지오코딩 + 최근접 노드
# ----------------------------------------------------

def geocode_kakao(q: str):
    """카카오 로컬 검색 API로 q를 검색해서 최상단 결과의 좌표를 반환."""
    try:
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {st.secrets['KAKAO_REST_KEY']}"}
        params = {
            "query": q,
            "size": 1,   # 최상단 1개만
        }
        r = requests.get(url, headers=headers, params=params, timeout=3)
        r.raise_for_status()
        data = r.json()
        docs = data.get("documents", [])
        if not docs:
            return None, None, None

        doc = docs[0]
        lat = float(doc["y"])
        lon = float(doc["x"])
        place_name = doc["place_name"]
        return lat, lon, place_name
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

    # 1) "36.35, 127.38" 형태면 바로 숫자로 처리
    if is_latlon(q):
        a, b = q.split(",", 1)
        return float(a), float(b)

    # 2) 카카오맵 검색(한글, 오타, 축약 이름에 강함)
    lat, lon, place_name = geocode_kakao(q)
    if lat is not None and lon is not None:
        # st.toast(f"카카오맵에서 '{place_name}'을(를) 찾았어요.")  # 원하면 사용
        return lat, lon

    # 3) geopy Nominatim (OSM) 시도 – 실패해도 조용히 넘어감
    try:
        loc = geocode(q)
    except Exception:
        loc = None
    if loc:
        return loc.latitude, loc.longitude

    # 4) "대전, 한국" 붙여서 다시 시도
    try:
        loc = geocode(f"{q}, Daejeon, South Korea")
    except Exception:
        loc = None
    if loc:
        return loc.latitude, loc.longitude

    # 5) osmnx geocode_to_gdf – 마지막 시도
    try:
        gdf = ox.geocode_to_gdf(f"{q}, Daejeon, South Korea")
        if len(gdf):
            c = gdf.geometry.iloc[0].centroid
            return float(c.y), float(c.x)
    except Exception:
        pass

    # 6) 완전 실패하면 대전 중심
    return 36.351, 127.385



def find_nearest_node(lat: float, lon: float):
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(5181).iloc[0]
    dx = nodes_proj.geometry.x - pt.x
    dy = nodes_proj.geometry.y - pt.y
    dist2 = dx * dx + dy * dy
    return dist2.idxmin()


# ----------------------------------------------------
# 3. Streamlit UI
# ----------------------------------------------------
st.title("🛡️ 대전 안전경로 탐색기")
st.write("가로등·CCTV·어린이보호구역·유성구 사고 데이터를 이용해 시간대별 안전 경로를 탐색합니다.")

# 이전 경로 결과 보관
if "route_result" not in st.session_state:
    st.session_state["route_result"] = None

col1, col2 = st.columns(2)

with col1:
    orig_in = st.text_input(
        "출발지 (주소 또는 위도,경도)",
        "대전광역시청",
        help='예: "대전광역시 서구 둔산동" 또는 "36.351, 127.385"',
    )

with col2:
    dest_in = st.text_input(
        "도착지 (주소 또는 위도,경도)",
        "충남대학교",
        help='예: "대전광역시 유성구 궁동" 또는 "36.366, 127.343"',
    )

if st.button("✅ 안전 경로 찾기"):
    with st.spinner("경로 탐색 중입니다..."):
        try:
            orig_latlon = geocode_robust(orig_in)
            dest_latlon = geocode_robust(dest_in)

            orig_node = find_nearest_node(orig_latlon[0], orig_latlon[1])
            dest_node = find_nearest_node(dest_latlon[0], dest_latlon[1])

            route = nx.shortest_path(G, orig_node, dest_node, weight="cost")

            path_nodes = [G.nodes[n] for n in route]
            latlons = [(d["y"], d["x"]) for d in path_nodes]

            st.session_state["route_result"] = {
                "path_latlons": latlons,
                "orig": orig_latlon,
                "dest": dest_latlon,
            }
        except nx.NetworkXNoPath:
            st.error("출발지와 도착지 사이에 도보 경로를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"경로 탐색 중 오류가 발생했습니다: {e}")


# ----------------------------------------------------
# 4. 지도 표시
# ----------------------------------------------------
if st.session_state["route_result"] is not None:
    data = st.session_state["route_result"]
    latlons = data["path_latlons"]
    orig_latlon = data["orig"]
    dest_latlon = data["dest"]

    center_lat, center_lon = latlons[0]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    folium.PolyLine(latlons, weight=6, opacity=0.7).add_to(m)
    folium.Marker(orig_latlon, popup="출발지").add_to(m)
    folium.Marker(dest_latlon, popup="도착지").add_to(m)

    st_folium(m, width=900, height=600)
else:
    st.info("출발지와 도착지를 입력하고 **[✅ 안전 경로 찾기]** 버튼을 눌러 주세요.")

