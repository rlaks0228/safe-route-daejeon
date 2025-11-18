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
# 1. 그래프 로드 (ZIP → GraphML) + cost 계산
# ----------------------------------------------------
@st.cache_resource
def load_graph_and_scores():
    # 1) zip 압축 해제
    zip_path = "daejeon_safe_graph.zip"  # 🔹 이 이름의 zip이 repo 루트에 있어야 함
    extract_dir = "graphdata"

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # 2) graphml 불러오기
    graph_path = os.path.join(extract_dir, "daejeon_safe_graph.graphml")
    G = ox.load_graphml(graph_path)

    # 3) 시간대별 가중치 설정 (야간일수록 안전 요소를 더 강하게)
    now = datetime.now(pytz.timezone("Asia/Seoul"))
    night = (now.hour >= 18 or now.hour < 6)

    if night:
        # 밤: 조명·CCTV·보호구역을 더 강하게 반영
        wL, wC, wZ = 2.0, 2.0, 2.5
    else:
        # 낮: 보호구역 중심, 그래도 조명·CCTV는 반영
        wL, wC, wZ = 1.0, 1.0, 2.0

    # 4) length 분포 수집
    length_vals = []
    edges_info = []  # (u,v,k,length,lamp,cctv,child,acc)

    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))  # meter
        lamp = float(data.get("lamp", 0.0))
        cctv = float(data.get("cctv", 0.0))
        child = float(data.get("child", 0.0))
        acc = float(data.get("acc", 0.0))

        length_vals.append(length)
        edges_info.append((u, v, k, length, lamp, cctv, child, acc))

    # 5) 길이 스케일 (너무 짧은/긴 길 bias 방지)
    if len(length_vals) > 0:
        median_len = float(np.median(length_vals))
        if median_len <= 0:
            median_len = 1.0
    else:
        median_len = 1.0

    # 6) cost 계산
    #    - cost ≈ (길이 / 중앙길이) / (1 + wL*lamp + wC*cctv + wZ*child)
    #    - 조명/ CCTV / 보호구역이 많을수록 cost가 작아져서 선호
    for (u, v, k, length, lamp, cctv, child, acc) in edges_info:
        length_factor = length / median_len
        safe_score = wL * lamp + wC * cctv + wZ * child
        cost = length_factor / (1.0 + safe_score)
        G[u][v][k]["cost"] = float(cost)

    # 7) 최근접 노드 계산용
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    nodes_proj = nodes.to_crs(5181)

    return G, nodes, nodes_proj


G, nodes, nodes_proj = load_graph_and_scores()


# ----------------------------------------------------
# 1-1. 전체 그래프 속성 요약 (사이드바에 표시)
# ----------------------------------------------------
def attr_summary_streamlit(G, attr_name: str):
    vals = [float(d.get(attr_name, 0.0)) for _, _, d in G.edges(data=True)]
    vals = np.array(vals, dtype=float)
    if len(vals) == 0:
        st.sidebar.write(f"{attr_name}: 엣지 없음")
        return
    nonzero = int(np.count_nonzero(vals))
    st.sidebar.write(
        f"{attr_name}: 0이 아닌 엣지 = {nonzero} / 최솟값 = {vals.min():.3f} / 최댓값 = {vals.max():.3f}"
    )

st.sidebar.markdown("### 🔍 전체 그래프 속성 요약")
for name in ["lamp", "cctv", "child", "acc"]:
    attr_summary_streamlit(G, name)


# ----------------------------------------------------
# 2. 지오코딩 + 최근접 노드
# ----------------------------------------------------

def geocode_kakao(q: str):
    """카카오 로컬 검색 API로 q를 검색해서 최상단 결과의 좌표를 반환."""
    try:
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {st.secrets['KAKAO_REST_KEY']}"}
        params = {"query": q, "size": 1}
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

    # 2) 카카오맵 검색
    lat, lon, _ = geocode_kakao(q)
    if lat is not None and lon is not None:
        return lat, lon

    # 3) geopy Nominatim (OSM)
    try:
        loc = geocode(q)
    except Exception:
        loc = None
    if loc:
        return loc.latitude, loc.longitude

    # 4) "Daejeon, South Korea" 붙여서 다시 시도
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
# 3. 경로별 지표 계산 (길이 가중 평균 기반)
# ----------------------------------------------------

def compute_route_stats(G: nx.MultiDiGraph, route: list[int]):
    total_len = 0.0
    acc_sum = 0.0
    lamp_sum = 0.0
    cctv_sum = 0.0
    child_sum = 0.0

    for u, v in zip(route[:-1], route[1:]):
        edge_datas = list(G[u][v].values())
        data = min(edge_datas, key=lambda d: d.get("length", 0.0))

        L = float(data.get("length", 0.0))  # meter
        lamp = float(data.get("lamp", 0.0))
        cctv = float(data.get("cctv", 0.0))
        child = float(data.get("child", 0.0))
        acc = float(data.get("acc", 0.0))

        total_len += L
        acc_sum += acc * L
        lamp_sum += lamp * L
        cctv_sum += cctv * L
        child_sum += child * L

    if total_len == 0:
        return {
            "length_m": 0.0,
            "acc_exposure": 0.0,
            "lamp_mean": 0.0,
            "cctv_mean": 0.0,
            "child_mean": 0.0,
        }

    return {
        "length_m": total_len,
        "acc_exposure": acc_sum / total_len,
        "lamp_mean": lamp_sum / total_len,
        "cctv_mean": cctv_sum / total_len,
        "child_mean": child_sum / total_len,
    }


def pct_change(new: float, base: float):
    if base == 0:
        return None
    return (new - base) / base * 100.0


def format_delta(p: float, positive_is_good: bool):
    if p is None or np.isnan(p):
        return "–"

    if positive_is_good:
        word = "증가" if p > 0 else "감소"
    else:
        word = "감소" if p < 0 else "증가"

    return f"{abs(p):.1f}% {word}"


# ----------------------------------------------------
# 3-1. 특정 경로에서 속성 요약 (사이드바)
# ----------------------------------------------------
def route_attr_summary_streamlit(G, route, label: str):
    vals = {"lamp": [], "cctv": [], "child": [], "acc": []}
    for u, v in zip(route[:-1], route[1:]):
        edge_datas = list(G[u][v].values())
        data = min(edge_datas, key=lambda d: d.get("length", 0.0))
        for k in vals.keys():
            vals[k].append(float(data.get(k, 0.0)))

    st.sidebar.markdown(f"#### 🚶 {label} 경로 속성 요약")
    for k, lst in vals.items():
        arr = np.array(lst, dtype=float)
        if len(arr) == 0:
            st.sidebar.write(f"{k}: 엣지 없음")
            continue
        nonzero = int(np.count_nonzero(arr))
        st.sidebar.write(
            f"{k}: 엣지 수 = {len(arr)}, 0이 아닌 엣지 = {nonzero}, 최댓값 = {arr.max():.3f}"
        )


# ----------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------
st.title("🛡️ 대전 안전경로 탐색기")
st.write("가로등·CCTV·어린이보호구역 데이터를 이용해 시간대별 **안전 경로**를 탐색하고,")
st.write("동일 출발/도착에 대해 **최단 거리 경로와 정량 비교**합니다.")

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
    with st.spinner("경로 탐색 및 비교 중입니다..."):
        try:
            orig_latlon = geocode_robust(orig_in)
            dest_latlon = geocode_robust(dest_in)

            orig_node = find_nearest_node(orig_latlon[0], orig_latlon[1])
            dest_node = find_nearest_node(dest_latlon[0], dest_latlon[1])

            route_shortest = nx.shortest_path(G, orig_node, dest_node, weight="length")
            route_safe = nx.shortest_path(G, orig_node, dest_node, weight="cost")

            latlons_short = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_shortest]
            latlons_safe = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_safe]

            stats_short = compute_route_stats(G, route_shortest)
            stats_safe = compute_route_stats(G, route_safe)

            deltas = {
                "distance_pct": pct_change(stats_safe["length_m"], stats_short["length_m"]),
                "acc_exposure_pct": pct_change(stats_safe["acc_exposure"], stats_short["acc_exposure"]),
                "lamp_mean_pct": pct_change(stats_safe["lamp_mean"], stats_short["lamp_mean"]),
                "cctv_mean_pct": pct_change(stats_safe["cctv_mean"], stats_short["cctv_mean"]),
                "child_mean_pct": pct_change(stats_safe["child_mean"], stats_short["child_mean"]),
            }

            st.session_state["route_result"] = {
                "latlons_safe": latlons_safe,
                "latlons_short": latlons_short,
                "orig": orig_latlon,
                "dest": dest_latlon,
                "stats_short": stats_short,
                "stats_safe": stats_safe,
                "deltas": deltas,
                "route_shortest": route_shortest,
                "route_safe": route_safe,
            }
        except nx.NetworkXNoPath:
            st.error("출발지와 도착지 사이에 도보 경로를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"경로 탐색 중 오류가 발생했습니다: {e}")


# ----------------------------------------------------
# 5. 지도 + 정량 지표 출력
# ----------------------------------------------------
if st.session_state["route_result"] is not None:
    data = st.session_state["route_result"]
    latlons_safe = data["latlons_safe"]
    latlons_short = data["latlons_short"]
    orig_latlon = data["orig"]
    dest_latlon = data["dest"]
    stats_short = data["stats_short"]
    stats_safe = data["stats_safe"]
    deltas = data["deltas"]
    route_shortest = data["route_shortest"]
    route_safe = data["route_safe"]

    center_lat, center_lon = latlons_safe[0]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    folium.PolyLine(
        latlons_short,
        weight=4,
        opacity=0.7,
        color="gray",
        tooltip="최단 거리 경로",
    ).add_to(m)

    folium.PolyLine(
        latlons_safe,
        weight=6,
        opacity=0.8,
        color="blue",
        tooltip="안전 경로",
    ).add_to(m)

    folium.Marker(orig_latlon, popup="출발지").add_to(m)
    folium.Marker(dest_latlon, popup="도착지").add_to(m)

    st_folium(m, width=900, height=600)

    # 👉 이 경로에서 실제로 lamp/cctv/child 엣지를 지나가는지 사이드바에 표시
    route_attr_summary_streamlit(G, route_shortest, "최단")
    route_attr_summary_streamlit(G, route_safe, "안전")

    st.subheader("📊 최단 경로 vs 안전 경로 정량 비교")

    dist_short_km = stats_short["length_m"] / 1000.0
    dist_safe_km = stats_safe["length_m"] / 1000.0

    df = pd.DataFrame(
        {
            "지표": [
                "이동 거리 (km)",
                "사고 위험 노출도 (acc, 길이 가중 평균)",
                "평균 밝기 (lamp, 길이 가중 평균)",
                "평균 CCTV 밀도 (cctv, 길이 가중 평균)",
                "평균 보호구역 점수 (child, 길이 가중 평균)",
            ],
            "최단 경로": [
                f"{dist_short_km:.2f}",
                f"{stats_short['acc_exposure']:.3f}",
                f"{stats_short['lamp_mean']:.3f}",
                f"{stats_short['cctv_mean']:.3f}",
                f"{stats_short['child_mean']:.3f}",
            ],
            "안전 경로": [
                f"{dist_safe_km:.2f}",
                f"{stats_safe['acc_exposure']:.3f}",
                f"{stats_safe['lamp_mean']:.3f}",
                f"{stats_safe['cctv_mean']:.3f}",
                f"{stats_safe['child_mean']:.3f}",
            ],
            "변화율 (최단 → 안전)": [
                format_delta(deltas["distance_pct"], positive_is_good=False),
                format_delta(deltas["acc_exposure_pct"], positive_is_good=False),
                format_delta(deltas["lamp_mean_pct"], positive_is_good=True),
                format_delta(deltas["cctv_mean_pct"], positive_is_good=True),
                format_delta(deltas["child_mean_pct"], positive_is_good=True),
            ],
        }
    )

    st.dataframe(df, use_container_width=True)

    st.markdown(
        """
        - **이동 거리**: 안전 경로가 최단 경로보다 얼마나 더/덜 걷는지  
        - **사고 위험 노출도**: edge별 acc 값을 길이로 가중 평균한 값 (현재 그래프에서는 0으로만 구성됨)  
        - **평균 밝기 / CCTV / 보호구역 점수**: 값이 클수록 청소년에게 더 안전한 환경에 가깝다는 뜻  
        """
    )

else:
    st.info("출발지와 도착지를 입력하고 **[✅ 안전 경로 찾기]** 버튼을 눌러 주세요.")
