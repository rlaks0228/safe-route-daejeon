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
# 1. 그래프 로드 (ZIP → GraphML) + 시간대별 cost, 지표 기준 계산
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

    # cost 계산 + lamp/cctv/child 값 수집 (지표 기준 계산용)
    lamp_vals, cctv_vals, child_vals = [], [], []

    for u, v, k, data in G.edges(keys=True, data=True):
        lamp = float(data.get("lamp", 0.0))
        cctv = float(data.get("cctv", 0.0))
        child = float(data.get("child", 0.0))
        acc = float(data.get("acc", 0.0))

        safe = wL * lamp + wC * cctv + wZ * child
        risk = (1 + wA * acc) / (1 + safe)

        data["cost"] = float(risk)

        lamp_vals.append(lamp)
        cctv_vals.append(cctv)
        child_vals.append(child)

    # 4) 최근접 노드 계산용 노드 GeoDataFrame
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    nodes_proj = nodes.to_crs(5181)

   lamp_vals_arr = np.array(lamp_vals)
cctv_vals_arr = np.array(cctv_vals)
child_vals_arr = np.array(child_vals)

# 0이 아닌 값만 따로 뽑아서 분위수 계산
lamp_pos = lamp_vals_arr[lamp_vals_arr > 0]
cctv_pos = cctv_vals_arr[cctv_vals_arr > 0]
child_pos = child_vals_arr[child_vals_arr > 0]

# 가로등/ CCTV: 값이 있는 edge들 중 하위 20%를 "취약" 기준으로
if len(lamp_pos) > 0:
    lamp_dark_thresh = float(np.quantile(lamp_pos, 0.2))
else:
    lamp_dark_thresh = 0.0

if len(cctv_pos) > 0:
    cctv_low_thresh = float(np.quantile(cctv_pos, 0.2))
else:
    cctv_low_thresh = 0.0

# 보호구역: 값이 있는 edge들 중 상위 20%를 "인근"으로
if len(child_pos) > 0:
    child_high_thresh = float(np.quantile(child_pos, 0.8))
else:
    child_high_thresh = 1.0

    return G, nodes, nodes_proj, lamp_dark_thresh, cctv_low_thresh, child_high_thresh


G, nodes, nodes_proj, lamp_dark_thresh, cctv_low_thresh, child_high_thresh = load_graph_and_scores()


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
# 3. 경로별 지표 계산 함수
# ----------------------------------------------------

def compute_route_stats(
    G: nx.MultiDiGraph,
    route: list[int],
    lamp_dark_thresh: float,
    cctv_low_thresh: float,
    child_high_thresh: float,
):
    """
    한 경로에 대해:
      - 총 길이 (m)
      - 사고 위험 노출도 (acc 길이 가중 평균)
      - 어두운 구간 비율 (lamp 하위 20%)
      - CCTV 취약 구간 비율 (cctv 하위 20%)
      - 어린이 보호구역 인근 비율 (child 상위 20%)
    를 계산해서 dict로 반환.
    """
    total_len = 0.0
    acc_weighted_sum = 0.0
    dark_len = 0.0
    lowcctv_len = 0.0
    child_len = 0.0

    for u, v in zip(route[:-1], route[1:]):
        # 멀티엣지일 경우, 가장 짧은 엣지 사용
        edge_datas = list(G[u][v].values())
        data = min(edge_datas, key=lambda d: d.get("length", 0.0))

        L = float(data.get("length", 0.0))  # meter
        lamp = float(data.get("lamp", 0.0))
        cctv = float(data.get("cctv", 0.0))
        child = float(data.get("child", 0.0))
        acc = float(data.get("acc", 0.0))

        total_len += L
        acc_weighted_sum += acc * L

        if lamp <= lamp_dark_thresh:
            dark_len += L
        if cctv <= cctv_low_thresh:
            lowcctv_len += L
        if child >= child_high_thresh:
            child_len += L

    if total_len == 0:
        return {
            "length_m": 0.0,
            "acc_exposure": 0.0,
            "dark_ratio": 0.0,
            "lowcctv_ratio": 0.0,
            "child_ratio": 0.0,
        }

    return {
        "length_m": total_len,
        "acc_exposure": acc_weighted_sum / total_len,
        "dark_ratio": dark_len / total_len,
        "lowcctv_ratio": lowcctv_len / total_len,
        "child_ratio": child_len / total_len,
    }


def pct_change(new: float, base: float):
    """(new - base) / base * 100. base가 0이면 None."""
    if base == 0:
        return None
    return (new - base) / base * 100.0


def format_delta(p: float, positive_is_good: bool = False):
    """
    p: 퍼센트 변화율
    positive_is_good:
      - False: 감소가 좋은 경우 (위험/노출)
      - True: 증가가 좋은 경우 (보호구역 비율 등)
    """
    if p is None:
        return "–"

    sign_word = ""
    if positive_is_good:
        sign_word = "증가" if p > 0 else "감소"
    else:
        sign_word = "감소" if p < 0 else "증가"

    return f"{abs(p):.1f}% {sign_word}"


# ----------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------
st.title("🛡️ 대전 안전경로 탐색기")
st.write("가로등·CCTV·어린이보호구역·유성구 사고 데이터를 이용해 시간대별 **안전 경로**를 탐색하고,")
st.write("동일 출발/도착에 대해 **최단 거리 경로와 정량 비교**합니다.")

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
    with st.spinner("경로 탐색 및 비교 중입니다..."):
        try:
            # 1) 좌표 → 노드 매핑
            orig_latlon = geocode_robust(orig_in)
            dest_latlon = geocode_robust(dest_in)

            orig_node = find_nearest_node(orig_latlon[0], orig_latlon[1])
            dest_node = find_nearest_node(dest_latlon[0], dest_latlon[1])

            # 2) 최단 거리 경로 (baseline)
            route_shortest = nx.shortest_path(G, orig_node, dest_node, weight="length")

            # 3) 안전 경로 (우리 모델)
            route_safe = nx.shortest_path(G, orig_node, dest_node, weight="cost")

            # 4) 지도 그리기용 좌표
            path_nodes_short = [G.nodes[n] for n in route_shortest]
            latlons_short = [(d["y"], d["x"]) for d in path_nodes_short]

            path_nodes_safe = [G.nodes[n] for n in route_safe]
            latlons_safe = [(d["y"], d["x"]) for d in path_nodes_safe]

            # 5) 정량 지표 계산
            stats_short = compute_route_stats(
                G, route_shortest,
                lamp_dark_thresh, cctv_low_thresh, child_high_thresh
            )
            stats_safe = compute_route_stats(
                G, route_safe,
                lamp_dark_thresh, cctv_low_thresh, child_high_thresh
            )

            deltas = {
                "distance_pct": pct_change(stats_safe["length_m"], stats_short["length_m"]),
                "acc_exposure_pct": pct_change(stats_safe["acc_exposure"], stats_short["acc_exposure"]),
                "dark_ratio_pct": pct_change(stats_safe["dark_ratio"], stats_short["dark_ratio"]),
                "lowcctv_ratio_pct": pct_change(stats_safe["lowcctv_ratio"], stats_short["lowcctv_ratio"]),
                "child_ratio_pct": pct_change(stats_safe["child_ratio"], stats_short["child_ratio"]),
            }

            st.session_state["route_result"] = {
                "latlons_safe": latlons_safe,
                "latlons_short": latlons_short,
                "orig": orig_latlon,
                "dest": dest_latlon,
                "stats_short": stats_short,
                "stats_safe": stats_safe,
                "deltas": deltas,
            }
        except nx.NetworkXNoPath:
            st.error("출발지와 도착지 사이에 도보 경로를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"경로 탐색 중 오류가 발생했습니다: {e}")


# ----------------------------------------------------
# 5. 지도 표시 + 지표 출력
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

    # 지도 중심: 안전 경로 첫 지점 기준
    center_lat, center_lon = latlons_safe[0]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # (1) 최단 거리 경로 – 회색
    folium.PolyLine(
        latlons_short,
        weight=4,
        opacity=0.7,
        color="gray",
        tooltip="최단 거리 경로",
    ).add_to(m)

    # (2) 안전 경로 – 파란색 (위에 덮어서 강조)
    folium.PolyLine(
        latlons_safe,
        weight=6,
        opacity=0.8,
        color="blue",
        tooltip="안전 경로",
    ).add_to(m)

    # (3) 출발 / 도착
    folium.Marker(orig_latlon, popup="출발지").add_to(m)
    folium.Marker(dest_latlon, popup="도착지").add_to(m)

    st_folium(m, width=900, height=600)

    # ------------------------
    # 정량 지표 표/설명 출력
    # ------------------------
    st.subheader("📊 최단 경로 vs 안전 경로 정량 비교")

    # 단위 변환
    dist_short_km = stats_short["length_m"] / 1000.0
    dist_safe_km = stats_safe["length_m"] / 1000.0

    # 비율 → %
    dark_short_pct = stats_short["dark_ratio"] * 100
    dark_safe_pct = stats_safe["dark_ratio"] * 100
    lowcctv_short_pct = stats_short["lowcctv_ratio"] * 100
    lowcctv_safe_pct = stats_safe["lowcctv_ratio"] * 100
    child_short_pct = stats_short["child_ratio"] * 100
    child_safe_pct = stats_safe["child_ratio"] * 100

    # 간단한 표 형태로 정리
    df = pd.DataFrame(
        {
            "지표": [
                "이동 거리 (km)",
                "사고 위험 노출도 (acc, 길이 가중 평균)",
                "어두운 구간 비율 (lamp 하위 20%)",
                "CCTV 취약 구간 비율 (cctv 하위 20%)",
                "보호구역 인근 비율 (child 상위 20%)",
            ],
            "최단 경로": [
                f"{dist_short_km:.2f}",
                f"{stats_short['acc_exposure']:.3f}",
                f"{dark_short_pct:.1f}%",
                f"{lowcctv_short_pct:.1f}%",
                f"{child_short_pct:.1f}%",
            ],
            "안전 경로": [
                f"{dist_safe_km:.2f}",
                f"{stats_safe['acc_exposure']:.3f}",
                f"{dark_safe_pct:.1f}%",
                f"{lowcctv_safe_pct:.1f}%",
                f"{child_safe_pct:.1f}%",
            ],
            "변화율 (최단 → 안전)": [
                format_delta(deltas["distance_pct"], positive_is_good=False),
                format_delta(deltas["acc_exposure_pct"], positive_is_good=False),
                format_delta(deltas["dark_ratio_pct"], positive_is_good=False),
                format_delta(deltas["lowcctv_ratio_pct"], positive_is_good=False),
                format_delta(deltas["child_ratio_pct"], positive_is_good=True),
            ],
        }
    )

    st.dataframe(df, use_container_width=True)

    st.markdown(
        """
        - **이동 거리**: 안전 경로가 얼마나 더 걷는지 / 덜 걷는지  
        - **사고 위험 노출도**: 교통사고 기반 acc 점수를 길이로 가중 평균한 값  
        - **어두운 구간 비율**: 전체 경로 중 조명이 상대적으로 부족한 구간 비율  
        - **CCTV 취약 구간 비율**: CCTV 밀도가 낮은 구간 비율  
        - **보호구역 인근 비율**: 어린이 보호구역·학교 인근을 따라 걷는 비율 (높을수록 좋음)
        """
    )

else:
    st.info("출발지와 도착지를 입력하고 **[✅ 안전 경로 찾기]** 버튼을 눌러 주세요.")

