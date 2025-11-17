# precompute_graph.py
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------
# 0. CSV → GeoDataFrame (앱에서 쓰던 거 그대로)
# ----------------------------------------------------
def load_point_csv(path):
    import chardet
    with open(path, 'rb') as f:
        enc = chardet.detect(f.read(50000))['encoding']
    print(f"[INFO] {path} 인코딩 감지 → {enc}")

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
# 1. 엣지별 시설물 밀도 계산 함수 (앱과 동일)
# ----------------------------------------------------
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


def main():
    # ------------------------------------------------
    # 2. CSV 로드 (이미 이름 바꿔둔 버전 기준)
    # ------------------------------------------------
    lamps_gdf = load_point_csv("lamps_daejeon.csv")
    cctv_gdf  = load_point_csv("cctv_daejeon.csv")
    child_gdf = load_point_csv("child_zone_daejeon.csv")
    acc_gdf   = load_point_csv("accident_yuseong.csv")

    print("[INFO] OSM 도로망 불러오는 중...")
    # 필요하면 여기서 행정구 좁히기 (Yuseong-gu 등)
    G = ox.graph_from_place("Daejeon, South Korea", network_type="walk")
    edges = ox.graph_to_gdfs(G, nodes=False)  # 4326

    # ------------------------------------------------
    # 3. 가로등 / CCTV / 보호구역 점수
    # ------------------------------------------------
    print("[INFO] 가로등 / CCTV / 보호구역 점수 계산...")
    edges_5181 = edges.to_crs(5181).copy()
    edges_5181["lamp"]  = safe_score_points(lamps_gdf, edges, 30)
    edges_5181["cctv"]  = safe_score_points(cctv_gdf,  edges, 30)
    edges_5181["child"] = safe_score_points(child_gdf, edges, 50)

    # ------------------------------------------------
    # 4. 사고 위험 점수 (유성구 기준)
    # ------------------------------------------------
    print("[INFO] 유성구 사고 위험 점수 계산...")
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

    # ------------------------------------------------
    # 5. 그래프에 lamp/cctv/child/acc 값만 저장
    #    (safe/risk는 웹에서 시간대에 따라 계산)
    # ------------------------------------------------
    print("[INFO] 그래프에 속성 주입 중...")
    edges_final = edges_5181.to_crs(4326)

    for (u, v, k), row in edges_final[["lamp", "cctv", "child", "acc"]].iterrows():
        data = G[u][v][k]
        data["lamp"]  = float(row["lamp"])
        data["cctv"]  = float(row["cctv"])
        data["child"] = float(row["child"])
        data["acc"]   = float(row["acc"])

    # ------------------------------------------------
    # 6. GraphML 저장
    # ------------------------------------------------
    out_path = "daejeon_safe_graph.graphml"
    ox.save_graphml(G, out_path)
    print(f"[DONE] 그래프 저장 완료 → {out_path}")


if __name__ == "__main__":
    main()
