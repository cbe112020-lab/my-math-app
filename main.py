import json
import os
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai  # 使用官方最新標準 SDK
from google.genai import types
import neal
import numpy as np
from pyqubo import Array, Constraint
import requests

app = Flask(__name__)
CORS(app)  # 允許 Cross-Origin 跨網域存取

# =====================================================================
# 1. API 金鑰配置 (優先採用寫死 Key，若為空則讀取環境變數)
# =====================================================================
HARDCODED_GEMINI_KEY = "AQ.Ab8RN6KLGY4L7IMd0eX_OW7xqHK8ROwFbTH3A8olaCuoF6GdbQ"
HARDCODED_MAPS_KEY = "AIzaSyDpQflWzh_2ylE2IxkPY5SSkq9ENzQ2L7I"

GEMINI_API_KEY = (
    HARDCODED_GEMINI_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
)
MAPS_API_KEY = (
    HARDCODED_MAPS_KEY.strip() or os.getenv("MAPS_API_KEY", "").strip()
)

# 初始化標準 Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================================================
# 2. API 爬取與工具模組
# =====================================================================
def fetch_city_spots_from_gemini(city_name, spot_count=10):
    prompt = (
        f"請化身為『{city_name}』的在地旅遊專家。\n"
        f"請列出屬於『{city_name}』最著名的 {spot_count} 個旅遊景點、名勝古蹟或觀光景點。\n"
        f"【重要限制】：請『只』返回一個標準 JSON 格式陣列，包含多個物件。每個物件欄位如下：\n"
        "1. `name`: 景點名稱 (純名稱，不要加括號補充)\n"
        "2. `stay_minutes`: 建議停留時間 (整數分鐘，如 60, 90, 120)\n"
        "3. `estimated_cost`: 門票預估消費 (整數新台幣，免費填 0)\n"
        "不要有任何 Markdown 標籤或額外說明文字。\n"
    )

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]

            return json.loads(clean_text.strip())

        except Exception as e:
            print(f"⚠️ Gemini API 第 {attempt} 次呼叫失敗: {e}")
            if attempt < 3:
                time.sleep(2)
            else:
                print("❌ Gemini API 嘗試多次後依然失敗。")
                return None


def get_place_details_from_google(city_name, spot_name):
    search_url = "[https://places.googleapis.com/v1/places:searchText](https://places.googleapis.com/v1/places:searchText)"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.rating,places.userRatingCount"
        ),
    }
    query = (
        f"{city_name} {spot_name}" if city_name not in spot_name else spot_name
    )
    try:
        res = requests.post(
            search_url, headers=headers, json={"textQuery": query}, timeout=5
        )
        result = res.json()
        if "places" in result and result["places"]:
            place_info = result["places"][0]
            return {
                "place_id": place_info.get("id"),
                "rating": float(place_info.get("rating", 4.2)),
                "user_ratings_total": int(
                    place_info.get("userRatingCount", 2000)
                ),
            }
    except Exception as e:
        print(f"Places API 錯誤 [{spot_name}]: {e}")
    return None


def get_distance_and_time_matrices(place_ids):
    n = len(place_ids)
    time_matrix = np.zeros((n, n), dtype=float)
    dist_matrix = np.zeros((n, n), dtype=float)

    valid_place_ids = [
        f"place_id:{pid}" if pid else "高雄火車站" for pid in place_ids
    ]
    url = "[https://maps.googleapis.com/maps/api/distancematrix/json](https://maps.googleapis.com/maps/api/distancematrix/json)"
    origins_destinations = "|".join(valid_place_ids)
    params = {
        "origins": origins_destinations,
        "destinations": origins_destinations,
        "mode": "driving",
        "language": "zh-TW",
        "key": MAPS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        if data.get("status") == "OK":
            rows = data.get("rows", [])
            for i in range(n):
                elements = rows[i].get("elements", [])
                for j in range(n):
                    if i == j:
                        time_matrix[i, j] = 0.0
                        dist_matrix[i, j] = 0.0
                    else:
                        if j < len(elements) and elements[j].get("status") == "OK":
                            duration_sec = (
                                elements[j]
                                .get("duration", {})
                                .get("value", 1200)
                            )
                            time_matrix[i, j] = round(duration_sec / 60.0, 1)
                            distance_m = (
                                elements[j]
                                .get("distance", {})
                                .get("value", 10000)
                            )
                            dist_matrix[i, j] = round(distance_m / 1000.0, 2)
                        else:
                            time_matrix[i, j] = 20.0
                            dist_matrix[i, j] = 10.0
        else:
            raise Exception(f"API 回傳 Status: {data.get('status')}")
    except Exception as e:
        print(f"Distance Matrix API 例外 ({e})，改用模擬矩陣")
        time_matrix = np.random.randint(10, 35, size=(n, n)).astype(float)
        dist_matrix = np.random.uniform(3.0, 20.0, size=(n, n)).round(2)
        np.fill_diagonal(time_matrix, 0)
        np.fill_diagonal(dist_matrix, 0)

    return time_matrix, dist_matrix


# =====================================================================
# 3. 健康檢查與主 API 路由
# =====================================================================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "QUBO Travel API is running!"})


@app.route("/api/plan_trip", methods=["POST"])
def plan_trip():
    start_time = time.time()
    req_data = request.get_json() or {}

    city = req_data.get("city", "高雄").strip()
    target_budget = float(req_data.get("budget", 600.0))
    target_time_limit = float(req_data.get("target_time", 600.0))

    raw_num_spots = req_data.get("num_spots", req_data.get("target_spots", 5))
    T_max = max(3, min(8, int(raw_num_spots)))

    weights = req_data.get("weights", {})
    penalties = req_data.get("penalties", {})

    alpha_r = float(weights.get("alpha_r", 15.0))
    alpha_p = float(weights.get("alpha_p", 0.05))
    alpha_d = float(weights.get("alpha_d", 0.02))
    alpha_t = float(weights.get("alpha_t", 0.01))
    alpha_t2 = float(weights.get("alpha_t2", 0.02))

    e_penalty = float(penalties.get("e_penalty", 10000.0))
    lam_penalty = float(penalties.get("lam_penalty", 10000.0))

    TARGET_N = 10
    MIN_RATING = 3.0

    raw_spots = fetch_city_spots_from_gemini(city, spot_count=TARGET_N)
    if not raw_spots:
        return (
            jsonify({"status": "error", "message": f"無法取得 {city} 的景點資料，請檢查 GEMINI_API_KEY 是否有效"}),
            400,
        )

    spots_data = []
    place_ids = []

    for item in raw_spots:
        if len(spots_data) >= TARGET_N:
            break
        spot_name = item.get("name")
        stay_time = item.get("stay_minutes", 90)
        cost = item.get("estimated_cost", 50)

        details = get_place_details_from_google(city, spot_name)
        if details:
            rating = details["rating"]
            v_count = details["user_ratings_total"]
            pid = details["place_id"]
        else:
            rating = 4.2
            v_count = 3000
            pid = None

        if rating <= MIN_RATING:
            continue

        spots_data.append({
            "id": len(spots_data),
            "name": spot_name,
            "R": rating,
            "v": v_count,
            "C": float(cost),
            "stay": float(stay_time),
        })
        place_ids.append(pid)

    N = len(spots_data)
    if N < T_max:
        return (
            jsonify({
                "status": "error",
                "message": f"合格景點數不足 ({N} 個)，少於要求的景點數 ({T_max} 個)。",
            }),
            400,
        )

    D_matrix, Dist_matrix = get_distance_and_time_matrices(place_ids)

    spot_names = [d["name"] for d in spots_data]
    R = np.array([d["R"] for d in spots_data], dtype=float)
    v = np.array([d["v"] for d in spots_data], dtype=float)
    C = np.array([d["C"] for d in spots_data], dtype=float)
    Stay = np.array([d["stay"] for d in spots_data], dtype=float)

    m = 3000.0
    mean_rating = np.mean(R)
    WR = (v / (v + m)) * R + (m / (v + m)) * mean_rating

    x = Array.create("x", shape=(N, T_max), vartype="BINARY")

    H_rating = -alpha_r * sum(
        WR[i] * x[i, t] for i in range(N) for t in range(T_max)
    )
    total_cost = sum(C[i] * x[i, t] for i in range(N) for t in range(T_max))
    H_price = alpha_p * ((total_cost - target_budget) ** 2)

    total_travel_time = sum(
        D_matrix[i, j] * x[i, t] * x[j, t + 1]
        for i in range(N)
        for j in range(N)
        for t in range(T_max - 1)
    )
    H_distance = alpha_d * total_travel_time

    total_stay_time = sum(
        Stay[i] * x[i, t] for i in range(N) for t in range(T_max)
    )
    T_target_stay = target_time_limit - 90
    H_time = (
        alpha_t * ((total_stay_time - T_target_stay) ** 2)
        + alpha_t2 * total_travel_time
    )

    H_C1 = e_penalty * sum(
        Constraint(
            ((sum(x[i, t] for i in range(N)) - 1) ** 2), label=f"C1_time_{t}"
        )
        for t in range(T_max)
    )
    H_C2 = lam_penalty * sum(
        Constraint(x[i, t1] * x[i, t2], label=f"C2_spot_{i}_{t1}_{t2}")
        for i in range(N)
        for t1 in range(T_max - 1)
        for t2 in range(t1 + 1, T_max)
    )

    H = H_rating + H_price + H_distance + H_time + H_C1 + H_C2
    model = H.compile()
    qubo, offset = model.to_qubo()

    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_reads=1000)

    best_sample = response.first.sample
    best_energy = float(response.first.energy + offset)

    selected_indices = []
    for t in range(T_max):
        for i in range(N):
            if best_sample.get(f"x[{i}][{t}]") == 1:
                selected_indices.append((t, i))

    itinerary = []
    if len(selected_indices) == T_max:
        tot_rating = float(sum(WR[i] for _, i in selected_indices))
        tot_cost = float(sum(C[i] for _, i in selected_indices))
        tot_stay = float(sum(Stay[i] for _, i in selected_indices))
        tot_time_min = float(
            sum(
                D_matrix[selected_indices[k][1], selected_indices[k + 1][1]]
                for k in range(T_max - 1)
            )
        )
        tot_dist_km = float(
            sum(
                Dist_matrix[selected_indices[k][1], selected_indices[k + 1][1]]
                for k in range(T_max - 1)
            )
        )

        for step, (t_idx, spot_i) in enumerate(selected_indices):
            step_info = {
                "step": t_idx + 1,
                "spot_name": spot_names[spot_i],
                "stay_minutes": int(Stay[spot_i]),
                "cost": int(C[spot_i]),
                "rating": round(float(WR[spot_i]), 2),
            }
            if step < len(selected_indices) - 1:
                next_spot_i = selected_indices[step + 1][1]
                step_info["next_travel_dist_km"] = float(
                    Dist_matrix[spot_i, next_spot_i]
                )
                step_info["next_travel_time_min"] = float(
                    D_matrix[spot_i, next_spot_i]
                )

            itinerary.append(step_info)

        return jsonify({
            "status": "success",
            "city": city,
            "target_spots": T_max,
            "algorithm": "PyQUBO + D-Wave Neal Simulated Annealing",
            "execution_time_sec": round(time.time() - start_time, 2),
            "min_energy": round(best_energy, 4),
            "summary": {
                "total_rating": round(tot_rating, 2),
                "total_cost": int(tot_cost),
                "target_budget": int(target_budget),
                "total_stay_minutes": int(tot_stay),
                "total_travel_time_minutes": int(tot_time_min),
                "total_travel_distance_km": round(tot_dist_km, 2),
                "total_real_time_minutes": int(tot_stay + tot_time_min),
            },
            "all_spots": spots_data,
            "matrices": {
                "time_matrix": D_matrix.tolist(),
                "dist_matrix": Dist_matrix.tolist(),
            },
            "parameters": {
                "weights": {
                    "alpha_r": alpha_r,
                    "alpha_p": alpha_p,
                    "alpha_d": alpha_d,
                    "alpha_t": alpha_t,
                    "alpha_t2": alpha_t2,
                },
                "penalties": {
                    "e_penalty": e_penalty,
                    "lam_penalty": lam_penalty,
                },
            },
            "itinerary": itinerary,
        })
    else:
        return (
            jsonify({
                "status": "warning",
                "message": f"退火未完美收斂（未選滿 {T_max} 個景點），建議嘗試重新呼叫 API 或調整懲罰係數。",
            }),
            500,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)