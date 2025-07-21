# chatbot.py
from flask import Flask, render_template, request, jsonify
import requests, re, joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unidecode

app = Flask(__name__)
API_KEY = "e079e359b6b067466c9590fa623568f9"

# === Load model & scaler ===
try:
    model = joblib.load("rain_model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'rain_model_xgb.pkl' hoặc 'scaler.pkl'.")
    print("Vui lòng đảm bảo các file này nằm cùng thư mục với chatbot.py")
    exit()


# === ÁNH XẠ TÊN THÀNH PHỐ SANG MÃ SỐ CHO MÔ HÌNH ML ===
city_to_province_code = {
    "Bac Lieu": 0,
    "Ben Tre": 1,
    "Bien Hoa": 2,
    "Buon Me Thuot": 3,
    "Ca Mau": 4,
    "Cam Pha": 5,
    "Cam Ranh": 6,
    "Can Tho": 7,
    "Chau Doc": 8,
    "Da Lat": 9,
    "Ha Noi": 10,
    "Hai Duong": 11,
    "Hai Phong": 12,
    "Hanoi": 13,
    "Ho Chi Minh City": 14,
    "Hoa Binh": 15,
    "Hong Gai": 16,
    "Hue": 17,
    "Long Xuyen": 18,
    "My Tho": 19,
    "Nam Dinh": 20,
    "Nha Trang": 21,
    "Phan Rang": 22,
    "Phan Thiet": 23,
    "Play Cu": 24,
    "Qui Nhon": 25,
    "Rach Gia": 26,
    "Soc Trang": 27,
    "Tam Ky": 28,
    "Tan An": 29,
    "Thai Nguyen": 30,
    "Thanh Hoa": 31,
    "Tra Vinh": 32,
    "Tuy Hoa": 33,
    "Uong Bi": 34,
    "Viet Tri": 35,
    "Vinh": 36,
    "Vinh Long": 37,
    "Vung Tau": 38,
    "Yen Bai": 39,
}

# === Detect ngày hỏi ===
def detect_day_offset(message):
    message = unidecode.unidecode(message.lower())
    if "ngay mai" in message or "mai" in message:
        return 1
    elif "ngay mot" in message or "mot" in message:
        return 2
    elif re.search(r"\d{1,2}/\d{1,2}", message):
        match = re.search(r"(\d{1,2})/(\d{1,2})", message)
        if match:
            today = datetime.today()
            day, month = int(match.group(1)), int(match.group(2))
            target = datetime(today.year, month, day)
            return (target - today).days
    return 0

# === Phát hiện có phải hỏi về "mưa" không ===
def is_rain_query(message):
    return any(word in unidecode.unidecode(message.lower()) for word in ["mua", "ao mua", "luong mua", "co mua", "troi uot"])

# === Hàm tách địa điểm ===
def get_searchable_name(city_name):
    searchable = unidecode.unidecode(city_name.lower())
    searchable = searchable.replace("city", "").strip()
    return searchable

def extract_city(message):
    message_decoded = unidecode.unidecode(message.lower())
    sorted_cities = sorted(city_to_province_code.keys(), key=len, reverse=True)
    
    for city in sorted_cities:
        city_search_term = get_searchable_name(city)
        if city_search_term in message_decoded:
            return city

    stopwords = ["thoi tiet", "du bao", "ngay mai", "ngay mot", "hom nay", "mai", "mot", "la", "co", "khong", "se", "du kien", "co mua", "trong", "thanh pho"]
    for sw in stopwords:
        message_decoded = message_decoded.replace(sw, "")
    
    message_decoded = re.sub(r"\s+", " ", message_decoded).strip()
    return " ".join(w.capitalize() for w in message_decoded.split())


# === Gọi OpenWeather API để lấy thông tin thời tiết hiện tại hoặc tương lai ===
def get_weather_forecast(city, day_offset=0):
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric&lang=vi"
        res = requests.get(url)
        if res.status_code != 200:
            error_data = res.json()
            error_message = error_data.get('message', 'Lỗi không xác định từ API.')
            return f"❌ Không thể lấy dữ liệu thời tiết cho **{city}**. Lý do: {error_message}"

        data = res.json()
        forecasts = data.get("list", [])
        if not forecasts:
            return f"❌ Không có dữ liệu dự báo cho **{city}**."

        target_date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        selected = None
        for f in forecasts:
            if target_date in f.get("dt_txt", "") and "12:00:00" in f.get("dt_txt", ""):
                selected = f
                break
        if not selected:
            selected = forecasts[min(day_offset * 8, len(forecasts) - 1)]

        weather_desc = selected.get("weather", [{}])[0].get("description", "Không rõ").capitalize()
        main_data = selected.get("main", {})
        wind_data = selected.get("wind", {})
        
        temp = main_data.get("temp", "N/A")
        feels_like = main_data.get("feels_like", "N/A")
        humidity = main_data.get("humidity", "N/A")
        wind_speed = wind_data.get("speed", "N/A")
        rain_3h = selected.get("rain", {}).get("3h", 0)

        label = "hôm nay" if day_offset == 0 else "ngày mai" if day_offset == 1 else f"{day_offset} ngày tới"
        rain_msg = "☀️ Không có dấu hiệu mưa." if rain_3h == 0 else (
            "🌦️ Có thể có mưa nhẹ." if rain_3h < 10 else "🌧️ Có mưa vừa hoặc lớn, hãy chuẩn bị áo mưa.")

        return (
            f"📍 **Thời tiết tại {city} ({label}):**\n"
            f"- 🌤️ Trạng thái: {weather_desc}\n"
            f"- 🌡️ Nhiệt độ: {temp}°C (cảm giác: {feels_like}°C)\n"
            f"- 💧 Độ ẩm: {humidity}%\n"
            f"- 🌬️ Gió: {wind_speed} m/s\n"
            f"- ☔ Lượng mưa API dự báo: {rain_3h} mm\n"
            f"\n{rain_msg}"
        )
    except Exception as e:
        return f"⚠️ Lỗi khi xử lý dữ liệu thời tiết: {str(e)}"

# === Hàm ML dự đoán lượng mưa (mm) ===
def predict_rain_by_model(city, day_offset=0):
    try:
        if city not in city_to_province_code:
            return f"⚠️ Rất tiếc, tôi chưa có dữ liệu để dự đoán mưa cho **{city}** bằng mô hình ML.", -1

        province_code = city_to_province_code[city]
        
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url)
        if res.status_code != 200:
            return f"❌ Không thể lấy dữ liệu thời tiết cho **{city}** để làm đầu vào cho mô hình.", -1
        data = res.json()

        forecasts = data.get("list", [])
        if not forecasts:
            return f"❌ Không có dữ liệu dự báo cho **{city}** để làm đầu vào cho mô hình.", -1

        target_date_str = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        selected_forecast = None
        for f in forecasts:
            if target_date_str in f.get("dt_txt", "") and "12:00:00" in f.get("dt_txt", ""):
                selected_forecast = f
                break
        if not selected_forecast:
            index = min(day_offset * 8, len(forecasts) - 1)
            selected_forecast = forecasts[index]

        main_data = selected_forecast.get('main', {})
        wind_data = selected_forecast.get('wind', {})
        clouds_data = selected_forecast.get('clouds', {})
        
        api_rain = selected_forecast.get("rain", {}).get("3h", 0)
        humidity = main_data.get('humidity', 0)
        cloud_cover = clouds_data.get('all', 0)

        rain_proxy = api_rain
        if api_rain == 0:
            if humidity > 90 and cloud_cover > 90:
                rain_proxy = 1.0
            elif humidity > 80 or cloud_cover > 75:
                rain_proxy = 0.2
        
        target_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)

        features_dict = {
            'province': province_code, 'year': target_day.year, 'month': target_day.month,
            'day': target_day.day, 'wind': wind_data.get('speed', 0), 'humidi': humidity,
            'cloud': cloud_cover, 'avg_temp': main_data.get('temp', 0), 'rain_1': rain_proxy,
            'rain_2': rain_proxy, 'rain_3': rain_proxy, 'rain_7': rain_proxy,
            'rain_mean_3d': rain_proxy, 'rain_mean_7d': rain_proxy, 'rain_mean_10d': rain_proxy,
            'rain_mean_14d': rain_proxy, 'temp_range': 0, 'season': 0, 'wind_d': 0, 'region': 0
        }

        feature_order = [
            'province', 'year', 'month', 'day', 'wind', 'humidi', 'cloud', 'rain_1', 'rain_2',
            'rain_3', 'rain_7', 'rain_mean_3d', 'rain_mean_7d', 'rain_mean_10d', 'rain_mean_14d',
            'avg_temp', 'temp_range', 'season', 'wind_d', 'region'
        ]

        features_df = pd.DataFrame([features_dict])[feature_order]

        scaled_features = scaler.transform(features_df)
        log_pred = model.predict(scaled_features)
        rain_mm = max(0, np.expm1(log_pred[0]))

        # SỬA LỖI 11: Ghi đè kết quả ML nếu nó dự đoán 0 nhưng API lại dự đoán có mưa
        if rain_mm <= 0.1 and api_rain > 0:
            rain_mm = api_rain

        if day_offset == 0: label = "hôm nay"
        elif day_offset == 1: label = "ngày mai"
        else: label = f"{day_offset} ngày tới"

        if rain_mm >= 20:
            msg = f"🌧️ Mưa rất lớn có thể xảy ra ở {city} {label}: khoảng {rain_mm:.1f} mm. Bạn nên chuẩn bị kỹ càng!"
        elif rain_mm >= 5:
            msg = f"🌦️ Có khả năng mưa rào ở {city} {label}: khoảng {rain_mm:.1f} mm."
        elif rain_mm > 0.1:
            msg = f"🌤️ Có thể có mưa nhẹ không đáng kể ở {city} {label}: khoảng {rain_mm:.1f} mm."
        else:
            msg = f"☀️ Trời khô ráo, mô hình dự đoán lượng mưa tại {city} {label} là {rain_mm:.2f} mm."
        
        return msg, rain_mm
    
    except Exception as e:
        return f"⚠️ Lỗi khi dự đoán bằng mô hình: {str(e)}", -1

# === Flask Routes ===
@app.route("/")
def index():
    return render_template("homepage.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_msg = request.json.get("message", "")
    city = extract_city(user_msg)
    if not city:
        return jsonify({"reply": "📍 Vui lòng nhập tên địa điểm (ví dụ: Huế, Đà Nẵng, Hà Nội) để tôi tra cứu nhé!"})

    offset = detect_day_offset(user_msg)
    reply = ""

    if is_rain_query(user_msg):
        ml_reply, ml_rain_mm = predict_rain_by_model(city, offset)

        if ml_rain_mm == -1:
            api_forecast = get_weather_forecast(city, offset)
            reply = (
                f"{ml_reply}\n\n"
                f"---\n\nTuy nhiên, đây là dự báo thời tiết chung từ API:\n\n"
                f"{api_forecast}"
            )
        elif ml_rain_mm <= 0.1:
            api_forecast = get_weather_forecast(city, offset)
            reply = (
                f"{ml_reply}\n\n"
                f"---\n\n**Để có thêm thông tin, đây là dự báo chi tiết từ API:**\n\n"
                f"{api_forecast}"
            )
        else:
            reply = ml_reply
            
    else:
        reply = get_weather_forecast(city, offset)

    return jsonify({"reply": reply})

# === RUN ===
if __name__ == "__main__":
    app.run(debug=True)
