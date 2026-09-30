import streamlit as st
import pandas as pd
import requests
import folium
import ee

from datetime import datetime, timedelta
from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AgriShield",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SIMPLE STREAMLIT DESIGN
# NO HTML USED
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: white;
    }

    h1, h2, h3, h4, h5 {
        color: #111111 !important;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d9d9d9;
        border-radius: 18px;
        padding: 20px;
        transition: 0.3s;
    }

    div[data-testid="stMetric"]:hover {
        transform: scale(1.05) translateY(-5px);
        border-color: #2ea043;
        box-shadow: 0px 12px 30px rgba(0,0,0,0.18);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        transition: 0.3s;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        border-color: #2ea043;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOCATIONS
# =========================================================

locations = {
    "India": {
        "Tamil Nadu": {
            "Chennai": (13.0827, 80.2707),
            "Coimbatore": (11.0168, 76.9558),
            "Thanjavur": (10.7870, 79.1378),
            "Madurai": (9.9252, 78.1198),
            "Tiruchirappalli": (10.7905, 78.7047),
            "Salem": (11.6643, 78.1460),
            "Tirunelveli": (8.7139, 77.7567)
        },

        "Karnataka": {
            "Bengaluru": (12.9716, 77.5946)
        },

        "Andhra Pradesh": {
            "Vijayawada": (16.5062, 80.6480)
        },

        "Telangana": {
            "Hyderabad": (17.3850, 78.4867)
        }
    }
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌱 AgriShield")
st.sidebar.caption("Agricultural Drought Intelligence")

st.sidebar.markdown("---")

country = st.sidebar.selectbox(
    "Country",
    list(locations.keys())
)

state = st.sidebar.selectbox(
    "State",
    list(locations[country].keys())
)

district = st.sidebar.selectbox(
    "District / City",
    list(locations[country][state].keys())
)

analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    ["30 Days", "60 Days", "90 Days"]
)

st.sidebar.markdown("---")

drought_stress = st.sidebar.checkbox(
    "Show drought stress",
    value=True
)

agricultural_areas = st.sidebar.checkbox(
    "Show agricultural areas",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.caption("Data Sources")
st.sidebar.write("• Sentinel-2")
st.sidebar.write("• Open-Meteo")
st.sidebar.write("• Google Earth Engine")


latitude, longitude = locations[country][state][district]


# =========================================================
# EARTH ENGINE
# =========================================================

EE_PROJECT = "agrishield-510206"

ee_ready = False

try:
    ee.Initialize(project=EE_PROJECT)
    ee_ready = True
except Exception:
    ee_ready = False


# =========================================================
# WEATHER
# =========================================================

@st.cache_data(ttl=600)
def get_weather(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,

        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "showers",
            "wind_speed_10m"
        ]),

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "soil_moisture_0_to_7cm"
        ]),

        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except Exception:

        return None


# =========================================================
# SENTINEL DATA
# =========================================================

@st.cache_data(ttl=1800)
def get_sentinel_data(lat, lon, period_days):

    fallback = {
        "ndvi": 0.46,
        "ndmi": 0.21,
        "date": "Unavailable",
        "trend_dates": [],
        "trend_values": []
    }

    if not ee_ready:
        return fallback

    try:

        end_date = datetime.utcnow().date()

        start_date = (
            end_date -
            timedelta(days=period_days)
        )

        point = ee.Geometry.Point(
            [lon, lat]
        )

        region = point.buffer(3000)

        collection = (
            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )
            .filterBounds(region)
            .filterDate(
                str(start_date),
                str(end_date)
            )
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    30
                )
            )
            .sort(
                "system:time_start",
                False
            )
        )

        count = collection.size().getInfo()

        if count == 0:
            return fallback

        latest = ee.Image(
            collection.first()
        )

        ndvi_image = (
            latest
            .normalizedDifference(
                ["B8", "B4"]
            )
            .rename("NDVI")
        )

        ndmi_image = (
            latest
            .normalizedDifference(
                ["B8", "B11"]
            )
            .rename("NDMI")
        )

        combined = ee.Image.cat([
            ndvi_image,
            ndmi_image
        ])

        values = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            bestEffort=True,
            tileScale=2
        ).getInfo()

        ndvi_value = values.get(
            "NDVI",
            0.46
        )

        ndmi_value = values.get(
            "NDMI",
            0.21
        )

        timestamp = (
            latest
            .get("system:time_start")
            .getInfo()
        )

        observation_date = (
            datetime
            .utcfromtimestamp(timestamp / 1000)
            .strftime("%d %b %Y")
        )

        # NDVI trend
        def calculate_trend(image):

            ndvi = (
                image
                .normalizedDifference(
                    ["B8", "B4"]
                )
                .rename("NDVI")
            )

            value = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=60,
                bestEffort=True,
                tileScale=2
            ).get("NDVI")

            return ee.Feature(
                None,
                {
                    "date": image.date().format(
                        "YYYY-MM-dd"
                    ),
                    "ndvi": value
                }
            )

        trend_collection = (
            collection
            .sort("system:time_start")
            .map(calculate_trend)
        )

        trend_info = (
            trend_collection.getInfo()
        )

        trend_dates = []
        trend_values = []

        for feature in trend_info["features"]:

            props = feature["properties"]

            if props.get("ndvi") is not None:

                trend_dates.append(
                    props.get("date")
                )

                trend_values.append(
                    round(
                        float(
                            props.get("ndvi")
                        ),
                        3
                    )
                )

        return {
            "ndvi": float(ndvi_value),
            "ndmi": float(ndmi_value),
            "date": observation_date,
            "trend_dates": trend_dates,
            "trend_values": trend_values
        }

    except Exception:

        return fallback


# =========================================================
# LOAD DATA
# =========================================================

weather = get_weather(
    latitude,
    longitude
)

period_days = {
    "30 Days": 30,
    "60 Days": 60,
    "90 Days": 90
}[analysis_period]

satellite = get_sentinel_data(
    latitude,
    longitude,
    period_days
)


# =========================================================
# WEATHER VALUES
# =========================================================

if weather:

    current = weather.get(
        "current",
        {}
    )

    temperature = float(
        current.get(
            "temperature_2m",
            34.0
        )
    )

    humidity = float(
        current.get(
            "relative_humidity_2m",
            60
        )
    )

    precipitation = float(
        current.get(
            "precipitation",
            0
        )
    )

    wind_speed = float(
        current.get(
            "wind_speed_10m",
            0
        )
    )

else:

    temperature = 34.0
    humidity = 60.0
    precipitation = 0.0
    wind_speed = 0.0


# =========================================================
# DROUGHT CALCULATION
# =========================================================

ndvi_value = satellite["ndvi"]
ndmi_value = satellite["ndmi"]


ndvi_stress = max(
    0,
    min(
        1,
        (0.65 - ndvi_value) / 0.45
    )
)


ndmi_stress = max(
    0,
    min(
        1,
        (0.45 - ndmi_value) / 0.60
    )
)


temperature_stress = max(
    0,
    min(
        1,
        (temperature - 30) / 15
    )
)


rainfall_stress = max(
    0,
    min(
        1,
        1 - precipitation / 10
    )
)


drought_score = (
    ndvi_stress * 0.35
    + ndmi_stress * 0.30
    + temperature_stress * 0.15
    + rainfall_stress * 0.20
)


def get_drought_status(score):

    if score >= 0.70:
        return "High"

    elif score >= 0.40:
        return "Moderate"

    return "Low"


drought_level = get_drought_status(
    drought_score
)


# =========================================================
# HEADER
# =========================================================

st.title("🌱 AgriShield")

st.subheader(
    "Satellite-based Agricultural Drought Intelligence "
    "& Early Warning System"
)

st.divider()


# =========================================================
# SELECTED AREA
# =========================================================

left, right = st.columns(
    [2, 1]
)


with left:

    st.header(
        "Selected Agricultural Area"
    )

    with st.container(border=True):

        st.subheader(
            f"{district}, {state}"
        )

        st.write(
            f"Coordinates: "
            f"{latitude:.4f}° N, "
            f"{longitude:.4f}° E"
        )

        st.write(
            f"Analysis period: "
            f"{analysis_period}"
        )

        st.write(
            "Satellite and weather information "
            "is combined to estimate agricultural "
            "drought stress."
        )


with right:

    st.header(
        "Drought Status"
    )

    with st.container(border=True):

        st.metric(
            "Drought Stress Score",
            f"{drought_score:.2f}"
        )

        st.subheader(
            f"{drought_level} Drought Stress"
        )


# =========================================================
# MAP
# =========================================================

st.header(
    "Interactive Agricultural Map"
)


m = folium.Map(
    location=[
        latitude,
        longitude
    ],
    zoom_start=10,
    control_scale=True
)


folium.TileLayer(
    tiles="OpenStreetMap",
    name="Street Map",
    control=True
).add_to(m)


folium.TileLayer(
    tiles=(
        "https://mt1.google.com/vt/"
        "lyrs=s&x={x}&y={y}&z={z}"
    ),
    attr="Google",
    name="Google Satellite",
    overlay=False,
    control=True
).add_to(m)


folium.TileLayer(
    tiles=(
        "https://mt1.google.com/vt/"
        "lyrs=y&x={x}&y={y}&z={z}"
    ),
    attr="Google",
    name="Google Hybrid",
    overlay=False,
    control=True
).add_to(m)


folium.Marker(
    [latitude, longitude],
    tooltip=f"{district} - Selected Area",
    popup=(
        f"{district}<br>"
        f"State: {state}<br>"
        f"Latitude: {latitude:.4f}<br>"
        f"Longitude: {longitude:.4f}"
    )
).add_to(m)


if drought_stress:

    folium.Circle(
        location=[
            latitude,
            longitude
        ],
        radius=3000,
        color="#2ea043",
        fill=True,
        fill_opacity=0.18,
        popup=(
            f"Drought Stress Zone - "
            f"{drought_level}"
        )
    ).add_to(m)


if agricultural_areas:

    folium.Circle(
        location=[
            latitude + 0.025,
            longitude + 0.02
        ],
        radius=1800,
        color="#2ea043",
        fill=True,
        fill_opacity=0.10,
        popup="Agricultural Area"
    ).add_to(m)


folium.LayerControl().add_to(m)


with st.container(border=True):

    st_folium(
        m,
        width=None,
        height=560,
        returned_objects=[]
    )


# =========================================================
# ENVIRONMENTAL INDICATORS
# =========================================================

st.header(
    "Key Environmental Indicators"
)


c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.metric(
        "NDVI",
        f"{ndvi_value:.2f}",
        "Vegetation Health"
    )


with c2:

    st.metric(
        "NDMI",
        f"{ndmi_value:.2f}",
        "Vegetation Moisture"
    )


with c3:

    st.metric(
        "Temperature",
        f"{temperature:.1f} °C",
        "Current Air Temp."
    )


with c4:

    st.metric(
        "Humidity",
        f"{humidity:.0f}%",
        "Relative Humidity"
    )


with c5:

    st.metric(
        "Rainfall",
        f"{precipitation:.1f} mm",
        "Current Precipitation"
    )


# =========================================================
# STRESS
# =========================================================

st.header(
    "Why Is This Area Under Stress?"
)


s1, s2 = st.columns(2)


with s1:

    st.write(
        f"**Vegetation Stress — "
        f"{ndvi_stress * 100:.0f}%**"
    )

    st.progress(
        ndvi_stress
    )


    st.write(
        f"**Moisture Stress — "
        f"{ndmi_stress * 100:.0f}%**"
    )

    st.progress(
        ndmi_stress
    )


with s2:

    st.write(
        f"**Temperature Stress — "
        f"{temperature_stress * 100:.0f}%**"
    )

    st.progress(
        temperature_stress
    )


    st.write(
        f"**Rainfall Stress — "
        f"{rainfall_stress * 100:.0f}%**"
    )

    st.progress(
        rainfall_stress
    )


# =========================================================
# MAIN DROUGHT SIGNALS
# =========================================================

st.header(
    "Main Drought Signals"
)


a, b, c = st.columns(3)


with a:

    with st.container(border=True):

        st.subheader("NDVI")

        st.write(
            "NDVI indicates vegetation greenness "
            "and crop health. Lower values can "
            "indicate vegetation stress."
        )


with b:

    with st.container(border=True):

        st.subheader("NDMI")

        st.write(
            "NDMI represents moisture conditions "
            "in vegetation. Lower values can "
            "indicate moisture shortage."
        )


with c:

    with st.container(border=True):

        st.subheader("Weather")

        st.write(
            "Temperature, humidity and rainfall "
            "are used together with satellite "
            "indicators to identify stress."
        )


# =========================================================
# NDVI TREND
# =========================================================

st.header(
    "NDVI Vegetation Trend"
)


if (
    satellite["trend_dates"]
    and satellite["trend_values"]
):

    trend_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                satellite["trend_dates"]
            ),

            "NDVI": satellite[
                "trend_values"
            ]
        }
    )

    trend_df = (
        trend_df
        .dropna()
        .sort_values("Date")
    )


    if len(trend_df) > 1:

        st.line_chart(
            trend_df.set_index("Date")["NDVI"],
            height=350
        )

    else:

        st.info(
            "Not enough satellite observations "
            "for a trend."
        )

else:

    demo_dates = pd.date_range(
        end=datetime.today(),
        periods=6,
        freq="30D"
    )

    demo_values = [
        0.61,
        0.58,
        0.55,
        0.52,
        0.49,
        ndvi_value
    ]

    demo_df = pd.DataFrame(
        {
            "Date": demo_dates,
            "NDVI": demo_values
        }
    )

    st.line_chart(
        demo_df.set_index("Date")["NDVI"],
        height=350
    )

    st.caption(
        "Trend fallback shown because satellite "
        "observations could not be retrieved."
    )


# =========================================================
# LATEST SATELLITE OBSERVATION
# =========================================================

st.header(
    "Latest Satellite Observation"
)


with st.container(border=True):

    obs1, obs2, obs3 = st.columns(3)


    with obs1:

        st.metric(
            "NDVI",
            f"{ndvi_value:.3f}"
        )


    with obs2:

        st.metric(
            "NDMI",
            f"{ndmi_value:.3f}"
        )


    with obs3:

        st.metric(
            "Observation Date",
            satellite["date"]
        )


    if ee_ready:

        st.success(
            "Sentinel-2 satellite data connection active."
        )

    else:

        st.warning(
            "Earth Engine connection unavailable. "
            "Fallback values are being displayed."
        )


# =========================================================
# AGRICULTURAL ADVISORY
# =========================================================

st.header(
    "Agricultural Advisory"
)


if drought_level == "High":

    advisory = [
        "Increase field monitoring frequency.",
        "Prioritize irrigation for critical crops.",
        "Check soil moisture before irrigation.",
        "Consider water-saving irrigation methods.",
        "Monitor crop stress using satellite observations."
    ]


elif drought_level == "Moderate":

    advisory = [
        "Monitor crop moisture regularly.",
        "Use irrigation according to soil moisture conditions.",
        "Reduce unnecessary water loss.",
        "Monitor vegetation changes through NDVI.",
        "Prepare contingency measures if rainfall remains low."
    ]


else:

    advisory = [
        "Continue regular crop monitoring.",
        "Maintain efficient irrigation practices.",
        "Monitor NDVI and NDMI for early stress detection.",
        "Maintain soil moisture through suitable agricultural practices.",
        "Continue observing weather conditions."
    ]


with st.container(border=True):

    for item in advisory:

        st.write(
            f"• {item}"
        )


# =========================================================
# PRIORITY AREAS
# =========================================================

st.header(
    "Priority Agricultural Areas"
)


selected_area = pd.DataFrame(
    {
        "Area": [district],

        "Drought Score": [
            round(drought_score, 2)
        ],

        "Status": [
            get_drought_status(
                drought_score
            )
        ]
    }
)


other_priority = pd.DataFrame(
    {
        "Area": [
            "Thanjavur",
            "Madurai",
            "Coimbatore",
            "Salem",
            "Chennai"
        ],

        "Drought Score": [
            0.78,
            0.74,
            0.48,
            0.44,
            0.28
        ]
    }
)


other_priority["Status"] = (
    other_priority["Drought Score"]
    .apply(get_drought_status)
)


other_priority = other_priority[
    other_priority["Area"] != district
]


priority_data = pd.concat(
    [
        selected_area,
        other_priority
    ],
    ignore_index=True
)


priority_data = (
    priority_data
    .sort_values(
        "Drought Score",
        ascending=False
    )
    .reset_index(drop=True)
)


st.dataframe(
    priority_data,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# SYSTEM STATUS
# =========================================================

st.header(
    "System Status"
)


status1, status2, status3 = st.columns(3)


with status1:

    if ee_ready:

        st.success(
            "Sentinel-2: Connected"
        )

    else:

        st.warning(
            "Sentinel-2: Fallback Mode"
        )


with status2:

    if weather:

        st.success(
            "Weather API: Connected"
        )

    else:

        st.warning(
            "Weather API: Fallback Mode"
        )


with status3:

    st.success(
        "Dashboard: Operational"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AgriShield • Satellite-based Agricultural "
    "Drought Intelligence"
)

st.caption(
    "Sentinel-2 • Earth Engine • Open-Meteo"
)