import math
import requests
import os
from datetime import datetime, date
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="BloodSync AI Engine")

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Initialization
SUPABASE_URL = "https://cogwavyapfpkxrfmpvtt.supabase.co"
SUPABASE_KEY = "sb_publishable_ovEZ75zdteeiDrna1j4JgQ_IYPk8yTk"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Constants ---
COMPATIBILITY_MATRIX = {
    "A+": ["A+", "A-", "O+", "O-"], "A-": ["A-", "O-"],
    "B+": ["B+", "B-", "O+", "O-"], "B-": ["B-", "O-"],
    "AB+": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    "AB-": ["AB-", "A-", "B-", "O-"],
    "O+": ["O+", "O-"], "O-": ["O-"]
}

CITY_COORDINATES = {
    "Hyderabad": (17.3850, 78.4867),
    "Nalgonda": (17.0575, 79.2684),
    "Warangal": (17.9784, 79.5941),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Nellore": (14.4426, 79.9865),
    "Tirupati": (13.6288, 79.4192),
    "Bengaluru": (12.9716, 77.5946),
    "Mysuru": (12.2958, 76.6394),
    "Mangaluru": (12.9141, 74.8560),
    "Hubli": (15.3647, 75.1240),
    "Belagavi": (15.8497, 74.4977)
}

DEFAULT_CITY_HOSPITALS = {
    "Nalgonda": [
        {"name": "Government General Hospital, Nalgonda", "distance_km": 1.2},
        {"name": "Kamineni Institute of Medical Sciences", "distance_km": 3.8},
        {"name": "Prathima Blood Bank & Medical Center", "distance_km": 2.5},
        {"name": "Red Cross Society Blood Bank Nalgonda", "distance_km": 1.9}
    ],
    "Hyderabad": [
        {"name": "Yashoda Hospitals, Secunderabad", "distance_km": 2.1},
        {"name": "Apollo Hospitals, Jubilee Hills", "distance_km": 4.5},
        {"name": "Nizam's Institute of Medical Sciences (NIMS)", "distance_km": 3.2},
        {"name": "KIMS Hospitals, Begumpet", "distance_km": 3.9}
    ]
}

# --- Pydantic Models ---
class UserRegister(BaseModel):
    name: str; email: str; password: str; role: str
    blood_group: str; state: str; city: str
    phone: str
    last_donation_date: str = None
    units_required: int = 1

class LoginPayload(BaseModel):
    email: str; password: str

class ToggleStatus(BaseModel):
    email: str; password: str; is_available: bool

class ProfileUpdate(BaseModel):
    email: str; password: str
    name: str; phone: str; blood_group: str; state: str; city: str
    last_donation_date: str

class EmergencyRequest(BaseModel):
    patient_name: str; required_group: str; units_required: int
    hospital_location_name: str; state: str; city: str

class LocationQuery(BaseModel):
    blood_group: str
    latitude: float
    longitude: float

# --- Helper Functions ---
def calculate_days(last_date_str):
    if not last_date_str: return 90
    try:
        d_date = datetime.strptime(str(last_date_str), "%Y-%m-%d").date()
        return (date.today() - d_date).days
    except: return 90

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def geocode_location(hospital_name: str, city: str, state: str):
    cleaned = hospital_name.replace('/', ', ').replace('-', ' ').replace(';', ', ').strip()
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    
    search_queries = []
    search_queries.append(f"{cleaned}, {city}, {state}, India")
    if parts:
        search_queries.append(f"{parts[0]}, {city}, {state}, India")
    if len(parts) > 1:
        search_queries.append(f"{parts[1]}, {city}, {state}, India")
    search_queries.append(f"{city}, {state}, India")
    
    headers = {"User-Agent": "BloodSyncAI/1.0"}
    for query in search_queries:
        try:
            encoded_query = requests.utils.quote(query, safe='')
            url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if data and len(data) > 0:
                    return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception:
            continue
            
    return CITY_COORDINATES.get(city, (17.3850, 78.4867))

def fetch_osm_hospitals(city, lat, lon, radius_meters=20000):
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:5];
    (
      node["amenity"="hospital"](around:{radius_meters},{lat},{lon});
      node["amenity"="clinic"](around:{radius_meters},{lat},{lon});
      node["amenity"="blood_bank"](around:{radius_meters},{lat},{lon});
      way["amenity"="hospital"](around:{radius_meters},{lat},{lon});
    );
    out center;
    """
    try:
        response = requests.post(overpass_url, data=overpass_query, timeout=5)
        data = response.json()
        hospitals = []
        for element in data.get("elements", [])[:8]:
            name = element.get("tags", {}).get("name", "Medical Center")
            h_lat = element.get("lat") or element.get("center", {}).get("lat")
            h_lon = element.get("lon") or element.get("center", {}).get("lon")
            if h_lat and h_lon:
                dist = haversine_distance(lat, lon, h_lat, h_lon)
                hospitals.append({"name": name, "distance_km": round(dist, 1)})
        
        if hospitals:
            return sorted(hospitals, key=lambda x: x["distance_km"])
    except:
        pass
        
    return DEFAULT_CITY_HOSPITALS.get(city, [
        {"name": f"Government Area Hospital, {city}", "distance_km": 1.5},
        {"name": f"Red Cross Society Blood Bank, {city}", "distance_km": 2.8},
        {"name": f"City Multi-Specialty Hospital, {city}", "distance_km": 3.4}
    ])

# --- Endpoints ---

@app.get("/api/stats")
def get_stats():
    res = supabase.table("users").select("id, role, is_available").execute()
    total_donors = sum(1 for u in res.data if u.get("role") == "donor")
    active_donors = sum(1 for u in res.data if u.get("role") == "donor" and u.get("is_available"))
    return {"total_donors": total_donors, "active_donors": active_donors}

@app.post("/api/auth/register")
def register_user(user: UserRegister):
    try:
        data = user.dict()
        data["is_available"] = True
        
        city_coords = CITY_COORDINATES.get(data["city"], (17.3850, 78.4867))
        data["latitude"] = city_coords[0]
        data["longitude"] = city_coords[1]

        if data["role"] == "requestor":
            data["last_donation_date"] = str(date.today())
        else:
            data["units_required"] = 1

        insert_res = supabase.table("users").insert(data).execute()
        created_user = insert_res.data[0]
        created_user['days_since_last_donation'] = calculate_days(created_user.get('last_donation_date'))
        
        redirect_target = "donor.html" if created_user["role"] == "donor" else "requestor.html"
        return {"status": "success", "user": created_user, "redirect": redirect_target}
    except Exception as e:
        error_msg = str(e)
        if "duplicate key" in error_msg or "already exists" in error_msg:
            raise HTTPException(status_code=400, detail="Gmail address is already registered.")
        else:
            raise HTTPException(status_code=400, detail=f"Database error: {error_msg}")

@app.post("/api/auth/login")
def login_user(payload: LoginPayload):
    res = supabase.table("users").select("*").eq("email", payload.email).eq("password", payload.password).execute()
    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid Gmail address or Password.")
    
    user = res.data[0]
    user['days_since_last_donation'] = calculate_days(user.get('last_donation_date'))
    redirect_target = "donor.html" if user["role"] == "donor" else "requestor.html"
    return {"status": "success", "user": user, "redirect": redirect_target}

@app.post("/api/donors/toggle")
def toggle_status(payload: ToggleStatus):
    res = supabase.table("users").select("id").eq("email", payload.email).eq("password", payload.password).execute()
    if not res.data: raise HTTPException(status_code=401, detail="Unauthorized")
    supabase.table("users").update({"is_available": payload.is_available}).eq("email", payload.email).execute()
    return {"message": "Status updated successfully!"}

@app.post("/api/donors/update_profile")
def update_donor_profile(payload: ProfileUpdate):
    res = supabase.table("users").select("id").eq("email", payload.email).eq("password", payload.password).execute()
    if not res.data: 
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    city_coords = CITY_COORDINATES.get(payload.city, (17.3850, 78.4867))
    update_data = {
        "name": payload.name,
        "phone": payload.phone,
        "blood_group": payload.blood_group,
        "state": payload.state,
        "city": payload.city,
        "latitude": city_coords[0],
        "longitude": city_coords[1],
        "last_donation_date": payload.last_donation_date
    }
    up_res = supabase.table("users").update(update_data).eq("email", payload.email).execute()
    updated_user = up_res.data[0]
    updated_user['days_since_last_donation'] = calculate_days(updated_user.get('last_donation_date'))
    return {"status": "success", "user": updated_user}

@app.get("/api/donors/requests_feed")
def get_donor_requests_feed(city: str, blood_group: str):
    res = supabase.table("blood_requests").select("*").eq("city", city).eq("is_resolved", False).order("created_at", desc=True).limit(5).execute()
    return {"requests": res.data}

@app.post("/api/requests/emergency_match")
def emergency_match(req: EmergencyRequest):
    req_data = req.dict()
    supabase.table("blood_requests").insert(req_data).execute()

    target_lat, target_lon = geocode_location(req.hospital_location_name, req.city, req.state)
    compatible = COMPATIBILITY_MATRIX.get(req.required_group, [req.required_group])
    res = supabase.table("users").select("*").eq("role", "donor").in_("blood_group", compatible).eq("city", req.city).eq("is_available", True).execute()
    
    donors_list = res.data
    features = []
    for d in donors_list:
        d_lat = d.get('latitude') or target_lat
        d_lon = d.get('longitude') or target_lon
        dist = haversine_distance(target_lat, target_lon, d_lat, d_lon)
        days = calculate_days(d.get('last_donation_date'))
        d['distance_km'] = round(dist, 1)
        d['days_since_last_donation'] = days
        
        recovery_score = min(days / 120, 1.0) * 50 
        distance_score = max((20 - dist) / 20, 0) * 50
        d['viability_score'] = round(recovery_score + distance_score, 1)
        features.append([dist, -days])

    ranked = []
    if donors_list:
        knn = NearestNeighbors(n_neighbors=min(len(donors_list), 5), algorithm='ball_tree')
        knn.fit(features)
        distances, indices = knn.kneighbors([[0, -120]])
        ranked = [donors_list[i] for i in indices[0]]
        ranked = sorted(ranked, key=lambda x: x['viability_score'], reverse=True)

    live_hospitals = fetch_osm_hospitals(req.city, target_lat, target_lon)
    return {
        "center_lat": target_lat,
        "center_lon": target_lon,
        "ai_ranked_donors": ranked, 
        "nearby_institutions": live_hospitals
    }

@app.post("/api/donors/nearby")
def get_nearby_donors(query: LocationQuery):
    try:
        response = supabase.table('users').select('*').eq('blood_group', query.blood_group).eq('role', 'donor').execute()
        donors = response.data

        nearby_donors = []
        for donor in donors:
            if donor.get('latitude') and donor.get('longitude'):
                dist = haversine_distance(query.latitude, query.longitude, donor['latitude'], donor['longitude'])
                if dist <= 50:
                    donor_info = donor.copy()
                    donor_info['distance_km'] = round(dist, 2)
                    donor_info.pop('password', None) 
                    nearby_donors.append(donor_info)
        
        nearby_donors.sort(key=lambda x: x['distance_km'])
        live_hospitals = fetch_osm_hospitals("Local", query.latitude, query.longitude, radius_meters=20000)
        
        return {
            "status": "success", 
            "data": nearby_donors,
            "hospitals": live_hospitals
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))