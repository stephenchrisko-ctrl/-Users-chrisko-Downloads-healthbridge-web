"""
HealthBridge Web Backend — Railway deployment
Reads all credentials from environment variables (set in Railway dashboard).
No database needed for single-user deployment.
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import garminconnect
import garth

from scheduler import init_scheduler, get_cached_data, set_cached_data
from emailer import send_health_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("healthbridge")

# ── Env config ────────────────────────────────────────────────────────────────
def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

APP_PIN         = env("APP_PIN", "healthbridge")   # Login PIN
GARMIN_EMAIL    = env("GARMIN_EMAIL")
GARMIN_PASSWORD = env("GARMIN_PASSWORD")
PELOTON_USER    = env("PELOTON_USERNAME")
PELOTON_PASS    = env("PELOTON_PASSWORD")
GMAIL_USER      = env("GMAIL_USER")
GMAIL_APP_PASS  = env("GMAIL_APP_PASSWORD")
DOCTOR_EMAIL    = env("DOCTOR_EMAIL")
PATIENT_NAME    = env("PATIENT_NAME", "Patient")
TIMEZONE        = env("TIMEZONE", "America/New_York")
FRONTEND_URL    = env("FRONTEND_URL", "http://localhost:5173")
GARMIN_TOKEN = env("GARMIN_TOKEN")
PELOTON_BASE = "https://api.onepeloton.com"

# ── Session store (in-memory) ─────────────────────────────────────────────────
garmin_client = None
peloton_session_data = {}


def get_garmin():
    global garmin_client if GARMIN_TOKEN:
        try:
            garth.client.loads(GARMIN_TOKEN)
            client = garminconnect.Garmin(GARMIN_EMAIL)
            client.garth = garth.client
            garmin_client = client
            logger.info("✅ Garmin connected via token")
            return client
        except Exception as e:
            logger.error(f"Garmin token auth failed: {e}")
    if garmin_client:
        return garmin_client
    if GARMIN_EMAIL and GARMIN_PASSWORD:
        try:
            client = garminconnect.Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
            client.login()
            garmin_client = client
            logger.info("✅ Garmin connected")
            return client
        except Exception as e:
            logger.error(f"Garmin connect failed: {e}")
    return None


def get_peloton():
    global peloton_session_data
    if peloton_session_data.get("user_id"):
        return peloton_session_data
    if PELOTON_USER and PELOTON_PASS:
        try:
            session = requests.Session()
            resp = session.post(
                f"{PELOTON_BASE}/auth/login",
                json={"username_or_email": PELOTON_USER, "password": PELOTON_PASS},
                headers={"Content-Type": "application/json"},
            )
            data = resp.json()
            user_id = data.get("user_id") or data.get("userId")
            peloton_session_data = {"session": session, "user_id": user_id}
            logger.info("✅ Peloton connected")
            return peloton_session_data
        except Exception as e:
            logger.error(f"Peloton connect failed: {e}")
    return None


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 HealthBridge Web starting...")
    get_garmin()
    get_peloton()
    init_scheduler(
        garmin_fn=fetch_garmin_summary,
        peloton_fn=fetch_peloton_workouts,
        email_fn=send_scheduled_report,
        timezone=TIMEZONE,
    )
    yield
    logger.info("HealthBridge shutting down")


app = FastAPI(title="HealthBridge API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.credentials != APP_PIN:
        raise HTTPException(status_code=401, detail="Invalid PIN")
    return True


# ── Models ────────────────────────────────────────────────────────────────────
class PinLogin(BaseModel):
    pin: str


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/login")
async def login(body: PinLogin):
    if body.pin != APP_PIN:
        raise HTTPException(status_code=401, detail="Wrong PIN")
    return {"success": True, "token": APP_PIN}


@app.get("/health")
async def health():
    _, _, last_sync = get_cached_data()
    return {
        "status": "ok",
        "garmin_configured": bool(GARMIN_EMAIL),
        "peloton_configured": bool(PELOTON_USER),
        "email_configured": bool(GMAIL_USER and DOCTOR_EMAIL),
        "garmin_connected": garmin_client is not None,
        "peloton_connected": bool(peloton_session_data.get("user_id")),
        "last_sync": last_sync.isoformat() if last_sync else None,
        "timezone": TIMEZONE,
        "doctor_email": DOCTOR_EMAIL,
        "patient_name": PATIENT_NAME,
    }


# ── Garmin ────────────────────────────────────────────────────────────────────
async def fetch_garmin_summary(start_date: str = None, end_date: str = None) -> dict:
    client = get_garmin()
    if not client:
        return None

    if not start_date or not end_date:
        end = datetime.now().date()
        start = (datetime.now() - timedelta(days=30)).date()
        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    def safe(fn, *args):
        try: return fn(*args)
        except Exception: return None

    heart_rates, hvrs, spO2s, stresses, body_batteries, steps_list, sleep_scores, sleep_details = [], [], [], [], [], [], [], []
    current = start
    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        hr = safe(client.get_heart_rates, ds)
        if hr: heart_rates.append(hr.get("restingHeartRate"))
        hrv = safe(client.get_hrv_data, ds)
        if hrv and hrv.get("hrvSummary"): hvrs.append(hrv["hrvSummary"].get("lastNight5MinHigh"))
        spo2 = safe(client.get_spo2_data, ds)
        if spo2: spO2s.append(spo2.get("averageSpO2"))
        stress = safe(client.get_stress_data, ds)
        if stress: stresses.append(stress.get("overallStressLevel"))
        bb = safe(client.get_body_battery, ds)
        if bb and isinstance(bb, list):
            charged = [x.get("charged") for x in bb if x.get("charged")]
            if charged: body_batteries.append(max(charged))
        steps = safe(client.get_steps_data, ds)
        if steps and isinstance(steps, list):
            s = sum(x.get("steps", 0) for x in steps)
            if s > 0: steps_list.append(s)
        sleep = safe(client.get_sleep_data, ds)
        if sleep and sleep.get("dailySleepDTO"):
            dto = sleep["dailySleepDTO"]
            sleep_scores.append(dto.get("sleepScores", {}).get("overall", {}).get("value"))
            sleep_details.append(dto)
        current += timedelta(days=1)

    weight_raw = safe(client.get_body_composition, start_date, end_date)
    weight_data = None
    if weight_raw and weight_raw.get("totalAverage"):
        wt = weight_raw["totalAverage"]
        weight_data = {
            "weight_lbs": round(wt.get("weight", 0) * 2.20462, 1) if wt.get("weight") else None,
            "bmi": wt.get("bmi"),
        }

    def avg_stage(key):
        vals = [d.get(key, 0) for d in sleep_details if d.get(key)]
        return round(avg(vals) / 60, 1) if vals else None

    return {
        "date_range": {"start": start_date, "end": end_date},
        "heart_rate": {"resting_avg": avg(heart_rates), "hrv_avg": avg(hvrs)},
        "spo2": {"avg": avg(spO2s)},
        "stress": {"avg_score": avg(stresses)},
        "body_battery": {"avg_peak": avg(body_batteries)},
        "steps": {"daily_avg": avg(steps_list), "total": sum(steps_list) if steps_list else None},
        "sleep": {
            "score_avg": avg(sleep_scores),
            "deep_hours_avg": avg_stage("deepSleepSeconds"),
            "rem_hours_avg": avg_stage("remSleepSeconds"),
            "light_hours_avg": avg_stage("lightSleepSeconds"),
            "awake_hours_avg": avg_stage("awakeSleepSeconds"),
        },
        "weight": weight_data,
    }


@app.get("/garmin/health-summary")
async def garmin_health_summary(start_date: str = None, end_date: str = None, auth=Depends(require_auth)):
    if not GARMIN_EMAIL:
        raise HTTPException(status_code=503, detail="Garmin not configured. Set GARMIN_EMAIL and GARMIN_PASSWORD in Railway.")
    if not garmin_client and not get_garmin():
        raise HTTPException(status_code=503, detail="Could not connect to Garmin. Check credentials in Railway env vars.")
    data = await fetch_garmin_summary(start_date, end_date)
    if not data:
        raise HTTPException(status_code=503, detail="Garmin fetch failed")
    return data


# ── Peloton ───────────────────────────────────────────────────────────────────
async def fetch_peloton_workouts(limit: int = 100) -> dict:
    pel = get_peloton()
    if not pel:
        return None

    resp = pel["session"].get(
        f"{PELOTON_BASE}/api/user/{pel['user_id']}/workouts",
        params={"joins": "ride,ride.instructor", "limit": limit, "page": 0, "sort_by": "-created"}
    )
    data = resp.json()
    workouts = []
    for w in data.get("data", []):
        ride = w.get("ride", {}) or {}
        instructor = ride.get("instructor", {}) or {}
        workouts.append({
            "id": w.get("id"),
            "date": datetime.fromtimestamp(w.get("created_at", 0)).strftime("%b %d, %Y"),
            "type": (w.get("fitness_discipline") or "").replace("_", " ").title(),
            "title": ride.get("title", ""),
            "instructor": f"{instructor.get('first_name', '')} {instructor.get('last_name', '')}".strip(),
            "duration_minutes": round(ride.get("duration", 0) / 60) if ride.get("duration") else None,
            "total_output_kj": round(w.get("total_work", 0) / 1000, 1) if w.get("total_work") else None,
            "avg_heart_rate": w.get("avg_heartrate"),
            "calories": w.get("calories"),
            "distance_miles": round(w.get("distance", 0) * 0.000621371, 2) if w.get("distance") else None,
        })
    return {"total": data.get("total", len(workouts)), "workouts": workouts}


@app.get("/peloton/workouts")
async def peloton_workouts(limit: int = 100, auth=Depends(require_auth)):
    if not PELOTON_USER:
        raise HTTPException(status_code=503, detail="Peloton not configured. Set PELOTON_USERNAME and PELOTON_PASSWORD in Railway.")
    data = await fetch_peloton_workouts(limit)
    if not data:
        raise HTTPException(status_code=503, detail="Could not connect to Peloton. Check credentials in Railway env vars.")
    return data


# ── Scheduled email ───────────────────────────────────────────────────────────
async def send_scheduled_report():
    if not (GMAIL_USER and GMAIL_APP_PASS and DOCTOR_EMAIL):
        logger.warning("Email not configured — skipping scheduled send")
        return

    garmin_data, peloton_data, _ = get_cached_data()
    if garmin_data is None:
        garmin_data = await fetch_garmin_summary()
    if peloton_data is None:
        peloton_data = await fetch_peloton_workouts()

    end = datetime.now().date()
    start = (datetime.now() - timedelta(days=30)).date()

    result = send_health_report(
        gmail_user=GMAIL_USER,
        gmail_app_password=GMAIL_APP_PASS,
        doctor_email=DOCTOR_EMAIL,
        patient_name=PATIENT_NAME,
        garmin_data=garmin_data,
        peloton_data=peloton_data,
        date_start=start.strftime("%Y-%m-%d"),
        date_end=end.strftime("%Y-%m-%d"),
    )
    if result["success"]:
        logger.info(f"✅ Report emailed to {DOCTOR_EMAIL}")
    else:
        logger.error(f"❌ Email failed: {result['error']}")


# ── Export ────────────────────────────────────────────────────────────────────
@app.get("/export/csv-data")
async def export_csv_data(start_date: str = None, end_date: str = None, auth=Depends(require_auth)):
    if not start_date:
        end = datetime.now().date()
        start = (datetime.now() - timedelta(days=30)).date()
        start_date, end_date = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    cached_g, cached_p, last_sync = get_cached_data()
    use_cache = last_sync and (datetime.now() - last_sync).seconds < 21600

    garmin_data = cached_g if use_cache else await fetch_garmin_summary(start_date, end_date)
    peloton_data = cached_p if use_cache else await fetch_peloton_workouts()

    return {
        "generated_at": datetime.now().isoformat(),
        "date_range": {"start": start_date, "end": end_date},
        "garmin": garmin_data,
        "peloton": peloton_data,
    }


# ── Automation actions ────────────────────────────────────────────────────────
@app.post("/automation/sync-now")
async def sync_now(auth=Depends(require_auth)):
    garmin_data = await fetch_garmin_summary()
    peloton_data = await fetch_peloton_workouts()
    set_cached_data(garmin_data, peloton_data)
    _, _, last_sync = get_cached_data()
    return {"success": True, "synced_at": last_sync.isoformat() if last_sync else None}


@app.post("/automation/send-now")
async def send_now(auth=Depends(require_auth)):
    await send_scheduled_report()
    return {"success": True}


@app.get("/automation/status")
async def automation_status(auth=Depends(require_auth)):
    from scheduler import scheduler
    jobs = []
    if scheduler:
        for job in scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })
    _, _, last_sync = get_cached_data()
    return {
        "jobs": jobs,
        "last_sync": last_sync.isoformat() if last_sync else None,
    }


@app.post("/email/test")
async def test_email(auth=Depends(require_auth)):
    if not (GMAIL_USER and GMAIL_APP_PASS and DOCTOR_EMAIL):
        raise HTTPException(status_code=400, detail="Email not fully configured in Railway env vars. Need GMAIL_USER, GMAIL_APP_PASSWORD, DOCTOR_EMAIL.")
    await send_scheduled_report()
    return {"success": True, "sent_to": DOCTOR_EMAIL}
