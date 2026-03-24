import os
import time
import uuid
import sqlite3
from flask import Flask, render_template, request,redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
#from muzzle_register import register_muzzle_automatically, link_muzzle_to_cow
#from googletrans import Translator
import google.generativeai as genai
from dotenv import load_dotenv
import qrcode
from datetime import date as dt_date, timedelta
from deep_translator import GoogleTranslator
import requests 
import cv2
import json
import numpy as np
from ultralytics import YOLO
import uuid
from datetime import datetime  # Add this import
import base64
from PIL import Image
import io
import psycopg2
from urllib.parse import urlparse
from psycopg2 import extras
from gtts import gTTS
from flask import send_file

translator = GoogleTranslator(source='auto', target='en')
print(translator.translate("ನಮಸ್ಕಾರ"))  # Kannada → English


#import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key from .env
genai.configure(api_key=os.getenv("api_key"))
SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")  # Add this line

app = Flask(__name__)
app.secret_key = "kamadhenu_secret"





DB_NAME = "kamadhenu_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres123"
DB_HOST = "localhost"
DB_PORT = "5432"
COW_UPLOAD_FOLDER = os.path.join("static", "uploads", "cow")
VET_UPLOAD_FOLDER = os.path.join("static", "uploads", "vets")
DATABASE_URL = f"postgresql://postgres:postgres123@localhost:5432/kamadhenu_db"

os.makedirs(COW_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VET_UPLOAD_FOLDER, exist_ok=True)

app.config["COW_UPLOAD_FOLDER"] = COW_UPLOAD_FOLDER
app.config["VET_UPLOAD_FOLDER"] = VET_UPLOAD_FOLDER

FARMER_UPLOAD_FOLDER = os.path.join("static", "uploads", "farmers")
os.makedirs(FARMER_UPLOAD_FOLDER, exist_ok=True)
app.config["FARMER_UPLOAD_FOLDER"] = FARMER_UPLOAD_FOLDER

import qrcode

QR_FOLDER = "static/qrcodes"
if not os.path.exists(QR_FOLDER):
    os.makedirs(QR_FOLDER)


def get_base_url():
    """Get the base URL dynamically for QR codes"""
    if 'RENDER' in os.environ:
        # Running on Render
        render_external_url = os.environ.get('RENDER_EXTERNAL_URL')
        if render_external_url:
            return render_external_url.rstrip('/')
    
    # Fallback to request host
    from flask import request
    if request and hasattr(request, 'host_url'):
        return request.host_url.rstrip('/')
    
    # Default fallback
    return "https://kamadhenu-zdm2.onrender.com"
# ================= YOLO MODEL LAZY LOADING ==================
import torch

# Global cache for models
_muzzle_model_cache = None
_disease_model_cache = None

def get_muzzle_model():
    """Lazy load the muzzle detection YOLO model"""
    global _muzzle_model_cache
    if _muzzle_model_cache is None:
        try:
            print("Loading muzzle detection model...")
            from ultralytics import YOLO
            
            # For PyTorch 2.6.0, we need to handle weights_only=True issue
            # Patch torch.load if needed
            try:
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    # Set weights_only=False to bypass security for model files
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load
            except:
                pass
            
            _muzzle_model_cache = YOLO("best_new.pt", verbose=False)
            print("✅ Muzzle detection model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load muzzle model: {e}")
            print("⚠️ Using dummy model for development")
            
            # Create a dummy model for development/testing
            class DummyModel:
                def __init__(self):
                    self.names = {0: 'muzzle'}
                    
                def __call__(self, frame, *args, **kwargs):
                    # Return a dummy result object
                    class Result:
                        def __init__(self):
                            class Boxes:
                                def __init__(self):
                                    self.boxes = None
                            self.boxes = Boxes()
                    return [Result()]
                    
            _muzzle_model_cache = DummyModel()
    
    return _muzzle_model_cache

def get_disease_model():
    """Lazy load the disease detection YOLO model"""
    global _disease_model_cache
    if _disease_model_cache is None:
        try:
            print("Loading disease detection model...")
            from ultralytics import YOLO
            
            # For PyTorch 2.6.0, we need to handle weights_only=True issue
            # Patch torch.load if needed
            try:
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    # Set weights_only=False to bypass security for model files
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load
            except:
                pass
            
            _disease_model_cache = YOLO("best.pt", verbose=False)
            print("✅ Disease detection model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load disease model: {e}")
            print("⚠️ Using dummy model for development")
            
            # Create a dummy model for development/testing
            class DummyModel:
                def __init__(self):
                    self.names = {}
                    
                def predict(self, source=None, conf=None, save=False):
                    # Return a dummy result object
                    class Result:
                        def __init__(self):
                            self.boxes = None
                            self.plot = lambda: None
                    return [Result()]
                    
            _disease_model_cache = DummyModel()
    
    return _disease_model_cache

# Add these after the lazy loading functions
DB_FILE = "cow_embeddings.json"
MUZZLE_FOLDER = "static/uploads/muzzles"
os.makedirs(MUZZLE_FOLDER, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.7
IDENTIFICATION_THRESHOLD = 0.6

# ================= SMS Sending Function ==================
def send_sms(phone, message):
    """Send SMS using Fast2SMS API"""
    url = "https://www.fast2sms.com/dev/bulkV2"
    
    # Clean phone number - remove any non-digit characters
    phone_clean = ''.join(filter(str.isdigit, str(phone)))
    if len(phone_clean) > 10:
        phone_clean = phone_clean[-10:]
    
    print(f"📞 Sending SMS to: {phone_clean}")
    
    # Create payload as dictionary
    payload = {
        'sender_id': 'LPOINT',
        'message': message,
        'language': 'english',
        'route': 'q',
        'numbers': phone_clean
    }
    
    headers = {
        'authorization': SMS_API_KEY
    }
    
    try:
        response = requests.post(url, data=payload, headers=headers)
        print("📱 SMS Response:", response.text)
        
        response_data = response.json()
        
        if response_data.get('return', False):
            print(f"✅ SMS sent successfully!")
            return {'success': True, 'message_id': response_data.get('request_id')}
        else:
            error_msg = response_data.get('message', 'Unknown error')
            print(f"❌ SMS failed: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        print(f"❌ SMS Error: {str(e)}")
        return {'success': False, 'error': str(e)}

def get_db():
    """Return a PostgreSQL database connection with dictionary cursor"""
    import psycopg2
    from psycopg2 import extras
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        # This makes all cursors return dictionaries
        conn.cursor_factory = extras.RealDictCursor
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        raise e

import random
from datetime import datetime, timedelta

def generate_otp():
    """Generate a 4-digit OTP"""
    return str(random.randint(1000, 9999))

def save_otp(phone, otp):
    """Save OTP to database with 10-minute expiration"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Expire any existing OTPs for this phone
    cursor.execute("UPDATE password_reset_otp SET verified = 1 WHERE phone = %s", (phone,))
    
    # Calculate expiration time (10 minutes from now)
    expires_at = datetime.now() + timedelta(minutes=10)
    
    # Save new OTP
    cursor.execute("""
        INSERT INTO password_reset_otp (phone, otp, expires_at) 
        VALUES (%s, %s, %s)
    """, (phone, otp, expires_at.isoformat()))
    
    conn.commit()
    conn.close()

def verify_otp(phone, otp):
    """Verify OTP is valid and not expired"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM password_reset_otp 
        WHERE phone = %s AND otp = %s AND verified = 0 
        AND expires_at > datetime('now')
    """, (phone, otp))
    
    otp_record = cursor.fetchone()
    
    if otp_record:
        # Mark OTP as verified
        cursor.execute("UPDATE password_reset_otp SET verified = 1 WHERE id = %s", (otp_record['id'],))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def update_password(phone, new_password):
    """Update farmer's password in database"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE farmers SET password = %s WHERE phone = %s", (new_password, phone))
    
    if cursor.rowcount > 0:
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

# Add these functions for language support
@app.context_processor
def inject_lang():
    """Make current language available to all templates"""
    return {'current_lang': session.get('language', 'en')}

@app.route('/set_language', methods=['POST'])
def set_language_route():
    """Set language in session and return success"""
    try:
        data = request.get_json()
        lang = data.get('lang', 'en')
        
        # Validate language
        if lang not in ['en', 'kn']:
            lang = 'en'
        
        # Store in session
        session['language'] = lang
        session.permanent = True  # Make session persistent
        
        # Also set cookie for JavaScript
        response = jsonify({"success": True, "lang": lang})
        response.set_cookie('user_language', lang, max_age=31536000, path='/')
        
        print(f"✅ Language set to: {lang}")
        return response
        
    except Exception as e:
        print(f"❌ Error setting language: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/")
def home():
    lang = request.args.get('lang', 'en')
    if lang in ['en', 'kn']:
        session['language'] = lang
    # Redirect to login page
    return redirect(url_for('login'))



# ✅ Admin credentials (hardcoded)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin@123"

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid Admin Credentials", "danger")
            return redirect(url_for("admin_login"))

    return render_template("admin_login.html")  # make a login form

@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()

    # Basic counts
    cursor.execute("SELECT COUNT(*) as total FROM farmers")
    farmer_count = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as total FROM cows")
    cow_count = cursor.fetchone()["total"]

    # FIXED: Add alias for vet count
    cursor.execute("SELECT COUNT(*) as count FROM veterinarians")
    vet_count = cursor.fetchone()["count"]

    # Detailed cattle type counts
    cursor.execute("""
        SELECT cattle_type, COUNT(*) as count 
        FROM cows 
        GROUP BY cattle_type
    """)
    cattle_type_counts = cursor.fetchall()
    
    # Convert to dictionary for easy access
    cattle_stats = {}
    for row in cattle_type_counts:
        cattle_stats[row['cattle_type']] = row['count']
    
    # Get counts for each cattle type (with 0 as default)
    total_cows = cattle_stats.get('cow', 0)
    total_buffalo = cattle_stats.get('buffalo', 0)
    total_bull = cattle_stats.get('bull', 0)
    total_male_buffalo = cattle_stats.get('male_buffalo', 0)
    total_calf = cattle_stats.get('calf', 0)

    # Enhanced stats
    today = dt_date.today().isoformat()
    
    # Today's milk production
    cursor.execute("SELECT COALESCE(SUM(total), 0) as total FROM milk_yield WHERE date=%s", (today,))
    total_milk_today = cursor.fetchone()["total"]

    # Sales data
    cursor.execute("SELECT COUNT(*) as sold_count, COALESCE(SUM(price), 0) as revenue FROM cows_for_sale WHERE is_sold=TRUE")
    sales_data = cursor.fetchone()
    cows_sold = sales_data["sold_count"]
    total_revenue = sales_data["revenue"]

    # Today's appointments
    cursor.execute("SELECT COUNT(*) as total FROM appointments WHERE date=%s", (today,))
    today_appointments_result = cursor.fetchone()
    today_appointments = today_appointments_result["total"] if today_appointments_result else 0

    # Active listings for sale
    cursor.execute("SELECT COUNT(*) as total FROM cows_for_sale WHERE is_sold=FALSE")
    active_listings_result = cursor.fetchone()
    active_listings = active_listings_result["total"] if active_listings_result else 0

    # Recent activities
    cursor.execute("""
        SELECT 'farmer' as type, name, email as contact, created_at FROM farmers 
        UNION ALL 
        SELECT 'cow' as type, breed as name, cow_id as contact, created_at FROM cows 
        UNION ALL 
        SELECT 'vet' as type, name, email as contact, created_at FROM veterinarians 
        ORDER BY created_at DESC LIMIT 5
    """)
    recent_data = cursor.fetchall()
    
    recent_activities = []
    recent_registrations = []
    
    for item in recent_data:
        if item["type"] == "farmer":
            recent_activities.append({
                "message": f"New farmer registered: {item['name']}",
                "timestamp": item["created_at"],
                "type": "Farmer",
                "color": "success"
            })
            recent_registrations.append({
                "type": "Farmer",
                "icon": "fa-user",
                "name": item["name"],
                "contact": item["contact"],
                "date": item["created_at"]
            })
        elif item["type"] == "cow":
            recent_activities.append({
                "message": f"New cow added: {item['name']}",
                "timestamp": item["created_at"],
                "type": "Cow",
                "color": "info"
            })
            recent_registrations.append({
                "type": "Cow",
                "icon": "fa-cow",
                "name": item["name"],
                "contact": item["contact"][:8] + "...",
                "date": item["created_at"]
            })
        elif item["type"] == "vet":
            recent_activities.append({
                "message": f"New vet registered: {item['name']}",
                "timestamp": item["created_at"],
                "type": "Vet",
                "color": "primary"
            })
            recent_registrations.append({
                "type": "Veterinarian",
                "icon": "fa-user-md",
                "name": item["name"],
                "contact": item["contact"],
                "date": item["created_at"]
            })

    # Milk production chart data (last 7 days) - FIXED for PostgreSQL
    cursor.execute("""
        SELECT date, SUM(total) as daily_total 
        FROM milk_yield 
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY date 
        ORDER BY date
    """)
    milk_data = cursor.fetchall()
    
    milk_chart_labels = []
    milk_chart_data = []
    
    if milk_data:
        for row in milk_data:
            milk_chart_labels.append(row["date"])
            milk_chart_data.append(row["daily_total"] or 0)
    else:
        for i in range(7):
            date = (dt_date.today() - timedelta(days=6-i)).isoformat()
            milk_chart_labels.append(date)
            milk_chart_data.append(0)

    # System status
    storage_usage = 0  # Placeholder value
    active_sessions = 1
    system_uptime = "Just started"
    current_date = dt_date.today().strftime("%B %d, %Y")
    
    active_sessions = 1
    system_uptime = "Just started"
    current_date = dt_date.today().strftime("%B %d, %Y")

    conn.close()

    return render_template(
        "admin_dashboard.html",
        farmer_count=farmer_count,
        cow_count=cow_count,
        vet_count=vet_count,
        total_cows=total_cows,
        total_buffalo=total_buffalo,
        total_bull=total_bull,
        total_male_buffalo=total_male_buffalo,
        total_calf=total_calf,
        total_milk_today=total_milk_today,
        cows_sold=cows_sold,
        total_revenue=total_revenue,
        today_appointments=today_appointments,
        active_listings=active_listings,
        recent_activities=recent_activities,
        recent_registrations=recent_registrations,
        milk_chart_labels=milk_chart_labels,
        milk_chart_data=milk_chart_data,
        storage_usage=storage_usage,
        active_sessions=active_sessions,
        system_uptime=system_uptime,
        current_date=current_date
    )

@app.route("/admin/analytics")
def admin_analytics():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()

    # -----------------------------
    # LOCATION FILTER SETUP
    # -----------------------------
    location_filter = request.args.get("location", "").strip()

    params = []
    farmer_where = ""
    cow_where = ""
    milk_where = ""
    sales_where = ""

    if location_filter:
        location_like = f"%{location_filter}%"
        params = [location_like, location_like, location_like]

        farmer_where = "WHERE (state ILIKE %s OR city ILIKE %s OR address ILIKE %s)"

        cow_where = """
            WHERE farmer_id IN (
                SELECT farmer_id FROM farmers
                WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
            )
        """

        milk_where = """
            WHERE cow_id IN (
                SELECT cow_id FROM cows
                WHERE farmer_id IN (
                    SELECT farmer_id FROM farmers
                    WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
                )
            )
        """

        sales_where = """
            AND farmer_id IN (
                SELECT farmer_id FROM farmers
                WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
            )
        """

    # -----------------------------
    # 1. Monthly Farmer Registration
    # -----------------------------
    if location_filter:
        farmer_query = f"""
            SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COUNT(*) AS farmer_count
            FROM farmers
            {farmer_where}
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(farmer_query, params)
    else:
        farmer_query = """
            SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COUNT(*) AS farmer_count
            FROM farmers
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(farmer_query)
    farmer_monthly = cursor.fetchall()

    # -----------------------------
    # 2. Monthly Cow Registration
    # -----------------------------
    if location_filter:
        cow_query = f"""
            SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COUNT(*) AS cow_count
            FROM cows
            {cow_where}
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(cow_query, params)
    else:
        cow_query = """
            SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COUNT(*) AS cow_count
            FROM cows
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(cow_query)
    cow_monthly = cursor.fetchall()

    # -----------------------------
    # 3. Vet Registrations
    # -----------------------------
    if location_filter:
        vet_query = f"""
            SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COUNT(*) AS vet_count
            FROM veterinarians
            WHERE vet_id IN (
                SELECT vet_id FROM veterinarians
                WHERE clinic ILIKE %s OR name ILIKE %s
            )
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(vet_query, (location_like, location_like))
    else:
        vet_query = """
            SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COUNT(*) AS vet_count
            FROM veterinarians
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(vet_query)
    vet_monthly = cursor.fetchall()

    # -----------------------------
    # Combine months
    # -----------------------------
    all_months = set()
    for r in farmer_monthly:
        all_months.add(r["month"])
    for r in cow_monthly:
        all_months.add(r["month"])
    for r in vet_monthly:
        all_months.add(r["month"])
    
    all_months = sorted(list(all_months), reverse=True)[:12]

    monthly_trends = []
    month_dict = {}
    
    for r in farmer_monthly:
        month_dict[(r["month"], 'farmer')] = r["farmer_count"]
    for r in cow_monthly:
        month_dict[(r["month"], 'cow')] = r["cow_count"]
    for r in vet_monthly:
        month_dict[(r["month"], 'vet')] = r["vet_count"]
    
    for m in all_months:
        monthly_trends.append({
            "month": m,
            "farmer_count": month_dict.get((m, 'farmer'), 0),
            "cow_count": month_dict.get((m, 'cow'), 0),
            "vet_count": month_dict.get((m, 'vet'), 0),
        })

    # -----------------------------
    # 4. Breed Distribution
    # -----------------------------
    if location_filter:
        breed_query = f"""
            SELECT breed, COUNT(*) as count
            FROM cows
            {cow_where}
            GROUP BY breed
            ORDER BY count DESC
        """
        cursor.execute(breed_query, params)
    else:
        breed_query = """
            SELECT breed, COUNT(*) as count
            FROM cows
            GROUP BY breed
            ORDER BY count DESC
        """
        cursor.execute(breed_query)
    breed_distribution = cursor.fetchall()

    # -----------------------------
    # 5. Milk Analytics
    # -----------------------------
    if location_filter:
        milk_query = f"""
            SELECT 
                TO_CHAR(date, 'YYYY-MM') AS month,
                COALESCE(SUM(total), 0) AS total_milk,
                COALESCE(AVG(total), 0) AS avg_daily_milk,
                COUNT(DISTINCT cow_id) as cows_tracked
            FROM milk_yield
            {milk_where}
            GROUP BY TO_CHAR(date, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(milk_query, params)
    else:
        milk_query = """
            SELECT 
                TO_CHAR(date, 'YYYY-MM') AS month,
                COALESCE(SUM(total), 0) AS total_milk,
                COALESCE(AVG(total), 0) AS avg_daily_milk,
                COUNT(DISTINCT cow_id) as cows_tracked
            FROM milk_yield
            GROUP BY TO_CHAR(date, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(milk_query)
    milk_analytics = cursor.fetchall()

    # -----------------------------
    # 6. Sales Analytics
    # -----------------------------
    if location_filter:
        sales_query = f"""
            SELECT 
                TO_CHAR(listed_at, 'YYYY-MM') AS month,
                COUNT(*) AS cows_sold,
                COALESCE(SUM(price), 0) AS total_revenue,
                COALESCE(AVG(price), 0) AS avg_price
            FROM cows_for_sale
            WHERE is_sold = TRUE
            {sales_where}
            GROUP BY TO_CHAR(listed_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(sales_query, params)
    else:
        sales_query = """
            SELECT 
                TO_CHAR(listed_at, 'YYYY-MM') AS month,
                COUNT(*) AS cows_sold,
                COALESCE(SUM(price), 0) AS total_revenue,
                COALESCE(AVG(price), 0) AS avg_price
            FROM cows_for_sale
            WHERE is_sold = TRUE
            GROUP BY TO_CHAR(listed_at, 'YYYY-MM')
            ORDER BY month DESC
            LIMIT 12
        """
        cursor.execute(sales_query)
    sales_analytics = cursor.fetchall()

    # -----------------------------
    # 7. Farmer Stats
    # -----------------------------
    if location_filter:
        cursor.execute("""
            SELECT 
                COUNT(*) AS total_farmers,
                COUNT(CASE WHEN TO_CHAR(created_at, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM') THEN 1 END) AS new_this_month
            FROM farmers
            WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
        """, (location_like, location_like, location_like))
    else:
        cursor.execute("""
            SELECT 
                COUNT(*) AS total_farmers,
                COUNT(CASE WHEN TO_CHAR(created_at, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM') THEN 1 END) AS new_this_month
            FROM farmers
        """)
    
    farmer_stats_base = cursor.fetchone()

    # farmers with cows
    if location_filter:
        cursor.execute(f"""
            SELECT COUNT(DISTINCT farmer_id) as farmers_with_cows 
            FROM cows 
            {cow_where}
        """, params)
    else:
        cursor.execute("SELECT COUNT(DISTINCT farmer_id) as farmers_with_cows FROM cows")
    
    farmers_with_cows = cursor.fetchone()["farmers_with_cows"] or 0

    # active farmers (give milk this month)
    if location_filter:
        cursor.execute("""
            SELECT COUNT(DISTINCT c.farmer_id) AS active_farmers
            FROM milk_yield m
            JOIN cows c ON m.cow_id = c.cow_id
            WHERE TO_CHAR(m.date, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM')
            AND c.farmer_id IN (
                SELECT farmer_id FROM farmers
                WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
            )
        """, (location_like, location_like, location_like))
    else:
        cursor.execute("""
            SELECT COUNT(DISTINCT c.farmer_id) AS active_farmers
            FROM milk_yield m
            JOIN cows c ON m.cow_id = c.cow_id
            WHERE TO_CHAR(m.date, 'YYYY-MM') = TO_CHAR(CURRENT_DATE, 'YYYY-MM')
        """)
    
    active_farmers = cursor.fetchone()["active_farmers"] or 0

    farmer_stats = {
        "total_farmers": farmer_stats_base["total_farmers"] or 0,
        "new_this_month": farmer_stats_base["new_this_month"] or 0,
        "farmers_with_cows": farmers_with_cows,
        "active_farmers": active_farmers
    }

    # -----------------------------
    # 8. Top Cows
    # -----------------------------
    if location_filter:
        top_cows_query = """
            SELECT 
                c.cow_id, c.breed, c.farmer_id,
                COALESCE(SUM(m.total), 0) AS total_milk,
                COALESCE(AVG(m.total), 0) AS avg_daily_milk
            FROM cows c
            LEFT JOIN milk_yield m ON c.cow_id = m.cow_id
            WHERE c.farmer_id IN (
                SELECT farmer_id FROM farmers
                WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
            )
            GROUP BY c.cow_id, c.breed, c.farmer_id
            ORDER BY total_milk DESC
            LIMIT 10
        """
        cursor.execute(top_cows_query, (location_like, location_like, location_like))
    else:
        top_cows_query = """
            SELECT 
                c.cow_id, c.breed, c.farmer_id,
                COALESCE(SUM(m.total), 0) AS total_milk,
                COALESCE(AVG(m.total), 0) AS avg_daily_milk
            FROM cows c
            LEFT JOIN milk_yield m ON c.cow_id = m.cow_id
            GROUP BY c.cow_id, c.breed, c.farmer_id
            ORDER BY total_milk DESC
            LIMIT 10
        """
        cursor.execute(top_cows_query)
    
    top_cows = cursor.fetchall()

    # -----------------------------
    # 9. Geographic Distribution
    # -----------------------------
    if location_filter:
        geo_query = f"""
            SELECT state, COUNT(*) AS farmer_count
            FROM farmers
            {farmer_where}
            GROUP BY state
            ORDER BY farmer_count DESC
        """
        cursor.execute(geo_query, params)
    else:
        geo_query = """
            SELECT state, COUNT(*) AS farmer_count
            FROM farmers
            GROUP BY state
            ORDER BY farmer_count DESC
        """
        cursor.execute(geo_query)
    
    geographic_data = cursor.fetchall()

    # -----------------------------
    # 10. Current Month Summary
    # -----------------------------
    current_month = dt_date.today().strftime("%Y-%m")

    # new farmers
    if location_filter:
        cursor.execute("""
            SELECT COUNT(*) AS new_farmers
            FROM farmers
            WHERE TO_CHAR(created_at, 'YYYY-MM') = %s
            AND (state ILIKE %s OR city ILIKE %s OR address ILIKE %s)
        """, (current_month, location_like, location_like, location_like))
    else:
        cursor.execute("""
            SELECT COUNT(*) AS new_farmers
            FROM farmers
            WHERE TO_CHAR(created_at, 'YYYY-MM') = %s
        """, (current_month,))
    
    new_farmers = cursor.fetchone()["new_farmers"] or 0

    # new cows
    if location_filter:
        cursor.execute("""
            SELECT COUNT(*) AS new_cows
            FROM cows
            WHERE TO_CHAR(created_at, 'YYYY-MM') = %s
            AND farmer_id IN (
                SELECT farmer_id FROM farmers
                WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
            )
        """, (current_month, location_like, location_like, location_like))
    else:
        cursor.execute("""
            SELECT COUNT(*) AS new_cows
            FROM cows
            WHERE TO_CHAR(created_at, 'YYYY-MM') = %s
        """, (current_month,))
    
    new_cows = cursor.fetchone()["new_cows"] or 0

    # milk this month
    if location_filter:
        cursor.execute("""
            SELECT COALESCE(SUM(total), 0) AS month_milk
            FROM milk_yield
            WHERE TO_CHAR(date, 'YYYY-MM') = %s
            AND cow_id IN (
                SELECT cow_id FROM cows
                WHERE farmer_id IN (
                    SELECT farmer_id FROM farmers
                    WHERE state ILIKE %s OR city ILIKE %s OR address ILIKE %s
                )
            )
        """, (current_month, location_like, location_like, location_like))
    else:
        cursor.execute("""
            SELECT COALESCE(SUM(total), 0) AS month_milk
            FROM milk_yield
            WHERE TO_CHAR(date, 'YYYY-MM') = %s
        """, (current_month,))
    
    month_milk = cursor.fetchone()["month_milk"] or 0

    # cows sold this month
    if location_filter:
        cursor.execute(f"""
            SELECT COUNT(*) AS cows_sold
            FROM cows_for_sale
            WHERE is_sold = TRUE AND TO_CHAR(listed_at, 'YYYY-MM') = %s
            {sales_where}
        """, (current_month, location_like, location_like, location_like))
    else:
        cursor.execute("""
            SELECT COUNT(*) AS cows_sold
            FROM cows_for_sale
            WHERE is_sold = TRUE AND TO_CHAR(listed_at, 'YYYY-MM') = %s
        """, (current_month,))
    
    cows_sold = cursor.fetchone()["cows_sold"] or 0

    current_month_stats = {
        "new_farmers": new_farmers,
        "new_cows": new_cows,
        "month_milk": month_milk,
        "cows_sold": cows_sold
    }

    # -----------------------------
    # Locations for dropdown
    # -----------------------------
    cursor.execute("SELECT DISTINCT state, city FROM farmers ORDER BY state, city")
    locations = cursor.fetchall()

    # -----------------------------
    # Get Vet Count for Dashboard
    # -----------------------------
    cursor.execute("SELECT COUNT(*) as vet_count FROM veterinarians")
    vet_count_result = cursor.fetchone()
    vet_count = vet_count_result["vet_count"] if vet_count_result else 0

    conn.close()

    return render_template(
        "admin_analytics.html",
        monthly_trends=monthly_trends,
        breed_distribution=breed_distribution,
        milk_analytics=milk_analytics,
        sales_analytics=sales_analytics,
        farmer_stats=farmer_stats,
        top_cows=top_cows,
        geographic_data=geographic_data,
        current_month_stats=current_month_stats,
        current_month=current_month,
        location_filter=location_filter,
        locations=locations,
        vet_count=vet_count  # 👈 ADD THIS LINE
    )

@app.route("/admin/reports")
def admin_reports():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()

    # Report parameters from request args
    report_type = request.args.get('type', 'farmers')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    # Base reports data
    reports_data = {}

    if report_type == 'farmers':
        # Farmers Report
        query = "SELECT * FROM farmers WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
            
        query += " ORDER BY created_at DESC"
        cursor.execute(query, params)
        reports_data['farmers'] = cursor.fetchall()

    elif report_type == 'cows':
        # Cows Report
        query = """
            SELECT c.*, f.name as farmer_name, f.phone as farmer_phone 
            FROM cows c 
            LEFT JOIN farmers f ON c.farmer_id = f.farmer_id 
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND DATE(c.created_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(c.created_at) <= %s"
            params.append(end_date)
            
        query += " ORDER BY c.created_at DESC"
        cursor.execute(query, params)
        reports_data['cows'] = cursor.fetchall()

    elif report_type == 'milk_production':
        # Milk Production Report
        query = """
            SELECT 
                m.date,
                c.cow_id,
                c.breed,
                f.name as farmer_name,
                m.morning,
                m.afternoon,
                m.evening,
                m.total
            FROM milk_yield m
            JOIN cows c ON m.cow_id = c.cow_id
            JOIN farmers f ON c.farmer_id = f.farmer_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND m.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND m.date <= %s"
            params.append(end_date)
            
        query += " ORDER BY m.date DESC, c.cow_id"
        cursor.execute(query, params)
        reports_data['milk_production'] = cursor.fetchall()

    elif report_type == 'sales':
        # Sales Report
        query = """
            SELECT 
                cfs.*,
                f.name as seller_name,
                f.phone as seller_phone,
                cfs.listed_at,
                CASE 
                    WHEN cfs.is_sold = TRUE THEN 'SOLD'
                    ELSE 'FOR SALE'
                END as status
            FROM cows_for_sale cfs
            JOIN farmers f ON cfs.farmer_id = f.farmer_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND DATE(cfs.listed_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(cfs.listed_at) <= %s"
            params.append(end_date)
            
        query += " ORDER BY cfs.listed_at DESC"
        cursor.execute(query, params)
        reports_data['sales'] = cursor.fetchall()

    elif report_type == 'appointments':
        # Appointments Report
        query = """
            SELECT 
                a.*,
                f.name as farmer_name,
                f.phone as farmer_phone,
                v.name as vet_name,
                v.clinic as vet_clinic
            FROM appointments a
            JOIN farmers f ON a.farmer_id = f.farmer_id
            JOIN veterinarians v ON a.vet_id = v.vet_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND a.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND a.date <= %s"
            params.append(end_date)
            
        query += " ORDER BY a.date DESC, a.time DESC"
        cursor.execute(query, params)
        reports_data['appointments'] = cursor.fetchall()

    # Summary statistics for the report
    cursor.execute("SELECT COUNT(*) as total_farmers FROM farmers")
    total_farmers = cursor.fetchone()["total_farmers"]

    cursor.execute("SELECT COUNT(*) as total_cows FROM cows")
    total_cows = cursor.fetchone()["total_cows"]

    cursor.execute("SELECT COALESCE(SUM(total), 0) as total_milk FROM milk_yield")
    total_milk = cursor.fetchone()["total_milk"]

    cursor.execute("SELECT COUNT(*) as total_sales, COALESCE(SUM(price), 0) as total_revenue FROM cows_for_sale WHERE is_sold = TRUE")
    sales_data = cursor.fetchone()

    conn.close()

    return render_template(
        "admin_reports.html",
        reports_data=reports_data,
        report_type=report_type,
        start_date=start_date,
        end_date=end_date,
        total_farmers=total_farmers,
        total_cows=total_cows,
        total_milk=total_milk,
        total_sales=sales_data["total_sales"],
        total_revenue=sales_data["total_revenue"]
    )

@app.route("/admin/export/<report_type>")
def export_report(report_type):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()

    # Get filter parameters from request args
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    if report_type == 'farmers':
        query = "SELECT * FROM farmers WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)
            
        query += " ORDER BY created_at DESC"
        cursor.execute(query, params)
        data = cursor.fetchall()
        filename = 'farmers_report.csv'
        headers = ['Farmer ID', 'Name', 'Email', 'Phone', 'State', 'City', 'Address', 'Registration Date']
        
        def row_formatter(row):
            return [
                row['farmer_id'], row['name'], row['email'], row['phone'],
                row['state'], row['city'], row['address'] or '', row['created_at']
            ]

    elif report_type == 'cows':
        query = """
            SELECT c.*, f.name as farmer_name 
            FROM cows c 
            LEFT JOIN farmers f ON c.farmer_id = f.farmer_id 
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND DATE(c.created_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(c.created_at) <= %s"
            params.append(end_date)
            
        query += " ORDER BY c.created_at DESC"
        cursor.execute(query, params)
        data = cursor.fetchall()
        filename = 'cows_report.csv'
        headers = ['Cow ID', 'Breed', 'Age', 'Weight', 'Color', 'Health Records', 'Vaccination History', 'Milk Yield', 'Farmer Name', 'Registration Date']
        
        def row_formatter(row):
            return [
                row['cow_id'], row['breed'], row['age'], row['weight'],
                row['color'], row['health_records'] or '', row['vaccination_history'] or '',
                row['milk_yield'], row['farmer_name'] or 'Unknown', row['created_at']
            ]

    elif report_type == 'milk_production':
        query = """
            SELECT 
                m.date,
                m.cow_id,
                c.breed,
                f.name as farmer_name,
                m.morning,
                m.afternoon,
                m.evening,
                m.total
            FROM milk_yield m
            JOIN cows c ON m.cow_id = c.cow_id
            JOIN farmers f ON c.farmer_id = f.farmer_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND m.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND m.date <= %s"
            params.append(end_date)
            
        query += " ORDER BY m.date DESC, c.cow_id"
        cursor.execute(query, params)
        data = cursor.fetchall()
        filename = 'milk_production_report.csv'
        headers = ['Date', 'Cow ID', 'Breed', 'Farmer Name', 'Morning (L)', 'Afternoon (L)', 'Evening (L)', 'Total (L)']
        
        def row_formatter(row):
            return [
                row['date'], row['cow_id'], row['breed'], row['farmer_name'],
                row['morning'], row['afternoon'], row['evening'], row['total']
            ]

    elif report_type == 'sales':
        query = """
            SELECT 
                cfs.*,
                f.name as seller_name
            FROM cows_for_sale cfs
            JOIN farmers f ON cfs.farmer_id = f.farmer_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND DATE(cfs.listed_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(cfs.listed_at) <= %s"
            params.append(end_date)
            
        query += " ORDER BY cfs.listed_at DESC"
        cursor.execute(query, params)
        data = cursor.fetchall()
        filename = 'sales_report.csv'
        headers = ['Sale ID', 'Cow ID', 'Breed', 'Age', 'Weight', 'Price (₹)', 'Seller Name', 'Status', 'Listed Date']
        
        def row_formatter(row):
            status = 'SOLD' if row['is_sold'] else 'FOR SALE'
            return [
                row['id'], row['cow_id'], row['breed'], row['age'],
                row['weight'], row['price'], row['seller_name'], status, row['listed_at']
            ]

    elif report_type == 'appointments':
        query = """
            SELECT 
                a.*,
                f.name as farmer_name,
                v.name as vet_name,
                v.clinic as vet_clinic
            FROM appointments a
            JOIN farmers f ON a.farmer_id = f.farmer_id
            JOIN veterinarians v ON a.vet_id = v.vet_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND a.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND a.date <= %s"
            params.append(end_date)
            
        query += " ORDER BY a.date DESC, a.time DESC"
        cursor.execute(query, params)
        data = cursor.fetchall()
        filename = 'appointments_report.csv'
        headers = ['Appointment ID', 'Date', 'Time', 'Farmer Name', 'Veterinarian Name', 'Clinic']
        
        def row_formatter(row):
            return [
                row['id'], row['date'], row['time'], row['farmer_name'],
                row['vet_name'], row['vet_clinic'] or 'N/A'
            ]

    else:
        conn.close()
        flash("Invalid report type", "danger")
        return redirect(url_for('admin_reports'))

    conn.close()

    # Generate CSV
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    
    for row in data:
        writer.writerow(row_formatter(row))
    
    output.seek(0)
    
    from flask import Response
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    """Convert Kannada text to speech and return audio file"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"})
        
        # Create audio directory if it doesn't exist
        audio_dir = os.path.join("static", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate unique filename
        audio_filename = f"speech_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Generate speech - gTTS supports Kannada!
        print(f"🎤 Generating speech for: {text[:50]}...")
        tts = gTTS(text=text, lang='kn', slow=False)
        tts.save(audio_path)
        print(f"✅ Audio saved: {audio_path}")
        
        # Return the URL to the audio file
        audio_url = url_for('static', filename=f'audio/{audio_filename}')
        
        # Also schedule cleanup of old files (optional)
        cleanup_old_audio_files()
        
        return jsonify({
            "success": True,
            "audio_url": audio_url
        })
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return jsonify({"success": False, "error": str(e)})
# Add this function for cleanup (optional but recommended)
def cleanup_old_audio_files():
    """Delete audio files older than 1 hour"""
    audio_dir = os.path.join("static", "audio")
    if os.path.exists(audio_dir):
        current_time = time.time()
        for filename in os.listdir(audio_dir):
            filepath = os.path.join(audio_dir, filename)
            if os.path.isfile(filepath) and (current_time - os.path.getmtime(filepath)) > 3600:
                try:
                    os.remove(filepath)
                except:
                    pass
@app.route("/speak", methods=["POST"])
def speak():
    """Convert Kannada text to speech and return audio URL"""
    try:
        text = request.json.get("text", "")
        if not text:
            return jsonify({"success": False, "error": "No text provided"})
        
        # Create audio directory if it doesn't exist
        audio_folder = os.path.join("static", "audio")
        os.makedirs(audio_folder, exist_ok=True)
        
        # Generate unique filename
        filename = f"speak_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(audio_folder, filename)
        
        # Generate speech using gTTS (supports Kannada)
        print(f"🔊 Generating speech for: {text[:50]}...")
        tts = gTTS(text=text, lang='kn', slow=False)
        tts.save(filepath)
        print(f"✅ Audio saved: {filepath}")
        
        # Clean up old files (optional)
        cleanup_old_audio_files()
        
        # Return the audio URL
        audio_url = url_for('static', filename=f'audio/{filename}')
        return jsonify({"success": True, "audio_url": audio_url})
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return jsonify({"success": False, "error": str(e)})
@app.route('/admin/cows')
def admin_cows():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT cow_id, breed, age, weight, color, health_records, vaccination_history, milk_yield, special_notes, farmer_id, photo,created_at FROM cows")
    cows = cursor.fetchall()
    conn.close()

    return render_template('admin_cows.html', cows=cows)

    
@app.route("/admin/farmers")
def admin_farmers():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT farmer_id, name, email, phone, state, city, address, photo, created_at FROM farmers")
    farmers = cursor.fetchall()
    conn.close()
    return render_template("admin_farmers.html", farmers=farmers)

@app.before_request
def before_request():
    """Debug: Print language info for every request"""
    lang_from_session = session.get('language')
    lang_from_cookie = request.cookies.get('user_language')
    lang_from_url = request.args.get('lang')
    print(f"🔍 DEBUG - Request: {request.path}")
    print(f"   Session lang: {lang_from_session}")
    print(f"   Cookie lang: {lang_from_cookie}")
    print(f"   URL lang: {lang_from_url}")
@app.route("/admin/delete_farmer/<int:farmer_id>", methods=["POST"])
def delete_farmer(farmer_id):
    if "admin" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Get farmer photo filename before deletion
        cursor.execute("SELECT photo FROM farmers WHERE farmer_id=%s", (farmer_id,))
        farmer = cursor.fetchone()
        
        # Delete the farmer
        cursor.execute("DELETE FROM farmers WHERE farmer_id=%s", (farmer_id,))
        conn.commit()
        
        # Delete the photo file if exists
        if farmer and farmer['photo']:
            photo_path = os.path.join(app.config["FARMER_UPLOAD_FOLDER"], farmer['photo'])
            if os.path.exists(photo_path):
                os.remove(photo_path)
        
        flash("Farmer deleted successfully!", "success")
    except Exception as e:
        conn.rollback()
        flash(f"Error deleting farmer: {str(e)}", "danger")
    finally:
        conn.close()

    return redirect(url_for("admin_farmers"))

@app.route("/admin/get_farmer_details/<int:farmer_id>")
def get_farmer_details(farmer_id):
    if "admin" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM farmers WHERE farmer_id=%s", (farmer_id,))
    farmer = cursor.fetchone()
    conn.close()

    if farmer:
        return jsonify(dict(farmer))
    else:
        return jsonify({"error": "Farmer not found"}), 404





@app.route("/farmer")
def farmer():
    return render_template("index.html")

@app.route("/veterinarian")
def veterinarian():
    return render_template("index2.html")

@app.route("/manage_vets")
def manage_vets():
    if "admin" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT vet_id, name, email, phone, clinic, education, experience, specialization, photo, created_at FROM veterinarians")
    vets = cursor.fetchall()
    conn.close()

    return render_template("admin_vets.html", vets=vets)

@app.route("/admin/delete_vet/<int:vet_id>", methods=["POST"])
def delete_vet(vet_id):
    if "admin" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Get vet photo filename before deletion
        cursor.execute("SELECT photo FROM veterinarians WHERE vet_id=%s", (vet_id,))
        vet = cursor.fetchone()
        
        # Delete the vet
        cursor.execute("DELETE FROM veterinarians WHERE vet_id=%s", (vet_id,))
        conn.commit()
        
        # Delete the photo file if exists
        if vet and vet['photo']:
            photo_path = os.path.join(app.config["VET_UPLOAD_FOLDER"], vet['photo'])
            if os.path.exists(photo_path):
                os.remove(photo_path)
        
        flash("Veterinarian deleted successfully!", "success")
    except Exception as e:
        conn.rollback()
        flash(f"Error deleting veterinarian: {str(e)}", "danger")
    finally:
        conn.close()

    return redirect(url_for("manage_vets"))

@app.route("/admin/get_vet_details/<int:vet_id>")
def get_vet_details(vet_id):
    if "admin" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM veterinarians WHERE vet_id=%s", (vet_id,))
    vet = cursor.fetchone()
    conn.close()

    if vet:
        return jsonify(dict(vet))
    else:
        return jsonify({"error": "Veterinarian not found"}), 404





@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        state = request.form["state"]
        city = request.form["city"]
        address = request.form["address"]
        password = request.form["password"]
        
        # Store form data in session to repopulate
        session['register_data'] = {
            'name': name, 'email': email, 'phone': phone,
            'state': state, 'city': city, 'address': address
        }
        
        # Handle photo upload
        photo_filename = None
        if "photo" in request.files:
            photo_file = request.files["photo"]
            if photo_file and photo_file.filename != "":
                file_extension = os.path.splitext(photo_file.filename)[1]
                photo_filename = f"farmer_{phone}_{int(time.time())}{file_extension}"
                photo_path = os.path.join(app.config["FARMER_UPLOAD_FOLDER"], photo_filename)
                photo_file.save(photo_path)

        conn = get_db()
        # Use RealDictCursor to get dictionary-like rows
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        try:
            # Check if email exists - use %s instead of %s
            cursor.execute("SELECT * FROM farmers WHERE email = %s", (email,))
            if cursor.fetchone():
                session['register_error'] = f"❌ Email '{email}' is already registered!"
                return redirect(url_for("register"))
            
            # Check if phone exists - use %s instead of %s
            cursor.execute("SELECT * FROM farmers WHERE phone = %s", (phone,))
            if cursor.fetchone():
                session['register_error'] = f"❌ Phone number '{phone}' is already registered!"
                return redirect(url_for("register"))
            
            # Insert new farmer - use %s instead of %s
            cursor.execute("""INSERT INTO farmers 
                            (name, email, phone, state, city, address, password, photo) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (name, email, phone, state, city, address, password, photo_filename))
            conn.commit()
            
            # Clear session data
            session.pop('register_data', None)
            session.pop('register_error', None)
            
            # Success message
            session['register_success'] = "✅ Registration successful! Please login."
            return redirect(url_for("login"))
            
        except psycopg2.Error as e:  # Changed from sqlite3.Error to psycopg2.Error
            session['register_error'] = f"❌ Registration failed: {str(e)}"
            return redirect(url_for("register"))
        finally:
            conn.close()
    
    return render_template("register.html")

# ---------------- Farmer Login ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        # Add cursor_factory here
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT * FROM farmers WHERE email=%s AND password=%s", (email, password))
        farmer = cursor.fetchone()
        conn.close()

        if farmer:
            session["farmer_id"] = farmer["farmer_id"]  # Now this works
            session["farmer_name"] = farmer["name"]      # Now this works
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!")
    return render_template("login.html")


#Forgot Password

@app.route("/forgot_password_modal", methods=["POST"])
def forgot_password_modal():
    """Handle forgot password from modal - Step 1: Send OTP"""
    phone = request.form.get("phone")
    
    if not phone:
        return jsonify({"success": False, "message": "Phone number is required"})
    
    # Check if phone exists in database
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM farmers WHERE phone = %s", (phone,))
    farmer = cursor.fetchone()
    conn.close()
    
    if not farmer:
        return jsonify({"success": False, "message": "Phone number not registered!"})
    
    # Generate and send OTP
    otp = generate_otp()
    save_otp(phone, otp)
    
    # Send OTP via SMS
    message = f"Your Kamadhenu password reset OTP is: {otp}. Valid for 10 minutes."
    sms_result = send_sms(phone, message)
    
    if sms_result['success']:
        # Store phone in session for verification
        session['reset_phone'] = phone
        return jsonify({"success": True, "message": f"OTP sent to {phone}", "phone": phone})
    else:
        return jsonify({"success": False, "message": "Failed to send OTP. Please try again."})

@app.route("/verify_otp_modal", methods=["POST"])
def verify_otp_modal():
    """Handle OTP verification from modal - Step 2: Verify OTP"""
    phone = session.get('reset_phone')
    otp = request.form.get("otp")
    
    if not phone or not otp:
        return jsonify({"success": False, "message": "Invalid request"})
    
    if verify_otp(phone, otp):
        session['otp_verified'] = True
        return jsonify({"success": True, "message": "OTP verified successfully!"})
    else:
        return jsonify({"success": False, "message": "Invalid or expired OTP!"})

@app.route("/reset_password_modal", methods=["POST"])
def reset_password_modal():
    """Handle password reset from modal - Step 3: Reset password"""
    phone = session.get('reset_phone')
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")
    
    # Check if OTP was verified
    if not session.get('otp_verified'):
        return jsonify({"success": False, "message": "Please verify OTP first"})
    
    if not phone or not new_password:
        return jsonify({"success": False, "message": "Invalid request"})
    
    if new_password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match!"})
    
    if len(new_password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters!"})
    
    # Update password
    if update_password(phone, new_password):
        # Clear session data
        session.pop('reset_phone', None)
        session.pop('otp_verified', None)
        return jsonify({"success": True, "message": "Password reset successfully! Please login."})
    else:
        return jsonify({"success": False, "message": "Failed to reset password"})

@app.route("/resend_otp_modal", methods=["POST"])
def resend_otp_modal():
    """Resend OTP from modal"""
    phone = session.get('reset_phone')
    
    if not phone:
        return jsonify({"success": False, "message": "Phone number not found in session"})
    
    # Generate and send new OTP
    otp = generate_otp()
    save_otp(phone, otp)
    
    # Send OTP via SMS
    message = f"Your Kamadhenu password reset OTP is: {otp}. Valid for 10 minutes."
    sms_result = send_sms(phone, message)
    
    if sms_result['success']:
        return jsonify({"success": True, "message": f"New OTP sent to {phone}"})
    else:
        return jsonify({"success": False, "message": "Failed to resend OTP"})

# Add to your existing OTP functions
def update_vet_password(phone, new_password):
    """Update vet's password in database"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE veterinarians SET password = %s WHERE phone = %s", (new_password, phone))
    
    if cursor.rowcount > 0:
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False


@app.route("/vet/forgot_password_modal", methods=["POST"])
def vet_forgot_password_modal():
    """Handle vet forgot password from modal - Step 1: Send OTP"""
    phone = request.form.get("phone")
    
    if not phone:
        return jsonify({"success": False, "message": "Phone number is required"})
    
    # Check if phone exists in vet database
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM veterinarians WHERE phone = %s", (phone,))
    vet = cursor.fetchone()
    conn.close()
    
    if not vet:
        return jsonify({"success": False, "message": "Phone number not registered!"})
    
    # Generate and send OTP
    otp = generate_otp()
    save_otp(phone, otp)
    
    # Send OTP via SMS
    message = f"Your Kamadhenu Vet password reset OTP is: {otp}. Valid for 10 minutes."
    sms_result = send_sms(phone, message)
    
    if sms_result['success']:
        # Store phone in session for verification
        session['vet_reset_phone'] = phone
        return jsonify({"success": True, "message": f"OTP sent to {phone}", "phone": phone})
    else:
        return jsonify({"success": False, "message": "Failed to send OTP. Please try again."})

@app.route("/vet/verify_otp_modal", methods=["POST"])
def vet_verify_otp_modal():
    """Handle vet OTP verification from modal - Step 2: Verify OTP"""
    phone = session.get('vet_reset_phone')
    otp = request.form.get("otp")
    
    if not phone or not otp:
        return jsonify({"success": False, "message": "Invalid request"})
    
    if verify_otp(phone, otp):
        session['vet_otp_verified'] = True
        return jsonify({"success": True, "message": "OTP verified successfully!"})
    else:
        return jsonify({"success": False, "message": "Invalid or expired OTP!"})

@app.route("/vet/reset_password_modal", methods=["POST"])
def vet_reset_password_modal():
    """Handle vet password reset from modal - Step 3: Reset password"""
    phone = session.get('vet_reset_phone')
    new_password = request.form.get("new_password")
    confirm_password = request.form.get("confirm_password")
    
    # Check if OTP was verified
    if not session.get('vet_otp_verified'):
        return jsonify({"success": False, "message": "Please verify OTP first"})
    
    if not phone or not new_password:
        return jsonify({"success": False, "message": "Invalid request"})
    
    if new_password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match!"})
    
    if len(new_password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters!"})
    
    # Update password
    if update_vet_password(phone, new_password):
        # Clear session data
        session.pop('vet_reset_phone', None)
        session.pop('vet_otp_verified', None)
        return jsonify({"success": True, "message": "Password reset successfully! Please login."})
    else:
        return jsonify({"success": False, "message": "Failed to reset password"})

@app.route("/vet/resend_otp_modal", methods=["POST"])
def vet_resend_otp_modal():
    """Resend OTP for vet from modal"""
    phone = session.get('vet_reset_phone')
    
    if not phone:
        return jsonify({"success": False, "message": "Phone number not found in session"})
    
    # Generate and send new OTP
    otp = generate_otp()
    save_otp(phone, otp)
    
    # Send OTP via SMS
    message = f"Your Kamadhenu Vet password reset OTP is: {otp}. Valid for 10 minutes."
    sms_result = send_sms(phone, message)
    
    if sms_result['success']:
        return jsonify({"success": True, "message": f"New OTP sent to {phone}"})
    else:
        return jsonify({"success": False, "message": "Failed to resend OTP"})







# ---------------- Dashboard ----------------
@app.route("/dashboard")
def dashboard():
    if "farmer_id" not in session:
        return redirect(url_for("login"))
    
    # Get language from multiple sources
    lang = session.get('language')
    
    if not lang:
        lang = request.cookies.get('user_language')
    
    if not lang:
        lang = request.args.get('lang')
    
    if lang not in ['en', 'kn']:
        lang = 'en'
    
    # Save language back to session for persistence
    session['language'] = lang
    
    conn = get_db()
    cursor = conn.cursor()

    # Get farmer details including photo
    cursor.execute("SELECT * FROM farmers WHERE farmer_id=%s", (session["farmer_id"],))
    farmer = cursor.fetchone()

    # Count total cows for this farmer
    cursor.execute("SELECT COUNT(*) as total FROM cows WHERE farmer_id=%s", (session["farmer_id"],))
    total_cows = cursor.fetchone()["total"]

    # Sum of milk yield for this farmer
    cursor.execute("SELECT SUM(milk_yield) as total_milk FROM cows WHERE farmer_id=%s", (session["farmer_id"],))
    total_milk = cursor.fetchone()["total_milk"] or 0

    # Count upcoming appointments (only scheduled, not completed)
    cursor.execute("""SELECT COUNT(*) as appointments 
                      FROM appointments 
                      WHERE farmer_id=%s AND date >= DATE('now') 
                      AND status != 'completed'""", 
                   (session["farmer_id"],))
    upcoming_appointments = cursor.fetchone()["appointments"]

    conn.close()

    return render_template(
        "dashboard.html",
        farmer_name=session["farmer_name"],
        farmer_id=session["farmer_id"],
        farmer_email=farmer["email"],
        farmer_phone=farmer["phone"],
        farmer_state=farmer["state"],
        farmer_city=farmer["city"],
        farmer_address=farmer["address"],
        farmer_photo=farmer["photo"],
        farmer_created_at=farmer["created_at"].strftime('%Y-%m-%d') if farmer["created_at"] else None,
        total_cows=total_cows,
        total_milk=total_milk,
        upcoming_appointments=upcoming_appointments,
        current_lang=lang  # IMPORTANT: Pass language to template
    )
@app.route("/update_profile", methods=["POST"])
def update_profile():
    if "farmer_id" not in session:
        return jsonify({"success": False, "message": "Please login first!"})

    farmer_id = session["farmer_id"]
    lang = session.get('language', request.cookies.get('user_language', 'en'))

    # Get form data
    name = request.form["name"]
    email = request.form["email"]
    phone = request.form["phone"]
    state = request.form["state"]
    city = request.form["city"]
    address = request.form.get("address", "")

    # Handle photo upload
    photo_filename = None
    if "photo" in request.files:
        photo_file = request.files["photo"]
        if photo_file and photo_file.filename != "":
            # Generate secure filename
            file_extension = os.path.splitext(photo_file.filename)[1]
            photo_filename = f"farmer_{farmer_id}_{int(time.time())}{file_extension}"
            photo_path = os.path.join(app.config["FARMER_UPLOAD_FOLDER"], photo_filename)
            photo_file.save(photo_path)

    conn = get_db()
    cursor = conn.cursor()
    
    try:
        if photo_filename:
            # Update with photo
            cursor.execute("""UPDATE farmers 
                            SET name=%s, email=%s, phone=%s, state=%s, city=%s, address=%s, photo=%s
                            WHERE farmer_id=%s""",
                         (name, email, phone, state, city, address, photo_filename, farmer_id))
        else:
            # Update without photo
            cursor.execute("""UPDATE farmers 
                            SET name=%s, email=%s, phone=%s, state=%s, city=%s, address=%s
                            WHERE farmer_id=%s""",
                         (name, email, phone, state, city, address, farmer_id))
        
        conn.commit()
        session["farmer_name"] = name  # Update session name
        return jsonify({"success": True, "message": "Profile updated successfully!"})
        
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email or phone already exists!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    finally:
        conn.close()
# ---------------- Add Cow (with Photo) ----------------
@app.route("/add_cow", methods=["GET", "POST"])
def add_cow():
    if "farmer_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    # Get breeds for dropdown
    cursor.execute("SELECT breed_name, cattle_type FROM breeds ORDER BY breed_name")
    breeds = cursor.fetchall()

    # Get existing cows for parent selection (only for current farmer)
    cursor.execute("SELECT cow_id, breed, cattle_type FROM cows WHERE farmer_id=%s", (session["farmer_id"],))
    existing_cows = cursor.fetchall()

    if request.method == "POST":
        # Handle cattle type (regular or "other" with custom input)
        cattle_type = request.form["cattle_type"]
        if cattle_type == 'other' and 'cattle_type_other' in request.form:
            cattle_type = request.form['cattle_type_other'].strip().lower()
        
        # Handle breed (regular or "other" with custom input)
        breed = request.form["breed"]
        if breed == 'other' and 'breed_other' in request.form:
            breed = request.form['breed_other'].strip()
        
        # Validate milk yield based on cattle type
        milk_yield = None
        cattle_type_lower = cattle_type.lower()
        
        # Define cattle type categories
        milk_producing_types = ['cow', 'buffalo']
        non_milk_producing_types = ['bull', 'male_buffalo', 'calf']
        
        # For milk-producing types: require milk yield
        if cattle_type_lower in milk_producing_types:
            milk_yield = request.form.get("milk_yield")
            if not milk_yield:
                flash('Milk Yield is required for milk-producing cattle (Cows and Female Buffalo)', 'error')
                return render_template("add_cow.html",
                                     pending_muzzle=session.get('pending_muzzle'),
                                     breeds=breeds,
                                     existing_cows=existing_cows)
        
        # For non-milk-producing types: ensure NO milk yield
        elif cattle_type_lower in non_milk_producing_types:
            milk_yield = None  # Explicitly set to None
            if request.form.get("milk_yield"):
                flash(f'{cattle_type.title()} does not produce milk. Milk yield should not be provided.', 'error')
                return render_template("add_cow.html",
                                     pending_muzzle=session.get('pending_muzzle'),
                                     breeds=breeds,
                                     existing_cows=existing_cows)
        
        # For "other" type (if user entered "other" in the textbox)
        elif cattle_type_lower == 'other':
            # Check if it's likely milk-producing
            other_type_input = request.form.get('cattle_type_other', '').lower()
            milk_keywords = ['cow', 'buffalo', 'dairy', 'milch', 'milk', 'female']
            non_milk_keywords = ['bull', 'male', 'calf', 'steer', 'ox', 'bullock']
            
            has_milk_keyword = any(keyword in other_type_input for keyword in milk_keywords)
            has_non_milk_keyword = any(keyword in other_type_input for keyword in non_milk_keywords)
            
            if has_milk_keyword and not has_non_milk_keyword:
                # Likely milk-producing, require milk yield
                milk_yield = request.form.get("milk_yield")
                if not milk_yield:
                    flash(f'{other_type_input.title()} appears to be milk-producing. Please enter milk yield.', 'error')
                    return render_template("add_cow.html",
                                         pending_muzzle=session.get('pending_muzzle'),
                                         breeds=breeds,
                                         existing_cows=existing_cows)
            elif has_non_milk_keyword:
                # Likely non-milk-producing, no milk yield
                milk_yield = None
                if request.form.get("milk_yield"):
                    flash(f'{other_type_input.title()} does not produce milk. Please remove milk yield.', 'error')
                    return render_template("add_cow.html",
                                         pending_muzzle=session.get('pending_muzzle'),
                                         breeds=breeds,
                                         existing_cows=existing_cows)
            else:
                # Ambiguous case, accept if provided
                milk_yield = request.form.get("milk_yield") or None
        
        # Generate cow ID
        cow_id = "COW-" + str(uuid.uuid4().hex[:6].upper())
        farmer_id = session["farmer_id"]
        date_of_birth = request.form["date_of_birth"]
        age = request.form["age"]
        weight = request.form["weight"]
        color = request.form["color"]
        health_records = request.form["health_records"]
        vaccination_history = request.form["vaccination_history"]
        special_notes = request.form["special_notes"]
        
        # Get insurance details
        insurance_by = request.form.get("insurance_by")
        insurance_policy_number = request.form.get("insurance_policy_number")
        insurance_valid_upto = request.form.get("insurance_valid_upto")
        
        # Get parent IDs only for calves
        father_id = None
        mother_id = None
        if cattle_type_lower == 'calf':
            father_id = request.form.get("father_id") or None
            mother_id = request.form.get("mother_id") or None

        # Handle photo upload
        photo = None
        if "photo" in request.files:
            file = request.files["photo"]
            if file.filename != "":
                filename = f"{cow_id}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config["COW_UPLOAD_FOLDER"], filename)
                file.save(filepath)
                photo = filename

        # Handle muzzle registration
        muzzle_id = None
        muzzle_photo = None
        if 'pending_muzzle' in session:
            muzzle_data = session['pending_muzzle']
            muzzle_id = muzzle_data['muzzle_id']
            muzzle_photo = muzzle_data['muzzle_filename']
            link_muzzle_to_cow(muzzle_id, cow_id)
            session.pop('pending_muzzle')

        # Insert into DB with new fields
        try:
            cursor.execute("""INSERT INTO cows 
                              (cow_id, farmer_id, cattle_type, breed, date_of_birth, age, weight, color, health_records, 
                               vaccination_history, milk_yield, special_notes, photo, muzzle_id, muzzle_photo,
                               father_id, mother_id, insurance_by, insurance_policy_number, insurance_valid_upto) 
                              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                           (cow_id, farmer_id, cattle_type, breed, date_of_birth, age, weight, color, health_records,
                            vaccination_history, milk_yield, special_notes, photo, muzzle_id, muzzle_photo,
                            father_id, mother_id, insurance_by, insurance_policy_number, insurance_valid_upto))
            conn.commit()
        except Exception as e:
            conn.rollback()
            flash(f'Error saving cow profile: {str(e)}', 'error')
            return render_template("add_cow.html",
                                 pending_muzzle=session.get('pending_muzzle'),
                                 breeds=breeds,
                                 existing_cows=existing_cows)
        finally:
            conn.close()

        # Generate QR Code
        try:
            qr_data = f"https://kamadhenu-zrxm.onrender.com/cow/{cow_id}"
            qr_img = qrcode.make(qr_data)
            qr_path = os.path.join(QR_FOLDER, f"{cow_id}.png")
            qr_img.save(qr_path)
        except Exception as e:
            print(f"Error generating QR code: {e}")

        flash(f"Cow profile added successfully! Cattle Type: {cattle_type.title()}", "success")
        return redirect(url_for("list_cows"))

    # Check if there's a pending muzzle registration
    pending_muzzle = session.get('pending_muzzle')
    
    conn.close()
    
    return render_template("add_cow.html", 
                         pending_muzzle=pending_muzzle,
                         breeds=breeds,
                         existing_cows=existing_cows)

@app.route("/capture_muzzle")
def capture_muzzle():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    """Route to capture muzzle and generate automatic ID"""
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))
    
    try:
        # Automatically register muzzle and generate ID
        muzzle_id, muzzle_filename = register_muzzle_automatically()
        
        if muzzle_id and muzzle_filename:
            # Store muzzle info in session for linking when cow is saved
            session['pending_muzzle'] = {
                'muzzle_id': muzzle_id,
                'muzzle_filename': muzzle_filename
            }
            
            flash(f"Muzzle successfully registered with ID: {muzzle_id}! Now fill cow details and save.", "success")
        else:
            flash("Muzzle registration cancelled or failed.", "warning")
            
    except Exception as e:
        flash(f"Error during muzzle registration: {str(e)}", "danger")
        print(f"Muzzle registration error: {e}")
    
    return redirect(url_for("add_cow"))

@app.route("/clear_muzzle")
def clear_muzzle():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    """Clear pending muzzle registration"""
    session.pop('pending_muzzle', None)
    flash("Muzzle registration cleared. You can register a new one.", "info")
    return redirect(url_for("add_cow"))




# Load YOLO muzzle model
#yolo_model = YOLO("best_new.pt")


def extract_features(crop):
    """Extract features from muzzle image"""
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_muzzle_database():
    """Load registered muzzle embeddings"""
    try:
        with open(DB_FILE, "r") as f:
            db = json.load(f)
            return db
    except:
        return {}

def identify_cow_from_muzzle():
    """Fully automatic cow identification using muzzle recognition"""
    db = load_muzzle_database()
    if not db:
        return None, "No registered muzzles found in database!"
    
    cap = cv2.VideoCapture(0)
    cow_details = None
    message = None
    
    print("🔍 Starting automatic cow identification...")
    print("📸 Scanning for muzzle patterns...")
    
    # Set a timeout to avoid infinite loop
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            break

        model = get_muzzle_model()
        results = model(frame)[0]
        best_detection = None
        max_confidence = 0

        # Find the best detection with highest confidence
        for box in results.boxes:
            confidence = box.conf[0]
            if confidence > max_confidence and confidence > CONFIDENCE_THRESHOLD:
                max_confidence = confidence
                best_detection = box

        if best_detection is not None:
            x1, y1, x2, y2 = map(int, best_detection.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Muzzle Detected", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {max_confidence:.2f}", (x1, y1-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Identifying...", (x1, y1-70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Wait a moment to ensure stable detection
            cv2.imshow("Cow Identification - Automatic", frame)
            cv2.waitKey(500)  # Wait 0.5 seconds for stable detection
            
            print(f"✅ Muzzle detected with confidence: {max_confidence:.2f}")
            print("🔍 Identifying cow...")
            
            # Identify cow from muzzle
            new_feat = extract_features(crop)
            best_match = None
            best_score = 0
            
            # Compare with all registered muzzles
            for muzzle_id, data in db.items():
                if 'features' in data and data.get('cow_id'):
                    ref_feat = np.array(data['features'])
                    # Cosine similarity
                    score = np.dot(new_feat, ref_feat) / (np.linalg.norm(new_feat) * np.linalg.norm(ref_feat))
                    
                    if score > best_score and score > IDENTIFICATION_THRESHOLD:
                        best_score = score
                        best_match = data['cow_id']
            
            if best_match:
                # Get cow details from database
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM cows WHERE cow_id=%s", (best_match,))
                cow = cursor.fetchone()
                conn.close()
                
                if cow:
                    cow_details = dict(cow)
                    message = f"Cow identified: {best_match} (Score: {best_score:.2f})"
                    print(f"✅ {message}")
                    
                    # Show success message
                    cv2.putText(frame, f"IDENTIFIED: {best_match}", (50, frame.shape[0]-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Score: {best_score:.2f}", (50, frame.shape[0]-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Auto-closing...", (50, frame.shape[0]-90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("Cow Identification - Automatic", frame)
                    cv2.waitKey(2000)  # Show success for 2 seconds
                else:
                    message = "Cow identified but details not found in database"
            else:
                message = "No matching cow found in database"
                print("❌ No matching cow found")
                
                # Show no match message
                cv2.putText(frame, "No matching cow found", (50, frame.shape[0]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Try again with different angle", (50, frame.shape[0]-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Cow Identification - Automatic", frame)
                cv2.waitKey(2000)  # Show message for 2 seconds
            
            break  # Exit after identification attempt

        else:
            # No muzzle detected
            cv2.putText(frame, "Scanning for muzzle...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Position cow's muzzle in frame", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time remaining: {int(timeout - (time.time() - start_time))}s", 
                        (50, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Cow Identification - Automatic", frame)
        
        # Check for manual exit (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Manual exit requested")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Timeout message
    if time.time() - start_time >= timeout and not cow_details:
        message = "Identification timeout. Please try again."
        print("⏰ Identification timeout")
    
    return cow_details, message

@app.route("/web_identify_cow")
def web_identify_cow():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    """Web-based automatic cow identification page"""
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))
    return render_template("web_identify_cow.html")

@app.route("/start_automatic_scan", methods=["POST"])
def start_automatic_scan():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    """Start automatic scanning and identification"""
    if "farmer_id" not in session:
        return jsonify({"success": False, "message": "Please login first!"})

    try:
        # Get image data from frontend
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"success": False, "message": "No image data received"})

        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Failed to process image"})

        # Detect muzzle using YOLO
        model = get_muzzle_model()
        results = model(frame)[0]
        cow_details = None
        message = "No muzzle detected"

        for box in results.boxes:
            confidence = box.conf[0]
            if confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                # Extract features and identify cow
                new_feat = extract_features(crop)
                db = load_muzzle_database()
                
                best_match = None
                best_score = 0
                
                # Compare with all registered muzzles
                for muzzle_id, data in db.items():
                    if 'features' in data and data.get('cow_id'):
                        ref_feat = np.array(data['features'])
                        # Cosine similarity
                        score = np.dot(new_feat, ref_feat) / (np.linalg.norm(new_feat) * np.linalg.norm(ref_feat))
                        
                        if score > best_score and score > IDENTIFICATION_THRESHOLD:
                            best_score = score
                            best_match = data['cow_id']
                
                if best_match:
                    # Get cow details from database
                    conn = get_db()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT c.*, f.name as farmer_name, f.phone as farmer_phone
                        FROM cows c 
                        LEFT JOIN farmers f ON c.farmer_id = f.farmer_id 
                        WHERE c.cow_id = %s AND c.farmer_id = %s
                    """, (best_match, session["farmer_id"]))
                    cow = cursor.fetchone()
                    conn.close()
                    
                    if cow:
                        cow_details = dict(cow)
                        message = f"Cow identified: {best_match} (Score: {best_score:.2f})"
                        break

        if cow_details:
            return jsonify({
                "success": True,
                "cow_details": cow_details,
                "message": message,
                "confidence": best_score
            })
        else:
            return jsonify({
                "success": False,
                "message": message
            })

    except Exception as e:
        print(f"Identification error: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/find_cow")
def find_cow():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    """Find Cow page - Identify cow using muzzle recognition"""
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))
    return render_template("find_cow.html")

@app.route("/identify_cow")
def identify_cow():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    """Identify cow using muzzle recognition"""
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))
    
    try:
        # Run muzzle identification
        cow_details, message = identify_cow_from_muzzle()
        
        if cow_details:
            # Check if the identified cow belongs to the current farmer
            if cow_details.get('farmer_id') == session["farmer_id"]:
                flash(f"Cow identified successfully: {cow_details['cow_id']}", "success")
                # Get complete cow details with farmer info
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.*, f.name as farmer_name, f.phone as farmer_phone
                    FROM cows c 
                    LEFT JOIN farmers f ON c.farmer_id = f.farmer_id 
                    WHERE c.cow_id = %s
                """, (cow_details['cow_id'],))
                cow = cursor.fetchone()
                conn.close()
                
                if cow:
                    return render_template("cow_details.html", cow=dict(cow))
                else:
                    flash("Cow details not found", "danger")
                    return redirect(url_for("find_cow"))
            else:
                flash("Identified cow does not belong to your farm.", "warning")
                return redirect(url_for("find_cow"))
        else:
            if message:
                flash(message, "warning")
            else:
                flash("No cow identified. Please try again.", "warning")
            return redirect(url_for("find_cow"))
            
    except Exception as e:
        flash(f"Error during identification: {str(e)}", "danger")
        return redirect(url_for("find_cow"))



# registration 

# ================= MUZZLE REGISTRATION FUNCTIONS ==================

def save_muzzle_database(db):
    """Save muzzle database to file"""
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

def generate_muzzle_id():
    """Generate unique muzzle ID with MUZ- prefix"""
    return f"MUZ-{uuid.uuid4().hex[:8].upper()}"

def save_muzzle_image(crop, muzzle_id):
    """Save muzzle image to folder with muzzle ID"""
    filename = f"{muzzle_id}.jpg"
    filepath = os.path.join(MUZZLE_FOLDER, filename)
    cv2.imwrite(filepath, crop)
    return filename

def register_muzzle_automatically():
    """Automatically register muzzle and return muzzle ID and filename"""
    cap = cv2.VideoCapture(0)  # webcam
    muzzle_id = generate_muzzle_id()
    registered = False
    muzzle_filename = None

    print(f"🔴 Auto-generating Muzzle ID: {muzzle_id}")
    print("📸 Press SPACE to CAPTURE muzzle")
    print("❌ Press Q to quit without saving")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        model = get_muzzle_model()
        results = model(frame)[0]
        muzzle_detected = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            muzzle_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Muzzle Detected - {muzzle_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "PRESS SPACE to CAPTURE", (x1, y1-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if not muzzle_detected:
            cv2.putText(frame, "No muzzle detected - Adjust camera", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display generated muzzle ID on screen
        cv2.putText(frame, f"Muzzle ID: {muzzle_id}", (50, frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Muzzle Registration - Auto ID Generation", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and muzzle_detected:  # SPACE to capture
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                # Extract features and save to database
                emb = extract_features(crop)
                db = load_muzzle_database()
                db[muzzle_id] = {
                    'features': emb.tolist(),  # Convert numpy array to list for JSON
                    'timestamp': datetime.now().isoformat(),
                    'cow_id': None  # Will be linked when cow is saved
                }
                save_muzzle_database(db)
                
                # Save muzzle image
                muzzle_filename = save_muzzle_image(crop, muzzle_id)
                registered = True
                
                print(f"✔ Muzzle registered with ID: {muzzle_id}!")
                break
            break

        elif key == ord('q'):
            print("❌ Muzzle registration cancelled")
            muzzle_id = None
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if registered:
        return muzzle_id, muzzle_filename
    else:
        return None, None

def link_muzzle_to_cow(muzzle_id, cow_id):
    """Link a muzzle ID to a cow ID in the database"""
    db = load_muzzle_database()
    if muzzle_id in db:
        db[muzzle_id]['cow_id'] = cow_id
        save_muzzle_database(db)
        print(f"✔ Linked muzzle {muzzle_id} to cow {cow_id}")
        return True
    return False

@app.route("/web_muzzle_registration")
def web_muzzle_registration():
    """Web-based muzzle registration page"""
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))
    return render_template("web_muzzle_registration.html")

@app.route("/capture_muzzle_web", methods=["POST"])
def capture_muzzle_web():
    """Capture muzzle from web camera"""
    if "farmer_id" not in session:
        return jsonify({"success": False, "message": "Please login first!"})

    try:
        # Get image data from frontend
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"success": False, "message": "No image data received"})

        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Failed to process image"})

        # Detect muzzle using YOLO
        model = get_muzzle_model()
        results = model(frame)[0]
        muzzle_detected = False
        muzzle_id = generate_muzzle_id()
        muzzle_filename = None

        for box in results.boxes:
            confidence = box.conf[0]
            if confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                muzzle_detected = True

                # Extract features and save to database
                emb = extract_features(crop)
                db = load_muzzle_database()
                db[muzzle_id] = {
                    'features': emb.tolist(),
                    'timestamp': datetime.now().isoformat(),
                    'cow_id': None
                }
                save_muzzle_database(db)

                # Save muzzle image
                muzzle_filename = save_muzzle_image(crop, muzzle_id)
                break

        if muzzle_detected:
            # Store muzzle info in session for linking when cow is saved
            session['pending_muzzle'] = {
                'muzzle_id': muzzle_id,
                'muzzle_filename': muzzle_filename
            }

            return jsonify({
                "success": True,
                "message": f"Muzzle registered successfully! ID: {muzzle_id}",
                "muzzle_id": muzzle_id
            })
        else:
            return jsonify({
                "success": False,
                "message": "No muzzle detected. Please ensure the muzzle is clearly visible."
            })

    except Exception as e:
        print(f"Muzzle registration error: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


# Disease prediction route
@app.route("/disease_prediction", methods=["GET", "POST"])
def disease_prediction():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        if file:
            try:
                # Save uploaded file temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join("static", "temp", filename)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                file.save(temp_path)
                
                # Load your disease prediction model
                # Replace "disease_model.pt" with your actual model path
                disease_model = get_disease_model()
                results = disease_model.predict(source=temp_path, conf=0.25, save=False)
                
                # Process results
                prediction_data = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        class_name = disease_model.names[cls]
                        conf = float(box.conf[0])
                        prediction_data.append({
                            'disease': class_name,
                            'confidence': f"{conf:.3f}",
                            'percentage': f"{conf*100:.1f}%"
                        })
                
                # Generate result image with bounding boxes
                result_image = r.plot()
                
                # Convert result image to base64 for display
                _, buffer = cv2.imencode('.jpg', result_image)
                result_image_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Clean up temporary file
                os.remove(temp_path)
                
                return render_template("disease_prediction.html", 
                                    prediction_data=prediction_data,
                                    result_image=result_image_b64,
                                    filename=filename)
                
            except Exception as e:
                flash(f'Error during prediction: {str(e)}', 'danger')
                return redirect(request.url)
    
    return render_template("disease_prediction.html")



@app.route("/cow/<cow_id>")
def cow_details(cow_id):
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    conn = get_db()
    cursor = conn.cursor()
    
    # Clean the cow_id to handle any whitespace or case issues
    cleaned_cow_id = cow_id.strip()
    
    # Get cow details - try case-insensitive search first
    cursor.execute("SELECT * FROM cows WHERE UPPER(cow_id) = UPPER(%s)", (cleaned_cow_id,))
    cow = cursor.fetchone()
    
    if cow:
        # Get farmer details
        cursor.execute("SELECT * FROM farmers WHERE farmer_id=%s", (cow["farmer_id"],))
        farmer = cursor.fetchone()
        
        # Use the actual cow_id from database for QR path
        actual_cow_id = cow["cow_id"]
        qr_path = f"/static/qrcodes/{actual_cow_id}.png"
        
        conn.close()
        
        return render_template("cow_details.html", 
                             cow=cow, 
                             farmer=farmer,
                             farmer_photo=farmer["photo"] if farmer else None,
                             qr=qr_path)
    else:
        conn.close()
        return "Cow not found!", 404

# ---------------- List Cows ----------------
@app.route("/cows")
def list_cows():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    if "farmer_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()
    
    # Show only cows that belong to farmer AND are not sold
    cursor.execute("""
    SELECT c.* FROM cows c
    WHERE c.farmer_id = %s 
    AND c.cow_id NOT IN (
        SELECT cow_id FROM cows_for_sale WHERE is_sold = TRUE
    )
""", (session["farmer_id"],))
    
    cows = cursor.fetchall()
    
    # Count different cattle types for statistics
    cursor.execute("""
        SELECT cattle_type, COUNT(*) as count 
        FROM cows 
        WHERE farmer_id = %s 
        AND cow_id NOT IN (SELECT cow_id FROM cows_for_sale WHERE is_sold = TRUE)
        GROUP BY cattle_type
    """, (session["farmer_id"],))
    
    cattle_counts = cursor.fetchall()
    
    # Convert to dictionary for easy access in template
    cattle_stats = {}
    for row in cattle_counts:
        cattle_stats[row['cattle_type']] = row['count']
    
    conn.close()
    
    return render_template("list_cows.html", cows=cows, cattle_stats=cattle_stats)


# ---------------- Delete Cow ----------------
@app.route("/delete_cow/<cow_id>", methods=["POST"])
def delete_cow(cow_id):
    if "farmer_id" not in session:
        flash("Unauthorized access!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    # Fetch cow details to check ownership and files
    cursor.execute("SELECT * FROM cows WHERE cow_id=%s AND farmer_id=%s", (cow_id, session["farmer_id"]))
    cow = cursor.fetchone()

    if not cow:
        conn.close()
        flash("Cow not found or you don't have permission!", "danger")
        return redirect(url_for("list_cows"))

    # ✅ DON'T delete the photo file at all for purchased cows
    # The photo might be used by sold_cows, so we'll keep it
    # Only delete QR code (which is safe to delete)

    # Delete from DB
    cursor.execute("DELETE FROM cows WHERE cow_id=%s AND farmer_id=%s", (cow_id, session["farmer_id"]))
    conn.commit()
    conn.close()

    # Remove QR code if exists (this is safe to delete)
    qr_path = os.path.join(QR_FOLDER, f"{cow_id}.png")
    if os.path.exists(qr_path):
        os.remove(qr_path)

    flash("Cow deleted successfully!", "success")
    return redirect(url_for("list_cows"))


@app.route("/buy_cow")
def buy_cow():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    # Fetch cows that are for sale (is_sold = FALSE) and not from current farmer
    cursor.execute("""
        SELECT cfs.*, f.name AS farmer_name 
        FROM cows_for_sale cfs
        JOIN farmers f ON cfs.farmer_id = f.farmer_id
        WHERE cfs.farmer_id != %s AND cfs.is_sold = FALSE
        ORDER BY cfs.listed_at DESC
    """, (session["farmer_id"],))

    cows_for_sale = cursor.fetchall()
    conn.close()

    return render_template("buy_cow.html", cows=cows_for_sale)


@app.route("/sell_cow", methods=["GET", "POST"])
def sell_cow():
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    # Fetch cows of this farmer that are NOT already listed for sale
    cursor.execute("""
        SELECT * FROM cows 
        WHERE farmer_id=%s 
        AND cow_id NOT IN (SELECT cow_id FROM cows_for_sale WHERE is_sold = FALSE)
    """, (session["farmer_id"],))
    cows = cursor.fetchall()

    if request.method == "POST":
        cow_id = request.form["cow_id"]
        price = float(request.form["price"])

        # Get cow details
        cursor.execute("SELECT * FROM cows WHERE cow_id=%s AND farmer_id=%s", (cow_id, session["farmer_id"]))
        cow = cursor.fetchone()

        if not cow:
            flash("Cow not found!", "danger")
            return redirect(url_for("sell_cow"))

        # Insert into cows_for_sale table (NOT sold yet)
        cursor.execute("""INSERT INTO cows_for_sale 
                          (cow_id, farmer_id, breed, age, weight, price, photo, is_sold)
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                       (cow["cow_id"], session["farmer_id"], cow["breed"], cow["age"], 
                        cow["weight"], price, cow["photo"], False))  # is_sold = False (Python boolean)

        conn.commit()
        flash(f"Cow {cow['breed']} listed for sale successfully!", "success")
        return redirect(url_for("dashboard"))

    conn.close()
    return render_template("sell_cow.html", cows=cows)

@app.route("/sold_cows")
def sold_cows():
    lang = session.get('language', request.cookies.get('user_language', 'en'))
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()
    
    # Get all cows that this farmer has listed for sale (both sold and unsold)
    cursor.execute("""
        SELECT cfs.*, 
               CASE 
                   WHEN cfs.is_sold = TRUE THEN 'SOLD'
                   ELSE 'FOR SALE'
               END as status
        FROM cows_for_sale cfs
        WHERE cfs.farmer_id = %s
        ORDER BY cfs.listed_at DESC
    """, (session["farmer_id"],))
    
    cows = cursor.fetchall()
    conn.close()
    
    return render_template("sold_cows.html", cows=cows)


@app.route("/purchase_cow/<int:sale_id>", methods=["POST"])
def purchase_cow(sale_id):
    lang = session.get('language', request.cookies.get('user_language', 'en'))

    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Get the cow for sale details
        cursor.execute("SELECT * FROM cows_for_sale WHERE id=%s AND is_sold=FALSE", (sale_id,))
        cow_for_sale = cursor.fetchone()

        if not cow_for_sale:
            flash("Cow not available for purchase!", "danger")
            return redirect(url_for("buy_cow"))

        # Get the original cow details from cows table to copy all information
        cursor.execute("SELECT * FROM cows WHERE cow_id=%s", (cow_for_sale["cow_id"],))
        original_cow = cursor.fetchone()

        if not original_cow:
            flash("Original cow details not found!", "danger")
            return redirect(url_for("buy_cow"))

        # Mark the cow as sold in cows_for_sale table
        cursor.execute("UPDATE cows_for_sale SET is_sold=TRUE WHERE id=%s", (sale_id,))

        # Remove from original owner's cows table
        cursor.execute("DELETE FROM cows WHERE cow_id=%s AND farmer_id=%s", 
                      (cow_for_sale["cow_id"], cow_for_sale["farmer_id"]))

        # Handle photo copying for sold_cows
        original_photo = original_cow["photo"]
        sold_photo_filename = None
        buyer_photo_filename = original_photo  # Buyer gets original filename

        if original_photo:
            try:
                # Create a unique filename for the sold cow photo
                file_extension = os.path.splitext(original_photo)[1]
                sold_photo_filename = f"sold_{cow_for_sale['cow_id']}_{int(time.time())}{file_extension}"
                
                original_photo_path = os.path.join(app.config["COW_UPLOAD_FOLDER"], original_photo)
                sold_photo_path = os.path.join(app.config["COW_UPLOAD_FOLDER"], sold_photo_filename)
                
                # Copy the photo file for sold_cows
                if os.path.exists(original_photo_path):
                    import shutil
                    shutil.copy2(original_photo_path, sold_photo_path)
                    print(f"Photo copied: {original_photo} -> {sold_photo_filename}")
                else:
                    print(f"Original photo not found: {original_photo_path}")
                    sold_photo_filename = original_photo  # Fallback to original name
                    
            except Exception as e:
                print(f"Error copying photo: {e}")
                sold_photo_filename = original_photo  # Fallback to original name

        # Add to buyer's cows table with ALL original details
        new_cow_id = "COW-" + str(uuid.uuid4().hex[:6].upper())
        cursor.execute("""INSERT INTO cows 
                          (cow_id, farmer_id, breed, age, weight, color, health_records, 
                           vaccination_history, milk_yield, special_notes, photo) 
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                       (new_cow_id, 
                        session["farmer_id"],  # buyer's farmer_id
                        original_cow["breed"], 
                        original_cow["age"], 
                        original_cow["weight"], 
                        original_cow["color"],
                        original_cow["health_records"],
                        original_cow["vaccination_history"],
                        original_cow["milk_yield"],
                        f"Purchased cow. Previous ID: {original_cow['cow_id']}",
                        buyer_photo_filename))  # Buyer gets original photo

        # Generate QR Code for the purchased cow
        qr_data = f"https://kamadhenu-zrxm.onrender.com/cow/{new_cow_id}"
        qr_img = qrcode.make(qr_data)
        qr_path = os.path.join(QR_FOLDER, f"{new_cow_id}.png")
        qr_img.save(qr_path)

        # Also copy any existing milk yield records to the new cow
        cursor.execute("""
            INSERT INTO milk_yield (cow_id, date, morning, afternoon, evening, total)
            SELECT %s, date, morning, afternoon, evening, total
            FROM milk_yield 
            WHERE cow_id = %s
        """, (new_cow_id, original_cow["cow_id"]))

        # Add to sold_cows table with the COPIED photo filename
        cursor.execute("""INSERT INTO sold_cows 
                          (cow_id, farmer_id, breed, age, weight, price, photo)
                          VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                       (cow_for_sale["cow_id"], cow_for_sale["farmer_id"], 
                        cow_for_sale["breed"], cow_for_sale["age"], 
                        cow_for_sale["weight"], cow_for_sale["price"], 
                        sold_photo_filename))  # Use copied photo for sold_cows

        conn.commit()
        flash(f"You have successfully purchased the {cow_for_sale['breed']} cow! QR code generated.", "success")
        
    except Exception as e:
        conn.rollback()
        flash("Error processing purchase. Please try again.", "danger")
        print(f"Purchase error: {e}")
        
    finally:
        conn.close()

    return redirect(url_for("list_cows"))



#translator = Translator()



def get_gemini_response(prompt):
    full_prompt = (
        f"You are an expert agriculture assistant. "
        f"Answer ONLY questions related to farming, crops, soil, fertilizers, irrigation, "
        f"plant diseases, animal husbandry, and ANY type of animal health (including farm animals and pets). "
        f"animal health, and rural agriculture practices. "
        f"If the question is not related to agriculture, reply politely in Kannada: "
        f"'ಕ್ಷಮಿಸಿ, ನಾನು ಕೃಷಿ ಮತ್ತು ಪಶುಪಾಲನೆಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆಗಳಿಗೆ ಮಾತ್ರ ಉತ್ತರ ನೀಡುತ್ತೇನೆ.'\n\n"
        f"When the user asks about a disease or treatment, "
        f"always mention the common medicine, pesticide, or veterinary treatment generally used. "    
        f"User asked: {prompt}\n\n"
        f"Answer in **Kannada only**, clear, natural, and grammatically correct. "
        f"Keep the answer short (max 3 sentences)."
    )
    try:
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception:
        return "ಕ್ಷಮಿಸಿ, ನಾನು ಪ್ರತಿಕ್ರಿಯೆ ನೀಡಲಾಗುವುದಿಲ್ಲ."

def get_gemini_response_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_gemini_response(prompt)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                wait_time = (attempt + 1) * 10
                time.sleep(wait_time)
            else:
                raise e
    return "ಕ್ಷಮಿಸಿ, ನಾನು ಪ್ರಸ್ತುತ ಪ್ರತಿಕ್ರಿಯೆ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_text = request.json.get("text")
    print(f"User input: {user_text}")
    lang = request.json.get("lang", "en")  # Default to English
    
    greetings_kn = ["ಹಾಯ್", "ನಮಸ್ತೆ", "ನಮಸ್ಕಾರ"]
    
    if "bye" in user_text.lower() or "ವಿದಾಯ" in user_text:
        bot_response_kn = "ವಿದಾಯ! 👋"
    elif user_text in greetings_kn:
        bot_response_kn = "ನಮಸ್ತೆ! ನಿಮಗೆ ಸಹಾಯ ಬೇಕೇ"
    else:
        # If input is in Kannada, translate to English for Gemini
        if lang == "kn":
            user_text_en = translate_to_english(user_text)
        else:
            user_text_en = user_text
            
        # Get response from Gemini
        bot_response_kn = get_gemini_response_with_retry(user_text_en)
        print(f"Gemini response (Kannada): {bot_response_kn}")

    return {"reply": bot_response_kn}

def translate_to_english(text, retries=3):
    for attempt in range(retries):
        try:
            return GoogleTranslator(source="kn", target="en").translate(text)
        except Exception as e:
            time.sleep(1)
    return text






"""
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_text = request.json.get("text")
    lang = request.json.get("lang")  # "en", "hi", "kn"

    # Force Gemini to reply in English (to keep responses stable)
    prompt = f""""""
    #You are a farm assistant chatbot. The farmer said: {user_text}.
    #Answer clearly in English (short, simple sentences).

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    reply = response.text

    # 🔹 Translate Gemini's English reply to farmer's language
    if lang != "en":
        translated = translator.translate(reply, dest=lang)
        reply = translated.text

    return {"reply": reply}
"""






import os
from werkzeug.utils import secure_filename

# Folder to store vet photos
UPLOAD_FOLDER = "static/uploads/vets"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/vet/register", methods=["GET", "POST"])
def vet_register():
    if request.method == "POST":
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        name = f"{first_name} {last_name}"

        email = request.form["email"]
        phone = request.form["phone"]
        clinic = request.form.get("clinic", "")
        education = request.form["education"]
        experience = float(request.form["experience"])
        specialization = request.form["specialization"]
        password = request.form["password"]

        # Handle photo upload
        photo_file = request.files.get("photo")
        photo_path = None
        if photo_file and photo_file.filename != "":
            filename = secure_filename(photo_file.filename)
            save_path = os.path.join(app.config["VET_UPLOAD_FOLDER"], filename)
            photo_file.save(save_path)
            photo_path = filename   # ✅ only save filename in DB

        conn = get_db()
        cursor = conn.cursor()
        try:
            # Insert vet data including new fields
            cursor.execute("""INSERT INTO veterinarians 
                              (name, email, phone, clinic, education, experience, specialization, password, photo)
                              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                           (name, email, phone, clinic, education, experience, specialization, password, photo_path))
            conn.commit()

            flash("Veterinarian registered successfully! Please login.", "success")
            return redirect(url_for("vet_login"))
        except sqlite3.IntegrityError:
            flash("Email or Phone already registered!", "danger")
        finally:
            conn.close()

    return render_template("vet_register.html")



@app.route("/vet/login", methods=["GET", "POST"])
def vet_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM veterinarians WHERE email=%s AND password=%s", (email, password))
        vet = cursor.fetchone()
        conn.close()

        if vet:
            session["vet_id"] = vet["vet_id"]
            session["vet_name"] = vet["name"]
            flash("Login successful!", "success")
            return redirect(url_for("vet_dashboard"))
        else:
            flash("Invalid credentials!", "danger")

    return render_template("vet_login.html")

from datetime import date as dt_date  # Already imported

@app.route("/vet/dashboard")
def vet_dashboard():
    lang = request.args.get('lang', 'en')
    if "vet_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    vet_id = session["vet_id"]
    today = dt_date.today().isoformat()

    conn = get_db()
    cursor = conn.cursor()

    # Fetch vet info
    cursor.execute("SELECT * FROM veterinarians WHERE vet_id=%s", (vet_id,))
    vet = cursor.fetchone()

    # Count today's appointments for this vet
    cursor.execute(
        "SELECT COUNT(*) as total_appointments FROM appointments WHERE vet_id=%s AND date=%s",
        (vet_id, today)
    )
    total_appointments = cursor.fetchone()["total_appointments"]

    # Count total animals under care (cows assigned to this vet via appointments)
    cursor.execute("""
        SELECT COUNT(DISTINCT c.cow_id) as animals_under_care
        FROM cows c
        JOIN appointments a ON c.farmer_id = a.farmer_id
        WHERE a.vet_id = %s
    """, (vet_id,))
    animals_under_care = cursor.fetchone()["animals_under_care"]

    # Count treatments this week
    week_start = (dt_date.today() - timedelta(days=7)).isoformat()
    cursor.execute("""
        SELECT COUNT(*) as weekly_treatments 
        FROM appointments 
        WHERE vet_id=%s AND date >= %s
    """, (vet_id, week_start))
    weekly_treatments = cursor.fetchone()["weekly_treatments"]

    # Count total health records (cows with health records)
    cursor.execute("""
        SELECT COUNT(*) as health_records
        FROM cows c
        JOIN appointments a ON c.farmer_id = a.farmer_id
        WHERE a.vet_id = %s AND c.health_records IS NOT NULL AND c.health_records != ''
    """, (vet_id,))
    health_records = cursor.fetchone()["health_records"]

    # Get recent appointments for the table
    cursor.execute("""
        SELECT a.id, f.name as farmer_name, c.breed as animal_breed, 
               a.date, a.time, c.health_records
        FROM appointments a
        JOIN farmers f ON a.farmer_id = f.farmer_id
        LEFT JOIN cows c ON f.farmer_id = c.farmer_id
        WHERE a.vet_id = %s
        ORDER BY a.date DESC, a.time DESC
        LIMIT 5
    """, (vet_id,))
    recent_appointments = cursor.fetchall()

    conn.close()

    return render_template(
        "vet_dashboard.html", 
        vet=vet, 
        total_appointments=total_appointments,
        animals_under_care=animals_under_care,
        weekly_treatments=weekly_treatments,
        health_records=health_records,
        recent_appointments=recent_appointments
    )




@app.route("/vet/edit", methods=["GET", "POST"])
def vet_edit_profile():
    if "vet_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    conn = get_db()
    cursor = conn.cursor()

    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        clinic = request.form.get("clinic", "")
        education = request.form["education"]
        experience = float(request.form["experience"])
        specialization = request.form["specialization"]

        # handle new photo upload
        photo_file = request.files.get("photo")
        if photo_file and photo_file.filename != "":
            filename = secure_filename(photo_file.filename)
            save_path = os.path.join(app.config["VET_UPLOAD_FOLDER"], filename)
            photo_file.save(save_path)
            photo_path = filename   # ✅ only save filename in DB

            cursor.execute("""UPDATE veterinarians 
                            SET name=%s, phone=%s, clinic=%s, education=%s, experience=%s, specialization=%s, photo=%s 
                            WHERE vet_id=%s""",
                        (name, phone, clinic, education, experience, specialization, photo_path, session["vet_id"]))
        else:
            cursor.execute("""UPDATE veterinarians 
                            SET name=%s, phone=%s, clinic=%s, education=%s, experience=%s, specialization=%s
                            WHERE vet_id=%s""",
                           (name, phone, clinic, education, experience, specialization, session["vet_id"]))

        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("vet_profile"))

    cursor.execute("SELECT * FROM veterinarians WHERE vet_id=%s", (session["vet_id"],))
    vet = cursor.fetchone()
    conn.close()

    return render_template("vet_edit.html", vet=vet)


@app.route("/vet/profile")
def vet_profile():
    if "vet_id" not in session:
        return redirect(url_for("vet_login"))

    conn = get_db()
    cursor = conn.cursor()  # Create a cursor first
    cursor.execute("SELECT * FROM veterinarians WHERE vet_id = %s", (session["vet_id"],))
    vet = cursor.fetchone()
    conn.close()

    return render_template("vet_profile.html", vet=vet)






@app.route('/book_appointment')
def book_appointment():
    conn = get_db()
    cursor = conn.cursor()  # Will automatically use RealDictCursor from your get_db()
    cursor.execute("SELECT vet_id, name, email, phone, clinic, photo FROM veterinarians")
    vets = cursor.fetchall()
    conn.close()
    return render_template("book_appointment.html", vets=vets)

@app.route("/delete_appointment/<int:appointment_id>", methods=["POST"])
def delete_appointment(appointment_id):
    if "vet_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    conn = get_db()
    cursor = conn.cursor()

    # Ensure the appointment belongs to this vet
    cursor.execute("SELECT * FROM appointments WHERE id=%s AND vet_id=%s", (appointment_id, session["vet_id"]))
    appointment = cursor.fetchone()

    if not appointment:
        conn.close()
        flash("Appointment not found or unauthorized.", "danger")
        return redirect(url_for("vet_appointments"))

    cursor.execute("DELETE FROM appointments WHERE id=%s AND vet_id=%s", (appointment_id, session["vet_id"]))
    conn.commit()
    conn.close()

    flash("Appointment deleted successfully!", "success")
    return redirect(url_for("vet_appointments"))




@app.route("/farmer/treatments")
def farmer_treatments():
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    farmer_id = session["farmer_id"]
    conn = get_db()
    cursor = conn.cursor()
    
    # Get all treatments for this farmer's animals
    cursor.execute("""
        SELECT t.*, v.name as vet_name, v.clinic as vet_clinic, 
               c.breed as cow_breed, c.cow_id,
               a.date as appointment_date
        FROM treatments t
        JOIN veterinarians v ON t.vet_id = v.vet_id
        JOIN cows c ON t.cow_id = c.cow_id
        JOIN appointments a ON t.appointment_id = a.id
        WHERE t.farmer_id = %s
        ORDER BY t.created_at DESC
    """, (farmer_id,))
    
    treatments = cursor.fetchall()
    conn.close()
    
    return render_template("farmer_treatments.html", treatments=treatments)

@app.route("/farmer/treatment_details/<int:treatment_id>")
def farmer_treatment_details(treatment_id):
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    farmer_id = session["farmer_id"]
    conn = get_db()
    cursor = conn.cursor()
    
    # Get specific treatment details with verification that it belongs to this farmer
    cursor.execute("""
        SELECT t.*, v.name as vet_name, v.clinic as vet_clinic, 
               v.phone as vet_phone, v.education as vet_education,
               c.breed as cow_breed, c.age as cow_age, c.color as cow_color,
               c.health_records as cow_health_history,
               a.date as appointment_date, a.time as appointment_time
        FROM treatments t
        JOIN veterinarians v ON t.vet_id = v.vet_id
        JOIN cows c ON t.cow_id = c.cow_id
        JOIN appointments a ON t.appointment_id = a.id
        WHERE t.id = %s AND t.farmer_id = %s
    """, (treatment_id, farmer_id))
    
    treatment = cursor.fetchone()
    conn.close()
    
    if not treatment:
        flash("Treatment record not found!", "danger")
        return redirect(url_for("farmer_treatments"))
    
    return render_template("farmer_treatment_details.html", treatment=treatment)



from datetime import date as dt_date  # rename import to avoid conflict

from datetime import date as dt_date
@app.route('/confirm_appointment/<int:vet_id>', methods=['GET', 'POST'])
def confirm_appointment(vet_id):
    # Ensure farmer is logged in
    farmer_id = session.get("farmer_id")
    if not farmer_id:
        flash("Please login first!", "danger")
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Fetch vet details
        cursor.execute("SELECT vet_id, name, email, phone, clinic, specialization FROM veterinarians WHERE vet_id=%s", (vet_id,))
        vet = cursor.fetchone()
        
        # Fetch farmer details
        cursor.execute("SELECT name, phone FROM farmers WHERE farmer_id=%s", (farmer_id,))
        farmer = cursor.fetchone()
        
        # If vet not found, redirect back
        if not vet:
            flash("Veterinarian not found!", "danger")
            return redirect(url_for('book_appointment'))
        
        if request.method == 'POST':
            appointment_date = request.form['date']
            appointment_time = request.form['time']
            reason = request.form.get('reason', '')  # Get the reason from form
            
            # Insert into appointments table WITH reason column
            cursor.execute(
                "INSERT INTO appointments (farmer_id, vet_id, date, time, reason, status) VALUES (%s, %s, %s, %s, %s, %s)",
                (farmer_id, vet_id, appointment_date, appointment_time, reason, 'scheduled')
            )
            conn.commit()
            
            # Send SMS to vet (optional)
            if vet and vet['phone']:
                sms_message = f"📅 New Appointment:\nFarmer: {farmer['name']}\nDate: {appointment_date}\nTime: {appointment_time}\nReason: {reason[:50] if reason else 'Not specified'}"
                send_sms(vet['phone'], sms_message)
            
            flash('✅ Appointment booked successfully! Vet has been notified.', 'success')
            return redirect(url_for('book_appointment'))
        
    except Exception as e:
        conn.rollback()
        flash(f"Error booking appointment: {str(e)}", "danger")
        print(f"Appointment error: {e}")
    finally:
        conn.close()
    
    today = dt_date.today().isoformat()
    return render_template("confirm_appointment.html", vet=vet, vet_id=vet_id, today=today)


from flask import Flask, request, redirect, url_for, flash
 # import your existing send_sms function

@app.route("/vet/confirm_appointment/<int:appointment_id>", methods=["POST"])
def confirm_appointment_vet(appointment_id):
    conn = get_db()
    cursor = conn.cursor()

    # Fetch appointment and farmer info
    cursor.execute("""
        SELECT a.farmer_id, f.name, f.phone, a.date, a.time
        FROM appointments a
        JOIN farmers f ON a.farmer_id = f.farmer_id
        WHERE a.id = %s
    """, (appointment_id,))
    appointment = cursor.fetchone()

    if not appointment:
        flash("Appointment not found!", "danger")
        conn.close()
        return redirect(url_for('vet_appointments'))

    # Update appointment status to confirmed
    cursor.execute("UPDATE appointments SET status='confirmed' WHERE id=%s", (appointment_id,))
    conn.commit()

    # Send SMS to farmer in Kannada
    farmer_name = appointment['name']
    farmer_phone = appointment['phone']
    date = appointment['date']
    time = appointment['time']

    message = f"ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಪಶು ವೈದ್ಯರ ನೇಮಕಾತಿ {date} ರಂದು {time} ಗಂಟೆಗೆ ದೃಢೀಕರಿಸಲಾಗಿದೆ. - ಕಾಮಧೇನು"
    send_sms(farmer_phone, message)


    flash(f"Appointment confirmed and SMS sent to {farmer_name}!", "success")
    conn.close()
    return redirect(url_for('vet_appointments'))



@app.route('/vet_appointments')
def vet_appointments():
    vet_id = session.get('vet_id')
    if not vet_id:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    conn = get_db()
    cursor = conn.cursor()  # Will automatically use RealDictCursor
    
    # Fetch appointments for this vet with farmer details and status
    cursor.execute("""
        SELECT a.id, f.name as farmer_name, f.phone, a.date, a.time, a.status,
               f.farmer_id, a.vet_id
        FROM appointments a
        JOIN farmers f ON a.farmer_id = f.farmer_id
        WHERE a.vet_id = %s
        ORDER BY a.date, a.time
    """, (vet_id,))

    appointments = cursor.fetchall()
    
    # Get vet name for display
    cursor.execute("SELECT name FROM veterinarians WHERE vet_id=%s", (vet_id,))
    vet = cursor.fetchone()
    
    conn.close()
    
    return render_template('vet_appointments.html', 
                         appointments=appointments, 
                         vet_name=vet['name'] if vet else 'Veterinarian')
# ---------------- Add Geo Tag Page ----------------
@app.route("/add_geotag", methods=["GET"])
def add_geotag():
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()
    # Fetch all cows for this farmer
    cursor.execute("SELECT cow_id, breed FROM cows WHERE farmer_id=%s", (session["farmer_id"],))
    cows = cursor.fetchall()
    conn.close()

    return render_template("add_geotag.html", cows=cows)



@app.route('/geo-tag/<cow_id>')
def geo_tag(cow_id):
    conn = get_db()
    cursor = conn.cursor()  # Create a cursor first
    cursor.execute("SELECT * FROM cows WHERE cow_id=%s AND farmer_id=%s", 
                   (cow_id, session["farmer_id"]))
    cow = cursor.fetchone()
    conn.close()

    if cow is None:
        return "Cow not found or not authorized", 403

    return render_template("geo_tag_map.html", cow=cow)

@app.route('/save-fence', methods=['POST'])
def save_fence():
    try:
        print("=" * 50)
        print("save_fence route called!")
        
        if "farmer_id" not in session:
            print("Unauthorized - no farmer in session")
            return jsonify({"status": "error", "message": "Not logged in"}), 403

        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400
            
        cow_id = data.get("cow_id")
        coordinates = data.get("coordinates", [])
        
        print(f"cow_id: {cow_id}")
        print(f"Number of coordinates: {len(coordinates)}")

        if not cow_id:
            return jsonify({"status": "error", "message": "No cow_id provided"}), 400

        if not coordinates or len(coordinates) < 3:
            return jsonify({"status": "error", "message": "Need at least 3 coordinates to form a fence"}), 400

        # Convert coordinates to JSON string for storage
        import json
        coords_json = json.dumps(coordinates)

        conn = get_db()
        cursor = conn.cursor()

        # Check if cow belongs to this farmer
        cursor.execute("SELECT cow_id FROM cows WHERE cow_id=%s AND farmer_id=%s", 
                      (cow_id, session["farmer_id"]))
        cow = cursor.fetchone()

        if not cow:
            conn.close()
            return jsonify({"status": "error", "message": "Cow not found or unauthorized"}), 403

        # Delete existing geofence for this cow
        cursor.execute("DELETE FROM geofence WHERE cow_id=%s", (cow_id,))
        print("Deleted existing geofence")

        # Get the next id value
        cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM geofence")
        next_id = cursor.fetchone()["next_id"]
        print(f"Next ID will be: {next_id}")

        # Insert new geofence with explicit id
        cursor.execute("""
            INSERT INTO geofence (id, cow_id, farmer_id, coordinates) 
            VALUES (%s, %s, %s, %s)
        """, (next_id, cow_id, session["farmer_id"], coords_json))

        conn.commit()
        conn.close()
        
        print(f"Geofence saved successfully for cow {cow_id}")
        print("=" * 50)

        return jsonify({"status": "success", "message": f"Geofence saved for Cow {cow_id}"})
        
    except Exception as e:
        print(f"ERROR in save_fence: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 

@app.route("/get_cow_details/<cow_id>")
def get_cow_details(cow_id):
    """Get cow details for editing"""
    if "farmer_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get cow details with all fields
    cursor.execute("""
        SELECT cow_id, cattle_type, breed, date_of_birth, age, weight, color, 
               milk_yield, health_records, vaccination_history, special_notes, 
               photo, father_id, mother_id, insurance_by, insurance_policy_number, 
               insurance_valid_upto
        FROM cows 
        WHERE cow_id=%s AND farmer_id=%s
    """, (cow_id, session["farmer_id"]))
    
    cow = cursor.fetchone()
    conn.close()
    
    if cow:
        # Convert to dict - no need for date conversion since SQLite returns strings
        cow_dict = dict(cow)
        return jsonify(cow_dict)
    else:
        return jsonify({"error": "Cow not found"}), 404
    

@app.route("/update_cow", methods=["POST"])
def update_cow():
    """Update cow details in database"""
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))
    
    cow_id = request.form["cow_id"]
    farmer_id = session["farmer_id"]
    
    # Get form data with proper handling of empty values
    cattle_type = request.form.get("cattle_type")
    breed = request.form.get("breed")
    date_of_birth = request.form.get("date_of_birth") or None
    # Don't get age from form - we'll calculate it automatically
    weight = request.form.get("weight")
    color = request.form.get("color")
    
    # Conditionally get milk_yield based on cattle_type
    milk_yield = None
    if cattle_type in ['cow', 'buffalo']:
        milk_yield = request.form.get("milk_yield") or None
    
    health_records = request.form.get("health_records") or None
    vaccination_history = request.form.get("vaccination_history") or None
    special_notes = request.form.get("special_notes") or None
    
    # Get insurance details
    insurance_by = request.form.get("insurance_by") or None
    insurance_policy_number = request.form.get("insurance_policy_number") or None
    insurance_valid_upto = request.form.get("insurance_valid_upto") or None
    
    # Get parent IDs only for calves
    father_id = None
    mother_id = None
    if cattle_type == 'calf':
        father_id = request.form.get("father_id") or None
        mother_id = request.form.get("mother_id") or None

    # Calculate age automatically from date of birth
    calculated_age = None
    if date_of_birth:
        try:
            from datetime import datetime
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = dt_date.today()
            
            # Calculate age in months
            age_in_months = (today.year - dob.year) * 12 + (today.month - dob.month)
            
            # Adjust if the day of month hasn't occurred yet
            if today.day < dob.day:
                age_in_months -= 1
                
            calculated_age = max(0, age_in_months)  # Ensure age is not negative
            print(f"Calculated age: {calculated_age} months from DOB: {date_of_birth}")
            
        except Exception as e:
            print(f"Error calculating age: {e}")
            # If age calculation fails, keep the existing age
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT age FROM cows WHERE cow_id=%s", (cow_id,))
            existing_cow = cursor.fetchone()
            calculated_age = existing_cow["age"] if existing_cow else None
            conn.close()

    # Handle photo upload
    photo_filename = None
    if "photo" in request.files:
        photo_file = request.files["photo"]
        if photo_file and photo_file.filename != "":
            # Generate secure filename
            file_extension = os.path.splitext(photo_file.filename)[1]
            photo_filename = f"{cow_id}_{int(time.time())}{file_extension}"
            filepath = os.path.join(app.config["COW_UPLOAD_FOLDER"], photo_filename)
            photo_file.save(filepath)
            
            # Delete old photo if exists
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT photo FROM cows WHERE cow_id=%s", (cow_id,))
            old_photo = cursor.fetchone()
            if old_photo and old_photo["photo"]:
                old_photo_path = os.path.join(app.config["COW_UPLOAD_FOLDER"], old_photo["photo"])
                if os.path.exists(old_photo_path):
                    os.remove(old_photo_path)

    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # First verify the cow belongs to the farmer
        cursor.execute("SELECT * FROM cows WHERE cow_id=%s AND farmer_id=%s", (cow_id, farmer_id))
        cow = cursor.fetchone()
        
        if not cow:
            flash("Cow not found or you don't have permission to edit!", "danger")
            return redirect(url_for("list_cows"))
        
        # Build update query with proper parameter handling
        update_fields = []
        update_values = []
        
        # Add all fields that should be updated
        update_fields.append("cattle_type = %s")
        update_values.append(cattle_type)
        
        update_fields.append("breed = %s")
        update_values.append(breed)
        
        update_fields.append("date_of_birth = %s")
        update_values.append(date_of_birth)
        
        # Use calculated age instead of form age
        update_fields.append("age = %s")
        update_values.append(calculated_age)
        
        update_fields.append("weight = %s")
        update_values.append(float(weight) if weight else None)
        
        update_fields.append("color = %s")
        update_values.append(color)
        
        update_fields.append("milk_yield = %s")
        update_values.append(float(milk_yield) if milk_yield else None)
        
        update_fields.append("health_records = %s")
        update_values.append(health_records)
        
        update_fields.append("vaccination_history = %s")
        update_values.append(vaccination_history)
        
        update_fields.append("special_notes = %s")
        update_values.append(special_notes)
        
        update_fields.append("insurance_by = %s")
        update_values.append(insurance_by)
        
        update_fields.append("insurance_policy_number = %s")
        update_values.append(insurance_policy_number)
        
        update_fields.append("insurance_valid_upto = %s")
        update_values.append(insurance_valid_upto)
        
        if cattle_type == 'calf':
            update_fields.append("father_id = %s")
            update_values.append(father_id)
            
            update_fields.append("mother_id = %s")
            update_values.append(mother_id)
        else:
            # Clear parent IDs if not a calf
            update_fields.append("father_id = NULL")
            update_fields.append("mother_id = NULL")
        
        if photo_filename:
            update_fields.append("photo = %s")
            update_values.append(photo_filename)
        
        # Add cow_id and farmer_id for WHERE clause
        update_values.extend([cow_id, farmer_id])
        
        # Execute update
        query = f"UPDATE cows SET {', '.join(update_fields)} WHERE cow_id = %s AND farmer_id = %s"
        cursor.execute(query, update_values)
        
        conn.commit()
        flash("Cattle details updated successfully! Age calculated automatically from date of birth.", "success")
        
    except ValueError as e:
        conn.rollback()
        flash(f"Invalid data format: {str(e)}", "danger")
        print(f"Value error: {e}")
    except Exception as e:
        conn.rollback()
        flash(f"Error updating cattle: {str(e)}", "danger")
        print(f"Update error: {e}")
        
    finally:
        conn.close()
    
    return redirect(url_for("list_cows"))



@app.route("/milk_yield", methods=["GET", "POST"])
def milk_yield():
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    cur = conn.cursor()
    today = dt_date.today().isoformat()

    # --- Handle Save Milk Entry ---
    if request.method == "POST" and "cow_id" in request.form:
        cow_id = request.form["cow_id"]
        morning = float(request.form.get("morning", 0))
        evening = float(request.form.get("evening", 0))
        total = morning + evening

        cur.execute("SELECT * FROM milk_yield WHERE cow_id=%s AND date=%s", (cow_id, today))
        existing = cur.fetchone()
        if existing:
            cur.execute("""UPDATE milk_yield 
                           SET morning=%s, evening=%s, total=%s 
                           WHERE cow_id=%s AND date=%s""",
                        (morning, evening, total, cow_id, today))
        else:
            cur.execute("""INSERT INTO milk_yield (cow_id, date, morning, evening, total)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (cow_id, today, morning, evening, total))
        conn.commit()
        flash("Milk yield saved successfully!", "success")

    # --- Get ONLY milk-yielding cattle (cows and buffaloes) ---
    # FIX: Separate execute and fetchall
    cur.execute("""
        SELECT * FROM cows 
        WHERE farmer_id=%s AND cattle_type IN ('cow', 'buffalo')
    """, (session["farmer_id"],))
    cows = cur.fetchall()

    # --- Calculate totals for milk-yielding cattle only ---
    # FIX: Separate execute and fetchone with proper null handling
    cur.execute("""
        SELECT COALESCE(SUM(total), 0) AS total
        FROM milk_yield m
        JOIN cows c ON m.cow_id = c.cow_id
        WHERE m.date=%s AND c.farmer_id=%s AND c.cattle_type IN ('cow', 'buffalo')
    """, (today, session["farmer_id"]))
    daily_total_result = cur.fetchone()
    daily_total = daily_total_result["total"] if daily_total_result else 0

    month_start = dt_date.today().replace(day=1).isoformat()
    cur.execute("""
        SELECT COALESCE(SUM(total), 0) AS total
        FROM milk_yield m
        JOIN cows c ON m.cow_id = c.cow_id
        WHERE m.date >= %s AND c.farmer_id=%s AND c.cattle_type IN ('cow', 'buffalo')
    """, (month_start, session["farmer_id"]))
    monthly_total_result = cur.fetchone()
    monthly_total = monthly_total_result["total"] if monthly_total_result else 0

    selected_date = request.args.get("date")
    if selected_date:
        cur.execute("""
            SELECT m.*, c.breed AS cow_name 
            FROM milk_yield m 
            JOIN cows c ON m.cow_id = c.cow_id 
            WHERE c.farmer_id=%s AND m.date=%s AND c.cattle_type IN ('cow', 'buffalo')
            ORDER BY m.cow_id
        """, (session["farmer_id"], selected_date))
        history = cur.fetchall()
    else:
        cur.execute("""
            SELECT m.*, c.breed AS cow_name 
            FROM milk_yield m 
            JOIN cows c ON m.cow_id = c.cow_id 
            WHERE c.farmer_id=%s AND c.cattle_type IN ('cow', 'buffalo')
            ORDER BY m.date DESC, m.cow_id
        """, (session["farmer_id"],))
        history = cur.fetchall()

    conn.close()
    current_date = dt_date.today().isoformat()

    return render_template("milk_yield.html", 
                           cows=cows, 
                           history=history,
                           daily_total=daily_total,
                           monthly_total=monthly_total,
                           selected_date=selected_date,
                           current_date=current_date)

@app.route("/admin/broadcast", methods=["GET", "POST"])
def admin_broadcast():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()

    # Get all farmers for SMS sending
    cursor.execute("SELECT name, phone FROM farmers")
    farmers = cursor.fetchall()
    
    farmer_count = len(farmers)

    if request.method == "POST":
        title = request.form.get("title")
        message = request.form.get("message")
        
        if not title or not message:
            flash("Please fill in both title and message fields!", "danger")
            return redirect(url_for("admin_broadcast"))
        
        # Create the full SMS message
        full_message = f"🚨 {title}\n\n{message}\n\n- Kamadhenu Alerts"
        
        # Track SMS sending results
        successful_sends = 0
        failed_sends = 0
        failed_numbers = []
        
        # Send SMS to all farmers
        for farmer in farmers:
            sms_result = send_sms(farmer['phone'], full_message)
            if sms_result['success']:
                successful_sends += 1
                print(f"✅ SMS sent to {farmer['name']} ({farmer['phone']})")
            else:
                failed_sends += 1
                failed_numbers.append(f"{farmer['name']} ({farmer['phone']})")
                print(f"❌ Failed to send SMS to {farmer['name']}: {sms_result['error']}")
        
        # Show results to admin
        if successful_sends > 0:
            flash(f"✅ Alert sent successfully to {successful_sends} farmers!", "success")
        
        if failed_sends > 0:
            flash(f"❌ Failed to send to {failed_sends} farmers. Check console for details.", "warning")
        
        conn.close()
        return redirect(url_for("admin_broadcast"))

    conn.close()
    return render_template("admin_broadcast.html", farmer_count=farmer_count)


# ---------------- Farmer Analytics ----------------
@app.route("/analytics")
def analytics():
    if "farmer_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("login"))

    farmer_id = session["farmer_id"]
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Get date filters from request
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Default to last 30 days if no dates provided
        if not start_date:
            start_date = (dt_date.today() - timedelta(days=30)).isoformat()
        if not end_date:
            end_date = dt_date.today().isoformat()

        # Ensure end_date is not before start_date
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        # Daily milk yield data
        cursor.execute("""
            SELECT 
                TO_CHAR(m.date, 'YYYY-MM-DD') as date,
                COALESCE(SUM(m.total), 0) as total_milk
            FROM milk_yield m
            JOIN cows c ON m.cow_id = c.cow_id
            WHERE c.farmer_id = %s 
                AND m.date BETWEEN %s AND %s
            GROUP BY m.date
            ORDER BY m.date
        """, (farmer_id, start_date, end_date))
        daily_milk = cursor.fetchall()

        # Daily cow registration data
        cursor.execute("""
            SELECT 
                TO_CHAR(created_at, 'YYYY-MM-DD') as date,
                COUNT(*) as count
            FROM cows 
            WHERE farmer_id = %s 
                AND created_at::date BETWEEN %s AND %s
            GROUP BY TO_CHAR(created_at, 'YYYY-MM-DD')
            ORDER BY date
        """, (farmer_id, start_date, end_date))
        daily_cows = cursor.fetchall()

        # Daily sales data
        cursor.execute("""
            SELECT 
                TO_CHAR(listed_at, 'YYYY-MM-DD') as date,
                COUNT(*) as sold_count
            FROM cows_for_sale 
            WHERE farmer_id = %s 
                AND is_sold = TRUE 
                AND listed_at::date BETWEEN %s AND %s
            GROUP BY TO_CHAR(listed_at, 'YYYY-MM-DD')
            ORDER BY date
        """, (farmer_id, start_date, end_date))
        daily_sales = cursor.fetchall()

        # Breed distribution
        cursor.execute("""
            SELECT breed, COUNT(*) as count 
            FROM cows 
            WHERE farmer_id = %s 
            GROUP BY breed
            ORDER BY count DESC
        """, (farmer_id,))
        breed_data = cursor.fetchall()

        # Current stats (today's data)
        today = dt_date.today().isoformat()
        cursor.execute("""
            SELECT COALESCE(SUM(m.total), 0) as current_milk
            FROM milk_yield m
            JOIN cows c ON m.cow_id = c.cow_id
            WHERE c.farmer_id = %s AND m.date = %s
        """, (farmer_id, today))
        current_milk_result = cursor.fetchone()
        current_milk = current_milk_result["current_milk"] if current_milk_result else 0

        # Current month sales and revenue
        current_month = dt_date.today().strftime('%Y-%m')
        cursor.execute("""
            SELECT 
                COALESCE(COUNT(*), 0) as current_sales, 
                COALESCE(SUM(price), 0) as current_revenue
            FROM cows_for_sale 
            WHERE farmer_id = %s 
                AND is_sold = TRUE 
                AND TO_CHAR(listed_at, 'YYYY-MM') = %s
        """, (farmer_id, current_month))
        current_sales_data = cursor.fetchone()
        current_sales = current_sales_data["current_sales"] if current_sales_data else 0
        current_revenue = current_sales_data["current_revenue"] if current_sales_data else 0

        # ADD THIS - Get total cows count
        cursor.execute("SELECT COUNT(*) as total FROM cows WHERE farmer_id = %s", (farmer_id,))
        total_cows_result = cursor.fetchone()
        total_cows = total_cows_result["total"] if total_cows_result else 0

    except Exception as e:
        flash(f"Error loading analytics: {str(e)}", "danger")
        # Set default empty values on error
        daily_milk = []
        daily_cows = []
        daily_sales = []
        breed_data = []
        current_milk = 0
        current_sales = 0
        current_revenue = 0
        current_month = dt_date.today().strftime('%Y-%m')
        start_date = (dt_date.today() - timedelta(days=30)).isoformat()
        end_date = dt_date.today().isoformat()
        total_cows = 0  # ADD THIS IN ERROR HANDLING AS WELL
        
    finally:
        conn.close()

    return render_template(
        "analytics.html",
        daily_milk=daily_milk,
        daily_cows=daily_cows,
        daily_sales=daily_sales,
        breed_data=breed_data,
        current_milk=current_milk,
        current_sales=current_sales,
        current_revenue=current_revenue,
        current_month=current_month,
        start_date=start_date,
        end_date=end_date,
        total_cows=total_cows  # ADD THIS LINE
    )
@app.route("/admin/milk_production")
def admin_milk_production():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()

    # Get only milk-yielding cattle (cows and buffaloes) with their milk data
    cursor.execute("""
        SELECT 
            c.farmer_id,
            c.cow_id,
            c.breed as cow_breed,
            COALESCE(c.milk_yield, 0) as avg_milk_yield,
            COALESCE(SUM(my.total), 0) as total_milk_produced
        FROM cows c
        LEFT JOIN milk_yield my ON c.cow_id = my.cow_id
        WHERE c.cattle_type IN ('cow', 'buffalo')  -- ONLY milk-yielding cattle
        GROUP BY c.farmer_id, c.cow_id, c.breed, c.milk_yield
        ORDER BY c.farmer_id, c.cow_id
    """)

    cows_data = cursor.fetchall()
    conn.close()

    return render_template("admin_milk_production.html", cows=cows_data)

@app.route("/admin/farmer_milk_graph/<int:farmer_id>")
def admin_farmer_milk_graph(farmer_id):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    conn = get_db()
    cursor = conn.cursor()
    
    # Get farmer details
    cursor.execute("SELECT name, email FROM farmers WHERE farmer_id=%s", (farmer_id,))
    farmer = cursor.fetchone()
    
    # Get milk production data for the last 30 days
    cursor.execute("""
        SELECT 
            c.cow_id,
            c.breed,
            my.date,
            SUM(my.total) as daily_milk
        FROM cows c
        LEFT JOIN milk_yield my ON c.cow_id = my.cow_id
        WHERE c.farmer_id = %s AND my.date >= date('now', '-30 days')
        GROUP BY c.cow_id, c.breed, my.date
        ORDER BY my.date
    """, (farmer_id,))
    
    milk_data = cursor.fetchall()
    
    # Process data for chart
    dates = []
    cow_data = {}
    
    for row in milk_data:
        date_str = row["date"]
        cow_id = row["cow_id"]
        breed = row["breed"]
        milk = row["daily_milk"] or 0
        
        if date_str not in dates:
            dates.append(date_str)
            
        cow_key = f"{cow_id} - {breed}"
        if cow_key not in cow_data:
            cow_data[cow_key] = []
        
        # Fill missing dates with zeros
        while len(cow_data[cow_key]) < len(dates) - 1:
            cow_data[cow_key].append(0)
            
        cow_data[cow_key].append(milk)
    
    # Fill any remaining gaps with zeros
    for cow_key in cow_data:
        while len(cow_data[cow_key]) < len(dates):
            cow_data[cow_key].append(0)
    
    conn.close()
    
    return render_template("admin_farmer_milk_graph.html", 
                         farmer=farmer, 
                         farmer_id=farmer_id,
                         dates=dates, 
                         cow_data=cow_data)


@app.route("/complete_appointment/<int:appointment_id>")
def complete_appointment(appointment_id):
    if "vet_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    conn = get_db()
    cursor = conn.cursor()
    
    # Get appointment details and ALL farmer's cows
    cursor.execute("""
        SELECT a.*, f.name as farmer_name, f.farmer_id
        FROM appointments a
        JOIN farmers f ON a.farmer_id = f.farmer_id
        WHERE a.id = %s AND a.vet_id = %s
    """, (appointment_id, session["vet_id"]))
    
    appointment = cursor.fetchone()
    
    if not appointment:
        conn.close()
        flash("Appointment not found!", "danger")
        return redirect(url_for("vet_appointments"))
    
    # Get ALL farmer's cows for treatment form
    cursor.execute("""
        SELECT cow_id, breed, age, vaccination_history, health_records
        FROM cows 
        WHERE farmer_id = %s
    """, (appointment["farmer_id"],))
    
    cows = cursor.fetchall()
    conn.close()
    
    # Convert cows to list of dictionaries for JSON serialization
    cows_list = []
    for cow in cows:
        cows_list.append({
            'cow_id': cow['cow_id'],
            'breed': cow['breed'],
            'age': cow['age'],
            'vaccination_history': cow['vaccination_history'],
            'health_records': cow['health_records']
        })
    
    # Convert appointment to dict
    appointment_dict = {
        'id': appointment['id'],
        'farmer_name': appointment['farmer_name'],
        'farmer_id': appointment['farmer_id'],
        'date': appointment['date'],
        'time': appointment['time']
    }
    
    return render_template("complete_appointment.html", 
                         appointment=appointment_dict, 
                         cows=cows_list)



@app.route("/save_treatment", methods=["POST"])
def save_treatment():
    if "vet_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    appointment_id = request.form["appointment_id"]
    
    # Get all treatment data as lists
    cow_ids = request.form.getlist("cow_id[]")
    diagnoses = request.form.getlist("diagnosis[]")
    medicines_list = request.form.getlist("medicines[]")
    vaccination_details_list = request.form.getlist("vaccination_details[]")
    instructions_list = request.form.getlist("instructions[]")
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Get appointment details
        cursor.execute("""
            SELECT farmer_id, vet_id FROM appointments 
            WHERE id = %s AND vet_id = %s
        """, (appointment_id, session["vet_id"]))
        
        appointment = cursor.fetchone()
        if not appointment:
            flash("Appointment not found!", "danger")
            return redirect(url_for("vet_appointments"))
        
        # Process each animal treatment
        treatment_count = 0
        for i, cow_id in enumerate(cow_ids):
            if cow_id:  # Ensure cow_id is not empty
                diagnosis = diagnoses[i] if i < len(diagnoses) else ""
                medicines = medicines_list[i] if i < len(medicines_list) else ""
                vaccination_details = vaccination_details_list[i] if i < len(vaccination_details_list) else ""
                instructions = instructions_list[i] if i < len(instructions_list) else ""
                
                # Insert treatment record for this animal
                cursor.execute("""
                    INSERT INTO treatments 
                    (appointment_id, cow_id, vet_id, farmer_id, diagnosis, medicines, 
                     vaccination_details, instructions, treatment_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, DATE('now'))
                """, (appointment_id, cow_id, session["vet_id"], appointment["farmer_id"],
                      diagnosis, medicines, vaccination_details, instructions))
                
                # Update cow's vaccination history if provided
                if vaccination_details:
                    cursor.execute("""
                        SELECT vaccination_history FROM cows WHERE cow_id = %s
                    """, (cow_id,))
                    
                    cow = cursor.fetchone()
                    current_vaccination = cow["vaccination_history"] or ""
                    
                    # Append new vaccination details with date
                    new_vaccination = f"{current_vaccination}\n{dt_date.today()}: {vaccination_details}".strip()
                    
                    cursor.execute("""
                        UPDATE cows SET vaccination_history = %s WHERE cow_id = %s
                    """, (new_vaccination, cow_id))
                
                treatment_count += 1
        
        # Update appointment status to completed
        cursor.execute("""
            UPDATE appointments SET status = 'completed' WHERE id = %s
        """, (appointment_id,))
        
        conn.commit()
        flash(f"Successfully recorded treatments for {treatment_count} animal(s)! Vaccination histories updated.", "success")
        
    except Exception as e:
        conn.rollback()
        flash(f"Error saving treatments: {str(e)}", "danger")
        print(f"Treatment error: {e}")
        
    finally:
        conn.close()
    
    return redirect(url_for("vet_appointments"))

@app.route("/vet/treatments")
def vet_treatments():
    if "vet_id" not in session:
        flash("Please login first!", "danger")
        return redirect(url_for("vet_login"))

    conn = get_db()
    cursor = conn.cursor()
    
    # Get all treatments by this vet
    cursor.execute("""
        SELECT t.*, f.name as farmer_name, c.breed as cow_breed, 
               a.date as appointment_date
        FROM treatments t
        JOIN farmers f ON t.farmer_id = f.farmer_id
        JOIN cows c ON t.cow_id = c.cow_id
        JOIN appointments a ON t.appointment_id = a.id
        WHERE t.vet_id = %s
        ORDER BY t.created_at DESC
    """, (session["vet_id"],))
    
    treatments = cursor.fetchall()
    conn.close()
    
    return render_template("vet_treatments.html", treatments=treatments)
# ---------------- Logout ----------------
@app.route("/farmer/logout")
def farmer_logout():
    session.clear()
    return redirect(url_for("home"))   # sends farmer back to main.html (your index page)


@app.route("/vet/logout")
def vet_logout():
    session.clear()
    return redirect(url_for("veterinarian"))

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
