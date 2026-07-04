from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from dotenv import load_dotenv
load_dotenv()
import ollama
import json
import requests
import re
import os
import logging
from datetime import timedelta

# ═══════════════════════════════════════════════════════
# KONFIGURASI
# ═══════════════════════════════════════════════════════

# Setup logging untuk debug & monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi Session Server-Side (Opsi C)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_sessions'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Auto cleanup

# Buat direktori session kalau belum ada
os.makedirs('./flask_sessions', exist_ok=True)

# Initialize Flask-Session
Session(app)

# Konfigurasi Model
MODEL_NAME = "qwen2.5:7b"

# API Configuration (pakai environment variable)
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', '')
RAPIDAPI_HOST = "internships-api.p.rapidapi.com"
API_URL = f"https://internships-api.p.rapidapi.com/active-jb-7d"

if not RAPIDAPI_KEY:
    logger.warning("RAPIDAPI_KEY tidak diset! Pakai environment variable atau .env file")


# ═══════════════════════════════════════════════════════
# FUNGSI INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════

def classify_intent(user_input):
    """
    Klasifikasi intensi pengguna ke CARI_MAGANG atau CHAT_BIASA.
    Default ke CARI_MAGANG kalau gagal (safer default).
    """
    prompt = f"""
    Tugas: Klasifikasikan intensi pesan dari pengguna berikut.
    Pesan: "{user_input}"

    Pilihlah satu dari dua kategori berikut:
    1. "CARI_MAGANG": Jika pengguna menceritakan profilnya, mencari lowongan baru, menyebutkan skill baru untuk dicari, atau meminta rekomendasi magang secara eksplisit.
    2. "CHAT_BIASA": Jika pengguna menyapa, berterima kasih, bertanya tentang tips umum, atau menanyakan detail/pertanyaan lanjutan terkait lowongan yang sebelumnya sudah diberikan AI.

    Output HANYA kata "CARI_MAGANG" atau "CHAT_BIASA" tanpa penjelasan apa pun.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        intent = response['message']['content'].strip()
        logger.info(f"Intent classified: '{user_input[:50]}...' → {intent}")
        
        if "CARI_MAGANG" in intent:
            return "CARI_MAGANG"
        return "CHAT_BIASA"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "CARI_MAGANG"  # Default aman


# ═══════════════════════════════════════════════════════
# FUNGSI EKSTRAKSI PARAMETER
# ═══════════════════════════════════════════════════════

def extract_params_with_qwen(user_story):
    """
    Ekstrak parameter pencarian dari profil user.
    Returns dict dengan rekomendasi_posisi, api_query, location, is_remote, reasoning.
    """
    prompt = f"""
    Tugas: Analisis profil user dan berikan rekomendasi posisi magang.
    Profil User: "{user_story}"

    Aturan:
    1. 'rekomendasi_posisi': List 3 posisi yang cocok dengan skill user.
    2. 'api_query': Buat string pencarian menggunakan operator OR.
        PENTING - Pertimbangkan variasi penamaan di industri Indonesia:
        - Title utama yang paling cocok
        - 2-3 sinonim atau title terkait
        - 1-2 variasi konvensi industri (formal/informal)
        - GUNAKAN 4-6 variasi total
        - JANGAN masukkan nama industri (FMCG, Banking, dll) ke api_query, 
          karena ini bukan nama posisi
        
        Contoh untuk "ingin magang sebagai pembuat konten":
        "Content Creator" OR "Social Media" OR "Digital Marketing" OR 
        "Multimedia" OR "Creative"
        
        Contoh untuk "mahasiswa Teknik Lingkungan":
        "Environmental" OR "Sustainability" OR "HSE" OR "EHS" OR 
        "Safety" OR "Environment Officer"
        
        Contoh untuk "ingin magang analisis data":
        "Data Analyst" OR "Business Analyst" OR "Data Science" OR 
        "Analytics" OR "Business Intelligence"

    3. 'location': Lokasi pencarian yang dapat berupa nama kota 
       atau nama negara, dalam format kanonikal.
       PENTING - Gunakan nama resmi yang dikenali API:
       - Untuk kota: 
         * "Yogyakarta" (BUKAN Jogja, DIY)
         * "Jakarta" (BUKAN JKT, DKI)
         * "Surabaya" (BUKAN SBY)
       - Untuk negara: 
         * "Indonesia" (BUKAN ID, INA)
         * "United States" (BUKAN US, USA)
         * "United Kingdom" (BUKAN UK)
       - Default null jika user tidak menyebutkan lokasi spesifik

    
    4. 'is_remote': true jika user menyebut remote/WFH/work from home.

    Output JSON saja:
    {{
        "rekomendasi_posisi": [],
        "api_query": "...",
        "location": "...",
        "is_remote": false,
        "reasoning": "Penjelasan rekomendasi posisi dan kenapa cocok dengan profil pengguna. Jika user menyebut industri spesifik, jelaskan bahwa hasil mencakup berbagai industri karena filter spesifik per industri tidak tersedia."
    }}
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(content)
        
        logger.info(f"Extracted params: query='{result.get('api_query')}', "
                    f"location='{result.get('location')}', remote={result.get('is_remote')}")
        
        return result
    except Exception as e:
        logger.error(f"Parameter extraction failed: {e}")
        return {
            "rekomendasi_posisi": ["General Intern"],
            "api_query": "Internship",
            "location": None,
            "is_remote": False,
            "reasoning": "Gagal menganalisis secara mendalam, mencari posisi umum."
        }


# ═══════════════════════════════════════════════════════
# FUNGSI PEMANGGILAN API
# ═══════════════════════════════════════════════════════

def call_job_api(params):
    """
    Panggil API fantastic.jobs untuk dapatkan data lowongan.
    Returns list of jobs atau empty list kalau gagal.
    """
    # Convert is_remote dengan benar
    is_remote = params.get('is_remote', False)
    if isinstance(is_remote, bool):
        remote_str = "true" if is_remote else "false"
    else:
        remote_str = str(is_remote).lower()
    
    api_params = {
        "title_filter": params.get('api_query'),
        "location_filter": params.get('location'),
        "remote": remote_str,
        "include_ai": "true",
        "limit": 10
    }
    
    # Hapus parameter dengan nilai None (kalau location null)
    api_params = {k: v for k, v in api_params.items() if v is not None}
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    
    try:
        logger.info(f"Calling API with params: {api_params}")
        r = requests.get(API_URL, params=api_params, headers=headers, timeout=20)
        
        if r.status_code == 200:
            data = r.json()
            jobs = data if isinstance(data, list) else data.get('data', [])
            logger.info(f"API returned {len(jobs)} jobs")
            return jobs
        else:
            logger.error(f"API error: status {r.status_code}, response: {r.text[:200]}")
            return []
    except requests.Timeout:
        logger.error("API request timeout")
        return []
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return []


# ═══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def safe_get_location(job):
    """Safe extraction lokasi dari job data (handle array atau string)."""
    locations = job.get("locations_derived")
    if isinstance(locations, list) and len(locations) > 0:
        return locations[0]
    elif isinstance(locations, str):
        return locations
    return "Tidak dicantumkan"


# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

@app.route('/')
def index():
    """Halaman utama, sekaligus clear session untuk sesi baru."""
    session.clear()
    logger.info("Session cleared, new session started")
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint utama untuk handle chat request."""
    user_input = request.json.get('message')
    chat_history = request.json.get('history', [])
    
    if not user_input:
        return jsonify({"reply": "Silakan ceritakan latar belakangmu."})
    
    logger.info(f"User input: '{user_input[:100]}...'")
    
    # Klasifikasi intent
    intent = classify_intent(user_input)
    
    # ─────────────────────────────────────────
    # JALUR 1: CHAT_BIASA
    # ─────────────────────────────────────────
    if intent == "CHAT_BIASA":
        last_jobs = session.get('last_jobs', [])
        # ✨ DEBUG: Log apa yang ada di session
        logger.info(f"🔍 CHAT_BIASA session data:")
        logger.info(f"   Total jobs in session: {len(last_jobs)}")
        for i, job in enumerate(last_jobs):
            logger.info(f"   Job {i}: title={job.get('title')}, "
                        f"org={job.get('organization')}")
        
        messages = []
        # Ambil 6 pesan terakhir untuk konteks
        for msg in chat_history[-6:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # ✨ PROMPT UPGRADED — Anti-Halusinasi + Scope Enforcement
        system_context = f"""Kamu adalah AI Internship Assistant yang ramah dan profesional, 
            khusus membantu mahasiswa Indonesia mencari informasi tentang magang.

            ================================================================
            KONTEKS DATA LOWONGAN TERAKHIR YANG DIBERIKAN:
            {json.dumps(last_jobs, ensure_ascii=False, indent=2)}
            ================================================================

            ATURAN PENTING (WAJIB DIPATUHI):

            1. ATURAN ANTI-HALUSINASI (PALING UTAMA):
            - HANYA gunakan informasi dari "DATA LOWONGAN TERAKHIR" di atas
            - JANGAN PERNAH mengarang nama perusahaan, posisi, gaji, lokasi, 
                atau detail lain yang TIDAK ADA dalam data tersebut
            - JANGAN PERNAH menambah lowongan baru dari pengetahuan umummu
            - Jika user menanyakan lowongan yang TIDAK ADA dalam data, 
                jawab dengan jujur dan tawarkan pencarian baru

            2. ATURAN PENGGUNAAN DATA:
            - Jika user bertanya tentang lowongan spesifik, cari di data 
                dan sebutkan detail SESUAI data yang ada
            - Jika user bertanya perbandingan, bandingkan ANTAR LOWONGAN 
                YANG ADA, berikan alasan berdasarkan informasi tersedia

            3. ATURAN HANDLE DATA KOSONG:
            - Jika data lowongan kosong, jawab dengan menawarkan pencarian baru

            4. ATURAN SCOPE:
            - Topik yang BOLEH: pencarian magang, rekomendasi karir, tips 
                melamar magang, persiapan CV/wawancara
            - Topik yang HARUS dialihkan: pertanyaan di luar konteks magang/karir
            - Format pengalihan: "Maaf, saya khusus membantu pencarian dan 
                persiapan magang. Apakah ada yang bisa saya bantu terkait magang?"

            5. ATURAN FORMAT:
            - Bahasa Indonesia yang ramah dan profesional
            - Format markdown yang rapi
            - Jika mereferensikan lowongan, sebutkan nama posisi dan perusahaan
            - Jangan terlalu panjang—to-the-point

            PRINSIP UTAMA: LEBIH BAIK MENGAKUI TIDAK TAHU DARIPADA MENGARANG."""
        
        messages.insert(0, {"role": "system", "content": system_context})
        messages.append({"role": "user", "content": user_input})
        
        try:
            res = ollama.chat(model=MODEL_NAME, messages=messages)
            return jsonify({"reply": res['message']['content']})
        except Exception as e:
            logger.error(f"CHAT_BIASA response failed: {e}")
            return jsonify({"reply": f"Terjadi kesalahan pada AI: {str(e)}"})
    
    # ─────────────────────────────────────────
    # JALUR 2: CARI_MAGANG
    # ─────────────────────────────────────────
    else:
        analysis = extract_params_with_qwen(user_input)
        jobs = call_job_api(analysis)
        
        if not jobs:
            logger.info("No jobs found for query")
            return jsonify({
                "reply": f"**Analisis Profil:**\n{analysis['reasoning']}\n\n"
                         f"Maaf, tidak ditemukan lowongan yang cocok saat ini. "
                         f"Hal ini dapat terjadi karena keterbatasan ketersediaan "
                         f"data lowongan dalam 7 hari terakhir."
            })
        
        # Simpan ke session (Opsi C: server-side, no size limit issue)
        session['last_jobs'] = jobs[:10]
        logger.info(f"Saved {len(jobs[:10])} jobs to session")
        
        # Payload untuk LLM (compact untuk hemat token)
        jobs_payload = []
        for idx, job in enumerate(jobs[:10]):
            jobs_payload.append({
                "index": idx,
                "title": job.get("title", ""),
                "company": job.get("organization", ""),
                "location": safe_get_location(job),
                "description": job.get("ai_core_responsibilities") or job.get("description", "")
            })
        
        final_prompt = f"""
        PROFIL USER: {user_input}
        DATA LOWONGAN: {json.dumps(jobs_payload)}
        
        TUGAS:
        Pilih maksimal 5 lowongan dari DATA LOWONGAN yang paling cocok dengan PROFIL USER.
        Terjemahkan informasi lokasi dan ringkasan deskripsi lowongan pilihan tersebut ke dalam Bahasa Indonesia yang baik dan profesional.
        
        ATURAN STRUKTUR OUTPUT (WAJIB):
        Anda HARUS merespon dalam format JSON bersih dengan struktur seperti berikut tanpa tambahan teks penjelas apa pun:
        {{
            "rekomendasi": [
                {{
                    "index": 0,
                    "lokasi_indo": "Hasil terjemahan lokasi ke Bahasa Indonesia (Contoh: 'Jakarta, Indonesia' atau 'Remote')",
                    "deskripsi_indo": "Ringkasan pendek deskripsi pekerjaan yang sudah diterjemahkan ke Bahasa Indonesia maksimal 2 kalimat",
                    "alasan_cocok": "Penjelasan singkat menggunakan Bahasa Indonesia kenapa posisi ini cocok dengan profil user"
                }}
            ]
        }}
        Tampilkan maksimal 5 rekomendasi. Jika hanya ada 1 data lowongan, berikan 1 saja di dalam list.
        """
        
        try:
            res = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": final_prompt}]
            )
            ai_response = res['message']['content'].strip()
            
            # Parse JSON dari response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            parsed_recommendations = json.loads(json_match.group()) if json_match else json.loads(ai_response)
            
            # Render output markdown
            output_markdown = f"**Analisis :** {analysis['reasoning']}\n\n---\n\nBerikut adalah rekomendasi untukmu:\n\n"
            
            recommendation_list = parsed_recommendations.get("rekomendasi", [])
            for i, rec in enumerate(recommendation_list):
                idx = int(rec.get("index", 0))
                
                if idx >= len(jobs):
                    continue
                
                original_job = jobs[idx]
                
                posisi = original_job.get("title", "Tidak dicantumkan")
                perusahaan = original_job.get("organization", "Tidak dicantumkan")
                gaji = original_job.get("ai_salary_value", "Tidak dicantumkan")
                reputasi = original_job.get("linkedin_org_followers", "0")
                link_asli = original_job.get("url", "#")
                
                lokasi_id = rec.get("lokasi_indo", safe_get_location(original_job))
                deskripsi_id = rec.get("deskripsi_indo", original_job.get("ai_core_responsibilities", "Tidak dicantumkan"))
                alasan = rec.get("alasan_cocok", "Cocok dengan kualifikasi Anda.")
                
                output_markdown += f"{i+1}. **{posisi}** di **{perusahaan}**\n"
                output_markdown += f"   📍 Lokasi: {lokasi_id}\n"
                output_markdown += f"   💰 Gaji: {gaji}\n"
                output_markdown += f"   📝 Deskripsi: {deskripsi_id}\n"
                output_markdown += f"   ⭐ Reputasi: {reputasi} followers\n"
                output_markdown += f"   🔗 Link: {link_asli}\n"
                output_markdown += f"   💡 *Kenapa cocok?* {alasan}\n\n"
            
            logger.info(f"Generated {len(recommendation_list)} recommendations")
            return jsonify({"reply": output_markdown})
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return jsonify({"reply": f"Terjadi kesalahan format AI, silakan coba lagi."})
        except Exception as e:
            logger.error(f"CARI_MAGANG processing failed: {e}")
            return jsonify({"reply": f"Terjadi kesalahan format AI, silakan coba lagi. (Error: {str(e)})"})


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True)