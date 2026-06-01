from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import secrets
import json
from werkzeug.utils import secure_filename
import psycopg2
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import time
import threading
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime, date
from zoneinfo import ZoneInfo

BUDAPEST_TZ = ZoneInfo("Europe/Budapest")
import statsmodels.api as sm
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import os
import base64
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from scipy.stats import ttest_ind, kruskal, mannwhitneyu, spearmanr, f_oneway, shapiro, normaltest
from ftplib import FTP
from urllib.parse import urlparse
from skimage.color import rgb2hsv


app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
nas_host = os.getenv("NAS_HOST")
nas_user = os.getenv("NAS_USER")
nas_password = os.getenv("NAS_PASS")
nas_folder = os.getenv("NAS_DIR")

def upload_to_nas(file_path, TAJ, measurement_type):
    if measurement_type not in ['mai_initial', 'mai_final', 'A2_gerinc', 'A2_bukkalis', 'A2_lingualis']:
        raise ValueError("measurement_type must be either 'initial_mai', 'final_mai', 'A2_gerinc', 'A2_bukkalis', or 'A2_lingualis'")
    if measurement_type == 'mai_initial' or measurement_type == 'mai_final':
        filename = f"{measurement_type}_{TAJ}.tiff"
    else:
        filename = f"modellanalizis_{TAJ}_{measurement_type}.stl"
    
    with FTP(nas_host) as ftp:
        ftp.login(nas_user, nas_password)
        ftp.cwd(nas_folder)
        with open(file_path, 'rb') as file:
            ftp.storbinary(f'STOR {filename}', file)
        ftp.quit()
    nas_file_path = f'ftp://{nas_host}/{nas_folder}/{filename}'
    return nas_file_path

def download_from_nas(ftp_url, local_path):
    # Parse the FTP URL to extract the host, path, and filename
    parsed_url = urlparse(ftp_url)
    ftp_host = parsed_url.hostname
    ftp_path = parsed_url.path
    ftp_filename = os.path.basename(ftp_path)
    
    # Establish FTP connection and download the file
    with FTP(ftp_host) as ftp:
        ftp.login(nas_user, nas_password)
        ftp.cwd(os.path.dirname(ftp_path))
        with open(local_path, 'wb') as local_file:
            ftp.retrbinary(f"RETR {ftp_filename}", local_file.write)
    
    return local_path

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Retrieve PostgreSQL connection URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL environment variable")

def create_db_connection():
    return psycopg2.connect(DATABASE_URL)

db = create_db_connection()

def ping_db():
    global db
    while True:
        time.sleep(600)
        try:
            if db.closed:
                db = create_db_connection()
            else:
                cur = db.cursor()
                cur.execute("SELECT 1")
                cur.close()
        except psycopg2.Error as err:
            print(f"Error pinging PostgreSQL: {err}")
            try:
                db = create_db_connection()
            except psycopg2.Error:
                pass

thread = threading.Thread(target=ping_db)
thread.daemon = True
thread.start()

def get_db_cursor():
    global db
    try:
        if db.closed:
            db = create_db_connection()
        else:
            db.rollback()
    except psycopg2.Error:
        db = create_db_connection()
    return db.cursor()

def stdev(histogram):
    start=0
    pixelintenzitas = np.arange(start, start + len(histogram))
    mean_intenzitas = np.average(pixelintenzitas, weights=histogram)
    variancia = np.average((pixelintenzitas - mean_intenzitas)**2, weights=histogram)
    return np.sqrt(variancia)

def process_image(image_path):
    # Check if the image_path is an FTP URL
    if image_path.startswith('ftp://'):
        # If the image is on the NAS, download it to a temporary local path
        filename = os.path.basename(urlparse(image_path).path)
        local_temp_path = os.path.join('/tmp', filename)
        download_from_nas(image_path, local_temp_path)
        image_path = local_temp_path  # Now use the local path for processing
    
    image = Image.open(image_path)
    rgb_image = image.convert('RGB')
    r, g, b = rgb_image.split()
    r = r.point(lambda i: i * 0.66)
    g = g.point(lambda i: i * 0.66)
    b = b.point(lambda i: i * 0.66)
    histogram_r = r.histogram()
    histogram_b = b.histogram()

    print(histogram_r)
    print(histogram_b)

    # leave out the first value of the list
    histogram_r = histogram_r[2:]
    histogram_b = histogram_b[2:]
    print(histogram_r)
    print(histogram_b)
    plt.figure(figsize=(10, 5))
    plt.plot(histogram_r, color='red', label="Vörös csatorna")
    plt.plot(histogram_b, color='blue', label="Kék csatorna")
    plt.title('A vörös és kék csatornák hisztogramja')
    plt.xlabel("Pixel intenzitás")
    plt.ylabel('Pixel szám')
    plt.xticks(np.arange(0, 256, 5))
    plt.xticks(rotation=90)
    plt.grid()
    plt.legend()

    std_dev_red = stdev(histogram_r)
    std_dev_blue = stdev(histogram_b)
    
    mai = std_dev_red + std_dev_blue
    return mai


def calculate_hue_circular_sd(image_path, saturation_threshold=0.20, value_min=0.05, value_max=0.98):
    # Check if the image_path is an FTP URL
    if image_path.startswith('ftp://'):
        filename = os.path.basename(urlparse(image_path).path)
        local_temp_path = os.path.join('/tmp', filename)
        download_from_nas(image_path, local_temp_path)
        image_path = local_temp_path

    image = Image.open(image_path).convert("RGB")
    rgb_array = np.asarray(image).astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb_array)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (saturation > saturation_threshold) &
        (value > value_min) &
        (value < value_max)
    )

    hue_angles = hue[mask] * 2 * np.pi

    if len(hue_angles) == 0:
        return np.nan

    mean_cos = np.mean(np.cos(hue_angles))
    mean_sin = np.mean(np.sin(hue_angles))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    R = np.clip(R, 1e-12, 1.0)

    circular_sd_rad = np.sqrt(-2 * np.log(R))
    circular_sd_deg = np.degrees(circular_sd_rad)

    return float(circular_sd_deg)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'tiff', 'tif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/questionnaire1')
def questionnaire1():
    return render_template('questionnaire1.html')

@app.route('/questionnaire2')
def questionnaire2():
    return render_template('questionnaire2.html')

@app.route('/questionnaire3')
def questionnaire3():
    return render_template('questionnaire3.html')

@app.route('/student_exam')
def student_exam():
    return render_template('student_exam.html')

@app.route('/submit_student_exam', methods=['POST'])
def submit_student_exam():
    cursor = get_db_cursor()
    student_name = request.form['student_name']
    TAJ = request.form['TAJ']
    denture_type = request.form['denture_type']

    F5 = request.form.get('F5')
    F7 = request.form.get('F7')
    F8 = request.form.get('F8')
    A1_Kaan = request.form.get('A1_Kaan')
    A3_jobb = request.form.get('A3_jobb')
    A3_bal = request.form.get('A3_bal')
    A4_jobb = request.form.get('A4_jobb')
    A4_bal = request.form.get('A4_bal')
    A5_jobb = request.form.get('A5_jobb')
    A5_bal = request.form.get('A5_bal')
    A6_jobb = request.form.get('A6_jobb')
    A6_bal = request.form.get('A6_bal')
    A7_jobb = request.form.get('A7_jobb')
    A7_bal = request.form.get('A7_bal')
    A8_jobb = request.form.get('A8_jobb')
    A8_bal = request.form.get('A8_bal')
    A9_jobb = request.form.get('A9_jobb')
    A9_bal = request.form.get('A9_bal')
    A11 = request.form.get('A11')
    A12 = request.form.get('A12')
    A13 = request.form.get('A13')
    A14 = request.form.get('A14')

    sql = """
    INSERT INTO student_examinations (TAJ, student_name, denture_type, F5, F7, F8, A1_Kaan, A3_jobb, A3_bal, A4_jobb, A4_bal, A5_jobb, A5_bal, A6_jobb, A6_bal, A7_jobb, A7_bal, A8_jobb, A8_bal, A9_jobb, A9_bal, A11, A12, A13, A14)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (TAJ, student_name, denture_type, F5, F7, F8, A1_Kaan, A3_jobb, A3_bal, A4_jobb, A4_bal, A5_jobb, A5_bal, A6_jobb, A6_bal, A7_jobb, A7_bal, A8_jobb, A8_bal, A9_jobb, A9_bal, A11, A12, A13, A14)
    try:
        cursor.execute(sql, values)
        db.commit()
        return render_template('confirmation.html')
    finally:
        cursor.close()

@app.route('/submit_questionnaire1', methods=['POST'])
def submit_questionnaire1():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    birthdate = request.form['birthdate']
    gender = request.form['gender']
    denture_type = request.form['denture_type']
    responsiveness_today_situation = request.form['responsiveness_today_situation']
    chewing_today_situation = request.form['chewing_today_situation']

    # Fetch responses for GOHAI questions
    GOHAI_questions = [request.form[f'GOHAI_{i}'] for i in range(1, 13)]

    # Fetch responses for OHIP questions
    OHIP_questions = [request.form[f'OHIP_{i}'] for i in range(1, 6)]

    F5 = request.form.get('F5')
    F7 = request.form.get('F7')
    F8 = request.form.get('F8')
    A1_Kaan = request.form.get('A1_Kaan')
    A3_jobb = request.form.get('A3_jobb')
    A3_bal = request.form.get('A3_bal')
    A4_jobb = request.form.get('A4_jobb')
    A4_bal = request.form.get('A4_bal')
    A5_jobb = request.form.get('A5_jobb')
    A5_bal = request.form.get('A5_bal')
    A6_jobb = request.form.get('A6_jobb')
    A6_bal = request.form.get('A6_bal')
    A7_jobb = request.form.get('A7_jobb')
    A7_bal = request.form.get('A7_bal')
    A8_jobb = request.form.get('A8_jobb')
    A8_bal = request.form.get('A8_bal')
    A9_jobb = request.form.get('A9_jobb')
    A9_bal = request.form.get('A9_bal')
    A11 = request.form.get('A11')
    A12 = request.form.get('A12')
    A13 = request.form.get('A13')
    A14 = request.form.get('A14')

    initials = request.form.get('initials')
    # Wall-clock time in Europe/Budapest (naive for TIMESTAMP columns)
    record_datetime = datetime.now(BUDAPEST_TZ).replace(tzinfo=None)
    sql = """
    INSERT INTO patients ("id", "TAJ", "record_datetime", "birthdate", "gender", "denture_type", "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12", "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5", "responsiveness_today_situation", "chewing_today_situation", "F5", "F7", "F8", "A1_Kaan", "A3_jobb", "A3_bal", "A4_jobb", "A4_bal", "A5_jobb", "A5_bal", "A6_jobb", "A6_bal", "A7_jobb", "A7_bal", "A8_jobb", "A8_bal", "A9_jobb", "A9_bal", "A11", "A12", "A13", "A14", "data_uploader")
    VALUES ((SELECT COALESCE(MAX("id"), 0) + 1 FROM patients), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (TAJ, record_datetime, birthdate, gender, denture_type, *GOHAI_questions, *OHIP_questions, responsiveness_today_situation, chewing_today_situation, F5, F7, F8, A1_Kaan, A3_jobb, A3_bal, A4_jobb, A4_bal, A5_jobb, A5_bal, A6_jobb, A6_bal, A7_jobb, A7_bal, A8_jobb, A8_bal, A9_jobb, A9_bal, A11, A12, A13, A14, initials)
    try:
        cursor.execute(sql, values)
        db.commit()
        return render_template('confirmation.html')
    finally:
        cursor.close()

@app.route('/submit_questionnaire2', methods=['POST'])
def submit_questionnaire2():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    jaw_selection = request.form.get('jawSelection', '')
    
    # Function to fetch form data safely
    def get_form_data(field_name):
        value = request.form.get(field_name, None)
        return value if value else None
    
    # F1 = numeric value in mm (with 2 decimal places)
    F1 = get_form_data('F1')  # felső állcsontgerinc profilja (mm)
    
    # Fetch the float values for F2
    F2 = get_form_data('F2') # alámenősség köbmilliméterben
    
    # Fetch responses for F3 to F9 questions
    F3 = get_form_data('F3') # szájpadboltozat
    F4 = get_form_data('F4') # felső gerinc alakja
    F6 = get_form_data('F6') # interalveolaris szög
    
    # Initialize A2 variables to None
    A2_nas_file_path_gerinc = None
    A2_nas_file_path_bukkal = None
    A2_nas_file_path_lingual = None
    
    # Only process A2 file uploads if lower jaw or both jaws are selected
    if jaw_selection in ['lower', 'both']:
        try:
            if 'stlFile_also_gerincelvonal' in request.files:
                A2_gerinc = request.files['stlFile_also_gerincelvonal']
                if A2_gerinc.filename:  # Check if file was actually uploaded
                    filename_alsogerinc = secure_filename(A2_gerinc.filename)
                    if '.' in filename_alsogerinc and filename_alsogerinc.rsplit('.', 1)[1].lower() == 'stl':
                        file_path_alsogerinc = os.path.join(app.config['UPLOAD_FOLDER'], filename_alsogerinc)
                        A2_gerinc.save(file_path_alsogerinc)
                        # Upload to NAS and get the NAS path
                        A2_nas_file_path_gerinc = upload_to_nas(file_path_alsogerinc, TAJ, 'A2_gerinc')
                        # Clean up the temporary file
                        os.remove(file_path_alsogerinc)
        except Exception as e:
            return render_template('error.html', message=f"Az alsó gerincélvonal STL feltöltésével probléma van. Kérlek próbáld újra! Üzenet: {str(e)}")
        
        try:
            if 'stlFile_also_bukkalis' in request.files:
                A2_bukkal = request.files["stlFile_also_bukkalis"]
                if A2_bukkal.filename:  # Check if file was actually uploaded
                    filename_alsobukkal = secure_filename(A2_bukkal.filename)
                    if '.' in filename_alsobukkal and filename_alsobukkal.rsplit('.', 1)[1].lower() == 'stl':
                        file_path_alsobukkal = os.path.join(app.config['UPLOAD_FOLDER'], filename_alsobukkal)
                        A2_bukkal.save(file_path_alsobukkal)
                        A2_nas_file_path_bukkal = upload_to_nas(file_path_alsobukkal, TAJ, 'A2_bukkalis')
                        os.remove(file_path_alsobukkal)
        except Exception as e:
            return render_template('error.html', message=f"Az alsó bukkális STL feltöltésével probléma van. Kérlek próbáld újra! Üzenet: {str(e)}")

        try:
            if 'stlFile_also_lingualis' in request.files:
                A2_lingual = request.files["stlFile_also_lingualis"]
                if A2_lingual.filename:  # Check if file was actually uploaded
                    filename_alsolingual = secure_filename(A2_lingual.filename)
                    if '.' in filename_alsolingual and filename_alsolingual.rsplit('.', 1)[1].lower() == 'stl':
                        file_path_alsolingual = os.path.join(app.config['UPLOAD_FOLDER'], filename_alsolingual)
                        A2_lingual.save(file_path_alsolingual)
                        A2_nas_file_path_lingual = upload_to_nas(file_path_alsolingual, TAJ, 'A2_lingualis')
                        os.remove(file_path_alsolingual)
        except Exception as e:
            return render_template('error.html', message=f"Az alsó lingualis STL feltöltésével probléma van. Kérlek próbáld újra! Üzenet: {str(e)}")

    A10 = get_form_data('A10') # állcsontreláció szögértéke

    print(request.form)
    print(request.files)  


    # Check if TAJ exists
    cursor.execute('SELECT COUNT(*) FROM patients WHERE "TAJ" = %s', (TAJ,))
    result = cursor.fetchone()
    
    if result[0] == 0:
        cursor.close()
        return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

    try:
        sql = """
            UPDATE patients SET
            "F1" = %s, "F2" = %s, "F3" = %s, "F4" = %s, "F6" = %s,
            "A2_gerincelvonal" = %s, "A2_bukkalisathajlas" = %s, "A2_lingualisathajlas" = %s, "A10" = %s
            WHERE "TAJ" = %s
            """
        values = (F1, F2, F3, F4, F6,
                  A2_nas_file_path_gerinc, A2_nas_file_path_bukkal, A2_nas_file_path_lingual, A10,
                  TAJ)
        cursor.execute(sql, values)
        db.commit()
        return render_template('confirmation.html')
    except Exception as e:
        db.rollback()
        error_msg = f"Adatbázis hiba történt: {str(e)}"
        print(f"Error in submit_questionnaire2: {error_msg}")
        return render_template('error.html', message=error_msg)
    finally:
        cursor.close()

@app.route('/submit_questionnaire3', methods=['POST'])
def submit_questionnaire3():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    
    # Function to fetch form data safely
    def get_form_data(field_name):
        return request.form.get(field_name, '')

    # Fetch responses for today's situation
    responsiveness_today_situation_recall = get_form_data('responsiveness_today_situation_recall')
    responsiveness_change = get_form_data('responsiveness_change')
    chewing_today_situation_recall = get_form_data('chewing_today_situation_recall')
    chewing_change = get_form_data('chewing_change')

    
    # Fetch responses for OHIP recall questions
    OHIP_1_recall = get_form_data('OHIP_1_recall')
    OHIP_2_recall = get_form_data('OHIP_2_recall')
    OHIP_3_recall = get_form_data('OHIP_3_recall')
    OHIP_4_recall = get_form_data('OHIP_4_recall')
    OHIP_5_recall = get_form_data('OHIP_5_recall')
    
    # Fetch responses for GOHAI recall questions
    GOHAI_1_recall = get_form_data('GOHAI_1_recall')
    GOHAI_2_recall = get_form_data('GOHAI_2_recall')
    GOHAI_3_recall = get_form_data('GOHAI_3_recall')
    GOHAI_4_recall = get_form_data('GOHAI_4_recall')
    GOHAI_5_recall = get_form_data('GOHAI_5_recall')
    GOHAI_6_recall = get_form_data('GOHAI_6_recall')
    GOHAI_7_recall = get_form_data('GOHAI_7_recall')
    GOHAI_8_recall = get_form_data('GOHAI_8_recall')
    GOHAI_9_recall = get_form_data('GOHAI_9_recall')
    GOHAI_10_recall = get_form_data('GOHAI_10_recall')
    GOHAI_11_recall = get_form_data('GOHAI_11_recall')
    GOHAI_12_recall = get_form_data('GOHAI_12_recall')

    # Fetch F9: A garatreflex erőssége
    F9 = get_form_data('F9')

    # Fetch dropout status (checkbox: checked = 1, unchecked = 0)
    dropout = True if 'dropout' in request.form and request.form['dropout'] == '1' else False

    # Check if TAJ exists
    cursor.execute('SELECT COUNT(*) FROM patients WHERE "TAJ" = %s', (TAJ,))
    result = cursor.fetchone()
    
    if result[0] == 0:
        return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

    sql = """
    UPDATE patients SET 
    "responsiveness_today_situation_recall" = %s, "responsiveness_change" = %s,
    "chewing_today_situation_recall" = %s, "chewing_change" = %s,
    "OHIP_1_recall" = %s, "OHIP_2_recall" = %s, "OHIP_3_recall" = %s, "OHIP_4_recall" = %s, "OHIP_5_recall" = %s,
    "GOHAI_1_recall" = %s, "GOHAI_2_recall" = %s, "GOHAI_3_recall" = %s, "GOHAI_4_recall" = %s, "GOHAI_5_recall" = %s,
    "GOHAI_6_recall" = %s, "GOHAI_7_recall" = %s, "GOHAI_8_recall" = %s, "GOHAI_9_recall" = %s, "GOHAI_10_recall" = %s,
    "GOHAI_11_recall" = %s, "GOHAI_12_recall" = %s, "F9" = %s, "dropout" = %s
    WHERE "TAJ" = %s
    """
    values = (responsiveness_today_situation_recall, responsiveness_change,
              chewing_today_situation_recall, chewing_change,
              OHIP_1_recall, OHIP_2_recall, OHIP_3_recall, OHIP_4_recall, OHIP_5_recall,
              GOHAI_1_recall, GOHAI_2_recall, GOHAI_3_recall, GOHAI_4_recall, GOHAI_5_recall,
              GOHAI_6_recall, GOHAI_7_recall, GOHAI_8_recall, GOHAI_9_recall, GOHAI_10_recall,
              GOHAI_11_recall, GOHAI_12_recall, F9, dropout, TAJ)
    try:
        cursor.execute(sql, values)
        db.commit()
        return render_template('confirmation.html')
    finally:
        cursor.close()

@app.route('/upload_init_mai')
def upload_init_mai():
    return render_template('upload_init_mai.html')

@app.route('/upload_final_mai')
def upload_final_mai():
    return render_template('upload_final_mai.html')

@app.route('/submit_init_mai', methods=['POST'])
def submit_init_mai():
    cursor = get_db_cursor()
    try:
        TAJ = request.form['TAJ']

        cursor.execute('SELECT COUNT(*) FROM patients WHERE "TAJ" = %s', (TAJ,))
        if cursor.fetchone()[0] == 0:
            return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            mai = process_image(file_path)
            mai_huedegree = calculate_hue_circular_sd(file_path)
            nas_file_path = upload_to_nas(file_path, TAJ, 'mai_initial')
            os.remove(file_path)

            sql = """
            UPDATE patients SET
            "init_mai" = %s, "init_mai_huedegree" = %s, "init_image_path" = %s
            WHERE "TAJ" = %s
            """
            cursor.execute(sql, (mai, mai_huedegree, nas_file_path, TAJ))
            db.commit()
            return render_template('confirmation.html')
        else:
            flash('Allowed file types are tiff, tif')
            return redirect(request.url)
    finally:
        cursor.close()

@app.route('/submit_final_mai', methods=['POST'])
def submit_final_mai():
    cursor = get_db_cursor()
    try:
        TAJ = request.form['TAJ']

        cursor.execute('SELECT COUNT(*) FROM patients WHERE "TAJ" = %s', (TAJ,))
        if cursor.fetchone()[0] == 0:
            return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            mai = process_image(file_path)
            mai_huedegree = calculate_hue_circular_sd(file_path)
            nas_file_path = upload_to_nas(file_path, TAJ, 'mai_final')
            os.remove(file_path)

            sql = """
            UPDATE patients SET
            "final_mai" = %s, "final_mai_huedegree" = %s, "final_image_path" = %s
            WHERE "TAJ" = %s
            """
            cursor.execute(sql, (mai, mai_huedegree, nas_file_path, TAJ))
            db.commit()
            return render_template('confirmation.html')
        else:
            flash('Allowed file types are tiff, tif')
            return redirect(request.url)
    finally:
        cursor.close()


def calculate_age(birthdate):
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age


def calculate_odds_ratios_and_ci(data, feature):
    try:
        if data[feature].sum() < 2 or data[feature].sum() > (len(data) - 2):
            # Not enough variation in the feature data, skip this calculation
            return np.nan, (np.nan, np.nan)
        model = sm.Logit(data['MAI_változás_binary'], sm.add_constant(data[feature]))
        result = model.fit(disp=False)
        odds_ratio = np.exp(result.params[1])
        ci_lower, ci_upper = np.exp(result.conf_int().iloc[1])
        return odds_ratio, (ci_lower, ci_upper)
    except (np.linalg.LinAlgError, IndexError, PerfectSeparationWarning):
        return np.nan, (np.nan, np.nan)
    

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def find_optimal_cutoff(risk_scores, outcome_values, outcome_type='continuous'):
    """
    Find optimal cut-off point for anatomical risk score using ROC analysis.
    This avoids p-hacking by using a single, principled method (Youden's index).
    
    Parameters:
    - risk_scores: array of anatomical risk counts
    - outcome_values: array of outcome values (OHIP, GOHAI, or responsiveness)
    - outcome_type: 'continuous' (OHIP/GOHAI) or 'categorical' (responsiveness)
    
    Returns:
    - optimal_cutoff: best cut-off value (using Youden's index or median)
    - best_effect_size: Cohen's d or AUC
    - best_p_value: p-value at optimal cut-off (with Bonferroni correction note)
    """
    df_temp = pd.DataFrame({'risk': risk_scores, 'outcome': outcome_values})
    df_temp = df_temp.dropna()
    
    if len(df_temp) < 10:
        return None, None, None
    
    max_risk = int(df_temp['risk'].max())
    min_risk = int(df_temp['risk'].min())
    
    if max_risk <= min_risk:
        return None, None, None
    
    # Use ROC analysis with Youden's index (maximizes sensitivity + specificity - 1)
    # This is a single, principled method that avoids multiple testing issues
    
    if outcome_type == 'continuous':
        # For continuous outcomes, define "poor outcome" as above median
        median_outcome = np.median(df_temp['outcome'])
        poor_outcome = (df_temp['outcome'] > median_outcome).astype(int)
        
        if len(poor_outcome.unique()) < 2:
            # Fallback to median cut-off
            optimal_cutoff = int(df_temp['risk'].median())
            low_group = df_temp[df_temp['risk'] <= optimal_cutoff]['outcome'].values
            high_group = df_temp[df_temp['risk'] > optimal_cutoff]['outcome'].values
            if len(low_group) >= 5 and len(high_group) >= 5:
                mean_low = np.mean(low_group)
                mean_high = np.mean(high_group)
                pooled_std = np.sqrt((np.var(low_group, ddof=1) + np.var(high_group, ddof=1)) / 2)
                if pooled_std > 0:
                    effect_size = abs(mean_high - mean_low) / pooled_std
                    try:
                        _, p_val = mannwhitneyu(low_group, high_group, alternative='two-sided')
                    except:
                        p_val = 1.0
                else:
                    effect_size = None
                    p_val = 1.0
            else:
                effect_size = None
                p_val = 1.0
            return optimal_cutoff, effect_size, p_val
        
        # Use ROC curve to find optimal cut-off
        try:
            fpr, tpr, thresholds = roc_curve(poor_outcome, df_temp['risk'])
            # Youden's index: J = max(sensitivity + specificity - 1) = max(tpr - fpr)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_cutoff = int(np.ceil(thresholds[optimal_idx])) if optimal_idx < len(thresholds) else int(df_temp['risk'].median())
            
            # Ensure cutoff is within valid range
            if optimal_cutoff < min_risk + 1:
                optimal_cutoff = min_risk + 1
            if optimal_cutoff >= max_risk:
                optimal_cutoff = max_risk - 1
            
            # Calculate stats at this cut-off
            low_group = df_temp[df_temp['risk'] <= optimal_cutoff]['outcome'].values
            high_group = df_temp[df_temp['risk'] > optimal_cutoff]['outcome'].values
            
            if len(low_group) >= 5 and len(high_group) >= 5:
                mean_low = np.mean(low_group)
                mean_high = np.mean(high_group)
                pooled_std = np.sqrt((np.var(low_group, ddof=1) + np.var(high_group, ddof=1)) / 2)
                if pooled_std > 0:
                    effect_size = abs(mean_high - mean_low) / pooled_std
                    try:
                        _, p_val = mannwhitneyu(low_group, high_group, alternative='two-sided')
                    except:
                        p_val = 1.0
                else:
                    effect_size = None
                    p_val = 1.0
            else:
                # Fallback to median
                optimal_cutoff = int(df_temp['risk'].median())
                low_group = df_temp[df_temp['risk'] <= optimal_cutoff]['outcome'].values
                high_group = df_temp[df_temp['risk'] > optimal_cutoff]['outcome'].values
                if len(low_group) >= 5 and len(high_group) >= 5:
                    mean_low = np.mean(low_group)
                    mean_high = np.mean(high_group)
                    pooled_std = np.sqrt((np.var(low_group, ddof=1) + np.var(high_group, ddof=1)) / 2)
                    if pooled_std > 0:
                        effect_size = abs(mean_high - mean_low) / pooled_std
                        try:
                            _, p_val = mannwhitneyu(low_group, high_group, alternative='two-sided')
                        except:
                            p_val = 1.0
                    else:
                        effect_size = None
                        p_val = 1.0
                else:
                    effect_size = None
                    p_val = 1.0
            
            return optimal_cutoff, effect_size, p_val
            
        except:
            # Fallback to median
            optimal_cutoff = int(df_temp['risk'].median())
            low_group = df_temp[df_temp['risk'] <= optimal_cutoff]['outcome'].values
            high_group = df_temp[df_temp['risk'] > optimal_cutoff]['outcome'].values
            if len(low_group) >= 5 and len(high_group) >= 5:
                mean_low = np.mean(low_group)
                mean_high = np.mean(high_group)
                pooled_std = np.sqrt((np.var(low_group, ddof=1) + np.var(high_group, ddof=1)) / 2)
                if pooled_std > 0:
                    effect_size = abs(mean_high - mean_low) / pooled_std
                    try:
                        _, p_val = mannwhitneyu(low_group, high_group, alternative='two-sided')
                    except:
                        p_val = 1.0
                else:
                    effect_size = None
                    p_val = 1.0
            else:
                effect_size = None
                p_val = 1.0
            return optimal_cutoff, effect_size, p_val
    
    elif outcome_type == 'categorical':
        # For responsiveness: define "poor" as <= 3 (Átlagos or worse)
        poor_outcome = (df_temp['outcome'] <= 3).astype(int)
        
        if len(poor_outcome.unique()) < 2:
            # Fallback to median
            optimal_cutoff = int(df_temp['risk'].median())
            return optimal_cutoff, None, 1.0
        
        # Use ROC curve with Youden's index
        try:
            fpr, tpr, thresholds = roc_curve(poor_outcome, df_temp['risk'])
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_cutoff = int(np.ceil(thresholds[optimal_idx])) if optimal_idx < len(thresholds) else int(df_temp['risk'].median())
            
            # Ensure cutoff is within valid range
            if optimal_cutoff < min_risk + 1:
                optimal_cutoff = min_risk + 1
            if optimal_cutoff >= max_risk:
                optimal_cutoff = max_risk - 1
            
            # Calculate AUC
            auc = roc_auc_score(poor_outcome, (df_temp['risk'] > optimal_cutoff).astype(int))
            
            # Calculate p-value
            from scipy.stats import chi2_contingency
            high_risk = (df_temp['risk'] > optimal_cutoff).astype(int)
            contingency = pd.crosstab(high_risk, poor_outcome)
            if contingency.shape == (2, 2):
                chi2, p_val, _, _ = chi2_contingency(contingency)
            else:
                p_val = 1.0
            
            return optimal_cutoff, auc, p_val
            
        except:
            # Fallback to median
            optimal_cutoff = int(df_temp['risk'].median())
            return optimal_cutoff, None, 1.0
    
    # Final fallback
    optimal_cutoff = int(df_temp['risk'].median())
    return optimal_cutoff, None, 1.0


def perform_cross_sectional_analysis(cursor):
    """Perform cross-sectional statistical analyses for all hypotheses."""
    results = {}
    
    # Route legacy init_mai requests to the new hue-degree MAI field.
    def _actual_db_field(field_name):
        if field_name == 'init_mai':
            return 'init_mai_huedegree'
        return field_name
    
    # Helper function to check data availability
    def check_data_availability(required_fields):
        """Check if sufficient data exists for analysis."""
        conditions = " AND ".join([f'"{_actual_db_field(field)}" IS NOT NULL' for field in required_fields])
        query = f"SELECT COUNT(*) FROM patients WHERE {conditions}"
        cursor.execute(query)
        count = cursor.fetchone()[0]
        return count >= 10  # Minimum sample size
    
    # Helper function to get data
    def get_data(fields, conditions=None):
        """Fetch data for analysis."""
        db_fields = [_actual_db_field(field) for field in fields]
        where_clause = " AND ".join([f'"{field}" IS NOT NULL' for field in db_fields])
        if conditions:
            conditions = conditions.replace('"init_mai"', '"init_mai_huedegree"')
            where_clause += " AND " + conditions
        query = f'''SELECT {', '.join(f'"{f}"' for f in db_fields)} FROM patients WHERE {where_clause}'''
        cursor.execute(query)
        return cursor.fetchall()
    
    # H1a: init_mai vs chewing_today_situation
    try:
        if check_data_availability(['init_mai', 'chewing_today_situation']):
            data = get_data(['init_mai', 'chewing_today_situation'])
            df = pd.DataFrame(data, columns=['init_mai', 'chewing_today_situation'])
            df = df.dropna()
            
            if len(df) >= 10:
                categories = df['chewing_today_situation'].unique()
                if len(categories) >= 2:
                    groups = [df[df['chewing_today_situation'] == cat]['init_mai'].values for cat in categories]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) >= 2:
                        # Check normality
                        normal = all([len(g) >= 3 and (shapiro(g)[1] > 0.05 if len(g) <= 5000 else normaltest(g)[1] > 0.05) for g in groups])
                        if normal and len(groups) == 2:
                            stat, p_value = ttest_ind(groups[0], groups[1])
                            test_name = "t-test"
                        elif normal:
                            stat, p_value = f_oneway(*groups)
                            test_name = "ANOVA"
                        else:
                            if len(groups) == 2:
                                stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                                test_name = "Mann-Whitney U"
                            else:
                                stat, p_value = kruskal(*groups)
                                test_name = "Kruskal-Wallis"
                        
                        desc_stats = df.groupby('chewing_today_situation')['init_mai'].agg(['mean', 'std', 'count']).to_dict('index')
                        
                        # Create box plot
                        fig = plt.figure(figsize=(10, 6))
                        categories_ordered = ["Kiváló", "Jó", "Átlagos", "Rossz", "Nagyon rossz"]
                        categories_present = [cat for cat in categories_ordered if cat in categories]
                        data_to_plot = [df[df['chewing_today_situation'] == cat]['init_mai'].values for cat in categories_present]
                        plt.boxplot(data_to_plot, labels=categories_present)
                        plt.xlabel('Szubjektív rágóképesség kategória')
                        plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                        plt.title(f'init_mai eloszlása szubjektív rágóképesség kategóriák szerint\n({test_name}, p = {p_value:.4f})')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plot_img = plot_to_base64(fig)
                        plt.close(fig)
                        
                        results['H1a'] = {
                            'status': 'success',
                            'test_name': test_name,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'n': len(df),
                            'descriptive_stats': desc_stats,
                            'plot_img': plot_img,
                            'data_needed': 'init_mai + chewing_today_situation',
                            'null_hypothesis': 'There is no difference in init_mai between subjective chewing categories.',
                            'alternative_hypothesis': 'Patients with better subjective chewing ratings have lower init_mai (better objective chewing).'
                        }
                    else:
                        results['H1a'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'init_mai + chewing_today_situation', 'n': len(df)}
                else:
                    results['H1a'] = {'status': 'insufficient_data', 'message': 'Not enough categories', 'data_needed': 'init_mai + chewing_today_situation', 'n': len(df)}
            else:
                results['H1a'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + chewing_today_situation', 'n': len(df)}
        else:
            results['H1a'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + chewing_today_situation'}
    except Exception as e:
        results['H1a'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + chewing_today_situation'}
    
    # H1b: init_mai vs OHIP_total
    try:
        if check_data_availability(['init_mai', 'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5']):
            data = get_data(['init_mai', 'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5'])
            df = pd.DataFrame(data, columns=['init_mai', 'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5'])
            df['OHIP_total'] = df[['OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5']].sum(axis=1)
            df = df[['init_mai', 'OHIP_total']].dropna()
            
            if len(df) >= 10:
                corr, p_value = spearmanr(df['init_mai'], df['OHIP_total'])
                
                # Create scatter plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(df['OHIP_total'], df['init_mai'], alpha=0.6)
                # Add regression line
                z = np.polyfit(df['OHIP_total'], df['init_mai'], 1)
                p = np.poly1d(z)
                plt.plot(df['OHIP_total'], p(df['OHIP_total']), "r--", alpha=0.8, label=f'Trend line')
                plt.xlabel('OHIP_total (magasabb = rosszabb QoL)')
                plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                plt.title(f'init_mai vs OHIP_total korreláció\n(Spearman r = {corr:.3f}, p = {p_value:.4f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                results['H1b'] = {
                    'status': 'success',
                    'test_name': 'Spearman correlation',
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'n': len(df),
                    'descriptive_stats': {
                        'init_mai': {'mean': float(df['init_mai'].mean()), 'std': float(df['init_mai'].std())},
                        'OHIP_total': {'mean': float(df['OHIP_total'].mean()), 'std': float(df['OHIP_total'].std())}
                    },
                    'plot_img': plot_img,
                    'data_needed': 'init_mai + OHIP_total',
                    'null_hypothesis': 'init_mai is not correlated with OHIP.',
                    'alternative_hypothesis': 'Higher OHIP (worse QoL) is associated with higher init_mai (worse objective chewing).'
                }
            else:
                results['H1b'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + OHIP_total', 'n': len(df)}
        else:
            results['H1b'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + OHIP_total'}
    except Exception as e:
        results['H1b'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + OHIP_total'}
    
    # H1c: init_mai vs GOHAI_total
    try:
        gohai_cols = [f'GOHAI_{i}' for i in range(1, 13)]
        if check_data_availability(['init_mai'] + gohai_cols):
            data = get_data(['init_mai'] + gohai_cols)
            df = pd.DataFrame(data, columns=['init_mai'] + gohai_cols)
            df['GOHAI_total'] = df[gohai_cols].sum(axis=1)
            df = df[['init_mai', 'GOHAI_total']].dropna()
            
            if len(df) >= 10:
                corr, p_value = spearmanr(df['init_mai'], df['GOHAI_total'])
                
                # Create scatter plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(df['GOHAI_total'], df['init_mai'], alpha=0.6)
                # Add regression line
                z = np.polyfit(df['GOHAI_total'], df['init_mai'], 1)
                p = np.poly1d(z)
                plt.plot(df['GOHAI_total'], p(df['GOHAI_total']), "r--", alpha=0.8, label=f'Trend line')
                plt.xlabel('GOHAI_total (magasabb = jobb QoL)')
                plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                plt.title(f'init_mai vs GOHAI_total korreláció\n(Spearman r = {corr:.3f}, p = {p_value:.4f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                results['H1c'] = {
                    'status': 'success',
                    'test_name': 'Spearman correlation',
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'n': len(df),
                    'descriptive_stats': {
                        'init_mai': {'mean': float(df['init_mai'].mean()), 'std': float(df['init_mai'].std())},
                        'GOHAI_total': {'mean': float(df['GOHAI_total'].mean()), 'std': float(df['GOHAI_total'].std())}
                    },
                    'plot_img': plot_img,
                    'data_needed': 'init_mai + GOHAI_total',
                    'null_hypothesis': 'init_mai is not correlated with GOHAI.',
                    'alternative_hypothesis': 'Higher GOHAI (better QoL) is associated with lower init_mai (better objective chewing).'
                }
            else:
                results['H1c'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + GOHAI_total', 'n': len(df)}
        else:
            results['H1c'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + GOHAI_total'}
    except Exception as e:
        results['H1c'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + GOHAI_total'}
    
    # H3a: init_mai vs denture_type
    try:
        if check_data_availability(['init_mai', 'denture_type']):
            data = get_data(['init_mai', 'denture_type'])
            df = pd.DataFrame(data, columns=['init_mai', 'denture_type'])
            df = df.dropna()
            
            if len(df) >= 10:
                categories = df['denture_type'].unique()
                if len(categories) >= 2:
                    groups = [df[df['denture_type'] == cat]['init_mai'].values for cat in categories]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) >= 2:
                        if len(groups) == 2:
                            stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                            test_name = "Mann-Whitney U"
                        else:
                            stat, p_value = kruskal(*groups)
                            test_name = "Kruskal-Wallis"
                        
                        desc_stats = df.groupby('denture_type')['init_mai'].agg(['mean', 'std', 'count']).to_dict('index')
                        
                        # Create box plot
                        fig = plt.figure(figsize=(10, 6))
                        categories_present = list(categories)
                        data_to_plot = [df[df['denture_type'] == cat]['init_mai'].values for cat in categories_present]
                        plt.boxplot(data_to_plot, labels=categories_present)
                        plt.xlabel('Fogpótlás típusa')
                        plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                        plt.title(f'init_mai eloszlása fogpótlás típus szerint\n({test_name}, p = {p_value:.4f})')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plot_img = plot_to_base64(fig)
                        plt.close(fig)
                        
                        results['H3a'] = {
                            'status': 'success',
                            'test_name': test_name,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'n': len(df),
                            'descriptive_stats': desc_stats,
                            'plot_img': plot_img,
                            'data_needed': 'init_mai + denture_type',
                            'null_hypothesis': 'Denture type does not affect init_mai.',
                            'alternative_hypothesis': 'Lower or both dentures have higher init_mai (worse objective chewing) than upper-only.'
                        }
                    else:
                        results['H3a'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'init_mai + denture_type', 'n': len(df)}
                else:
                    results['H3a'] = {'status': 'insufficient_data', 'message': 'Not enough categories', 'data_needed': 'init_mai + denture_type', 'n': len(df)}
            else:
                results['H3a'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + denture_type', 'n': len(df)}
        else:
            results['H3a'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + denture_type'}
    except Exception as e:
        results['H3a'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + denture_type'}
    
    # H4a: init_mai vs age
    try:
        if check_data_availability(['init_mai', 'birthdate']):
            data = get_data(['init_mai', 'birthdate'])
            df = pd.DataFrame(data, columns=['init_mai', 'birthdate'])
            df['age'] = df['birthdate'].apply(calculate_age)
            df = df[['init_mai', 'age']].dropna()
            
            if len(df) >= 10:
                corr, p_value = spearmanr(df['init_mai'], df['age'])
                
                # Create scatter plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(df['age'], df['init_mai'], alpha=0.6)
                # Add regression line
                z = np.polyfit(df['age'], df['init_mai'], 1)
                p = np.poly1d(z)
                plt.plot(df['age'], p(df['age']), "r--", alpha=0.8, label=f'Trend line')
                plt.xlabel('Kor (év)')
                plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                plt.title(f'init_mai vs kor korreláció\n(Spearman r = {corr:.3f}, p = {p_value:.4f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                results['H4a'] = {
                    'status': 'success',
                    'test_name': 'Spearman correlation',
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'n': len(df),
                    'descriptive_stats': {
                        'init_mai': {'mean': float(df['init_mai'].mean()), 'std': float(df['init_mai'].std())},
                        'age': {'mean': float(df['age'].mean()), 'std': float(df['age'].std())}
                    },
                    'plot_img': plot_img,
                    'data_needed': 'init_mai + age',
                    'null_hypothesis': 'Age is not associated with init_mai.',
                    'alternative_hypothesis': 'Older age predicts higher init_mai (worse objective chewing).'
                }
            else:
                results['H4a'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + age', 'n': len(df)}
        else:
            results['H4a'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + age'}
    except Exception as e:
        results['H4a'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + age'}
    
    # H5: init_mai vs gender
    try:
        if check_data_availability(['init_mai', 'gender']):
            data = get_data(['init_mai', 'gender'])
            df = pd.DataFrame(data, columns=['init_mai', 'gender'])
            df = df.dropna()
            
            if len(df) >= 10:
                categories = df['gender'].unique()
                if len(categories) >= 2:
                    groups = [df[df['gender'] == cat]['init_mai'].values for cat in categories]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) == 2:
                        stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        test_name = "Mann-Whitney U"
                        
                        desc_stats = df.groupby('gender')['init_mai'].agg(['mean', 'std', 'count']).to_dict('index')
                        
                        # Create box plot
                        fig = plt.figure(figsize=(10, 6))
                        categories_present = list(categories)
                        data_to_plot = [df[df['gender'] == cat]['init_mai'].values for cat in categories_present]
                        plt.boxplot(data_to_plot, labels=categories_present)
                        plt.xlabel('Nem')
                        plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                        plt.title(f'init_mai eloszlása nem szerint\n({test_name}, p = {p_value:.4f})')
                        plt.tight_layout()
                        plot_img = plot_to_base64(fig)
                        plt.close(fig)
                        
                        results['H5'] = {
                            'status': 'success',
                            'test_name': test_name,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'n': len(df),
                            'descriptive_stats': desc_stats,
                            'plot_img': plot_img,
                            'data_needed': 'init_mai + gender',
                            'null_hypothesis': 'Gender has no association with init_mai.',
                            'alternative_hypothesis': 'Gender differences exist in init_mai (exploratory).'
                        }
                    else:
                        results['H5'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'init_mai + gender', 'n': len(df)}
                else:
                    results['H5'] = {'status': 'insufficient_data', 'message': 'Not enough categories', 'data_needed': 'init_mai + gender', 'n': len(df)}
            else:
                results['H5'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + gender', 'n': len(df)}
        else:
            results['H5'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + gender'}
    except Exception as e:
        results['H5'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + gender'}
    
    # H2a: init_mai vs F5 (mandibular ridge form)
    # Note: F1 is now a numeric value (mm), so we use F5 which is categorical
    try:
        if check_data_availability(['init_mai', 'F5']):
            data = get_data(['init_mai', 'F5'])
            df = pd.DataFrame(data, columns=['init_mai', 'F5'])
            df = df.dropna()
            df['F5'] = df['F5'].astype(str)  # Ensure categorical
            
            if len(df) >= 10:
                categories = df['F5'].unique()
                if len(categories) >= 2:
                    groups = [df[df['F5'] == cat]['init_mai'].values for cat in categories]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) >= 2:
                        if len(groups) == 2:
                            stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                            test_name = "Mann-Whitney U"
                        else:
                            stat, p_value = kruskal(*groups)
                            test_name = "Kruskal-Wallis"
                        
                        desc_stats = df.groupby('F5')['init_mai'].agg(['mean', 'std', 'count']).to_dict('index')
                        
                        # Create box plot
                        fig = plt.figure(figsize=(10, 6))
                        categories_present = sorted([c for c in categories if c in df['F5'].values])
                        data_to_plot = [df[df['F5'] == cat]['init_mai'].values for cat in categories_present]
                        plt.boxplot(data_to_plot, labels=[f'F5={c}' for c in categories_present])
                        plt.xlabel('F5 (Mandibular ridge form)')
                        plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                        plt.title(f'init_mai eloszlása F5 kategóriák szerint\n({test_name}, p = {p_value:.4f})')
                        plt.tight_layout()
                        plot_img = plot_to_base64(fig)
                        plt.close(fig)
                        
                        results['H2a'] = {
                            'status': 'success',
                            'test_name': test_name,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'n': len(df),
                            'descriptive_stats': desc_stats,
                            'plot_img': plot_img,
                            'data_needed': 'init_mai + F5',
                            'null_hypothesis': 'Mandibular ridge form (F5) has no association with init_mai.',
                            'alternative_hypothesis': 'Unfavourable mandibular ridge form predicts higher init_mai (worse objective chewing).'
                        }
                    else:
                        results['H2a'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'init_mai + F5', 'n': len(df)}
                else:
                    results['H2a'] = {'status': 'insufficient_data', 'message': 'Not enough categories', 'data_needed': 'init_mai + F5', 'n': len(df)}
            else:
                results['H2a'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + F5', 'n': len(df)}
        else:
            results['H2a'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + F5'}
    except Exception as e:
        results['H2a'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + F5'}
    
    # H2b: init_mai vs A3-A9 values (average or max of left/right)
    try:
        a_vars = ['A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal', 
                  'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal', 'A9_jobb', 'A9_bal']
        required_vars = ['init_mai'] + a_vars
        if check_data_availability(required_vars):
            data = get_data(required_vars)
            df = pd.DataFrame(data, columns=required_vars)
            # Calculate average A3-A9 score (higher = worse)
            df['A3_avg'] = df[['A3_jobb', 'A3_bal']].mean(axis=1)
            df['A4_avg'] = df[['A4_jobb', 'A4_bal']].mean(axis=1)
            df['A5_avg'] = df[['A5_jobb', 'A5_bal']].mean(axis=1)
            df['A6_avg'] = df[['A6_jobb', 'A6_bal']].mean(axis=1)
            df['A7_avg'] = df[['A7_jobb', 'A7_bal']].mean(axis=1)
            df['A8_avg'] = df[['A8_jobb', 'A8_bal']].mean(axis=1)
            df['A9_avg'] = df[['A9_jobb', 'A9_bal']].mean(axis=1)
            df['A3_A9_mean'] = df[['A3_avg', 'A4_avg', 'A5_avg', 'A6_avg', 'A7_avg', 'A8_avg', 'A9_avg']].mean(axis=1)
            df = df[['init_mai', 'A3_A9_mean']].dropna()
            
            if len(df) >= 10:
                corr, p_value = spearmanr(df['init_mai'], df['A3_A9_mean'])
                
                # Create scatter plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(df['A3_A9_mean'], df['init_mai'], alpha=0.6)
                z = np.polyfit(df['A3_A9_mean'], df['init_mai'], 1)
                p = np.poly1d(z)
                plt.plot(df['A3_A9_mean'], p(df['A3_A9_mean']), "r--", alpha=0.8, label='Trend line')
                plt.xlabel('A3-A9 átlag (magasabb = kedvezőtlenebb anatómia)')
                plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                plt.title(f'init_mai vs A3-A9 átlag korreláció\n(Spearman r = {corr:.3f}, p = {p_value:.4f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                results['H2b'] = {
                    'status': 'success',
                    'test_name': 'Spearman correlation',
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'n': len(df),
                    'descriptive_stats': {
                        'init_mai': {'mean': float(df['init_mai'].mean()), 'std': float(df['init_mai'].std())},
                        'A3_A9_mean': {'mean': float(df['A3_A9_mean'].mean()), 'std': float(df['A3_A9_mean'].std())}
                    },
                    'plot_img': plot_img,
                    'data_needed': 'init_mai + A3-A9 values',
                    'null_hypothesis': 'A-variable undercuts and ridge shapes do not affect init_mai.',
                    'alternative_hypothesis': 'Unfavourable A3–A9 anatomy is associated with higher init_mai (worse objective chewing).'
                }
            else:
                results['H2b'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + A3-A9 values', 'n': len(df)}
        else:
            results['H2b'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + A3-A9 values'}
    except Exception as e:
        results['H2b'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + A3-A9 values'}
    
    # H2c: Anatomical risk score vs init_mai
    try:
        # Create risk score from multiple unfavorable features
        risk_vars = ['F5', 'F7', 'F8'] + a_vars
        required_vars = ['init_mai'] + [v for v in risk_vars if v]
        if check_data_availability(required_vars):
            data = get_data(required_vars)
            df = pd.DataFrame(data, columns=required_vars)
            
            # Calculate risk score: count unfavorable values (higher values = worse)
            # For F5, F7, F8: values 2-3 are unfavorable
            # For A variables: higher values are unfavorable
            risk_score = 0
            for var in ['F5', 'F7', 'F8']:
                if var in df.columns:
                    df[var] = pd.to_numeric(df[var], errors='coerce')
                    risk_score += (df[var] >= 2).astype(int)
            
            for var in a_vars:
                if var in df.columns:
                    df[var] = pd.to_numeric(df[var], errors='coerce')
                    # Higher A values = worse, so count values above median or threshold
                    if df[var].notna().sum() > 0:
                        threshold = df[var].median()
                        risk_score += (df[var] > threshold).astype(int)
            
            df['anatomical_risk_score'] = risk_score
            df = df[['init_mai', 'anatomical_risk_score']].dropna()
            
            if len(df) >= 10:
                corr, p_value = spearmanr(df['init_mai'], df['anatomical_risk_score'])
                
                # Try linear regression
                try:
                    X = sm.add_constant(df['anatomical_risk_score'])
                    y = df['init_mai']
                    model = sm.OLS(y, X).fit()
                    r_squared = model.rsquared
                    reg_p_value = model.pvalues[1]
                    coef = model.params[1]
                except:
                    r_squared = np.nan
                    reg_p_value = p_value
                    coef = corr
                
                # Create scatter plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(df['anatomical_risk_score'], df['init_mai'], alpha=0.6)
                z = np.polyfit(df['anatomical_risk_score'], df['init_mai'], 1)
                p = np.poly1d(z)
                plt.plot(df['anatomical_risk_score'], p(df['anatomical_risk_score']), "r--", alpha=0.8, label='Trend line')
                plt.xlabel('Anatómiai kockázati pontszám (magasabb = kedvezőtlenebb)')
                plt.ylabel('init_mai (magasabb = rosszabb rágóképesség)')
                plt.title(f'init_mai vs anatómiai kockázati pontszám\n(Spearman r = {corr:.3f}, p = {p_value:.4f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                results['H2c'] = {
                    'status': 'success',
                    'test_name': 'Spearman correlation + Linear regression',
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'regression_coefficient': float(coef),
                    'r_squared': float(r_squared) if not np.isnan(r_squared) else None,
                    'regression_p_value': float(reg_p_value),
                    'n': len(df),
                    'descriptive_stats': {
                        'init_mai': {'mean': float(df['init_mai'].mean()), 'std': float(df['init_mai'].std())},
                        'anatomical_risk_score': {'mean': float(df['anatomical_risk_score'].mean()), 'std': float(df['anatomical_risk_score'].std())}
                    },
                    'plot_img': plot_img,
                    'data_needed': 'anatomical_risk_score + init_mai',
                    'null_hypothesis': 'Number of unfavourable anatomical features does not influence init_mai.',
                    'alternative_hypothesis': 'Higher anatomical risk score predicts higher init_mai (worse objective chewing).'
                }
            else:
                results['H2c'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'anatomical_risk_score + init_mai', 'n': len(df)}
        else:
            results['H2c'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'anatomical_risk_score + init_mai'}
    except Exception as e:
        results['H2c'] = {'status': 'error', 'message': str(e), 'data_needed': 'anatomical_risk_score + init_mai'}
    
    # H3b: Multiple regression - denture_type adjusted for age and anatomy
    try:
        if check_data_availability(['init_mai', 'denture_type', 'birthdate', 'F5']):
            data = get_data(['init_mai', 'denture_type', 'birthdate', 'F5'])
            df = pd.DataFrame(data, columns=['init_mai', 'denture_type', 'birthdate', 'F5'])
            df['age'] = df['birthdate'].apply(calculate_age)
            df = df[['init_mai', 'denture_type', 'age', 'F5']].dropna()
            df['F5'] = pd.to_numeric(df['F5'], errors='coerce')
            
            if len(df) >= 15:  # Need more for multiple regression
                # Create dummy variables for denture_type
                df_dummies = pd.get_dummies(df['denture_type'], prefix='denture')
                df_reg = pd.concat([df[['init_mai', 'age', 'F5']], df_dummies], axis=1)
                
                # Remove reference category (first dummy)
                if len(df_dummies.columns) > 1:
                    X_cols = ['age', 'F5'] + [col for col in df_dummies.columns[1:]]
                    X = sm.add_constant(df_reg[X_cols])
                    y = df_reg['init_mai']
                    
                    model = sm.OLS(y, X).fit()
                    
                    # Extract denture type coefficients
                    denture_coefs = {}
                    for col in df_dummies.columns[1:]:
                        if col in model.params.index:
                            denture_coefs[col] = {
                                'coefficient': float(model.params[col]),
                                'p_value': float(model.pvalues[col])
                            }
                    
                    # Create summary plot
                    fig = plt.figure(figsize=(12, 6))
                    coefs = model.params[1:]  # Exclude intercept
                    pvals = model.pvalues[1:]
                    colors = ['red' if p < 0.05 else 'blue' for p in pvals]
                    plt.barh(range(len(coefs)), coefs.values, color=colors)
                    plt.yticks(range(len(coefs)), coefs.index)
                    plt.xlabel('Regressziós együttható')
                    plt.title(f'Többváltozós regresszió: init_mai ~ denture_type + age + F5\n(R² = {model.rsquared:.3f}, p = {model.f_pvalue:.4f})')
                    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plot_img = plot_to_base64(fig)
                    plt.close(fig)
                    
                    results['H3b'] = {
                        'status': 'success',
                        'test_name': 'Multiple linear regression',
                        'r_squared': float(model.rsquared),
                        'f_statistic': float(model.fvalue),
                        'f_p_value': float(model.f_pvalue),
                        'n': len(df),
                        'denture_coefficients': denture_coefs,
                        'age_coefficient': {'coefficient': float(model.params['age']), 'p_value': float(model.pvalues['age'])} if 'age' in model.params.index else None,
                        'F5_coefficient': {'coefficient': float(model.params['F5']), 'p_value': float(model.pvalues['F5'])} if 'F5' in model.params.index else None,
                        'plot_img': plot_img,
                        'data_needed': 'init_mai + denture_type + age + F5',
                        'null_hypothesis': 'Denture type is not an independent predictor of init_mai.',
                        'alternative_hypothesis': 'Denture type remains significant after adjusting for age and anatomy.'
                    }
                else:
                    results['H3b'] = {'status': 'insufficient_data', 'message': 'Not enough denture type categories', 'data_needed': 'init_mai + denture_type + age + F5', 'n': len(df)}
            else:
                results['H3b'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 15)', 'data_needed': 'init_mai + denture_type + age + F5', 'n': len(df)}
        else:
            results['H3b'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + denture_type + age + F5'}
    except Exception as e:
        results['H3b'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + denture_type + age + F5'}
    
    # H4b: Multiple regression - age adjusted for anatomy and denture_type
    try:
        if check_data_availability(['init_mai', 'birthdate', 'F5', 'denture_type']):
            data = get_data(['init_mai', 'birthdate', 'F5', 'denture_type'])
            df = pd.DataFrame(data, columns=['init_mai', 'birthdate', 'F5', 'denture_type'])
            df['age'] = df['birthdate'].apply(calculate_age)
            df = df[['init_mai', 'age', 'F5', 'denture_type']].dropna()
            df['F5'] = pd.to_numeric(df['F5'], errors='coerce')
            
            if len(df) >= 15:
                df_dummies = pd.get_dummies(df['denture_type'], prefix='denture')
                df_reg = pd.concat([df[['init_mai', 'age', 'F5']], df_dummies], axis=1)
                
                if len(df_dummies.columns) > 0:
                    X_cols = ['age', 'F5'] + list(df_dummies.columns)
                    X = sm.add_constant(df_reg[X_cols])
                    y = df_reg['init_mai']
                    
                    model = sm.OLS(y, X).fit()
                    
                    # Create summary plot
                    fig = plt.figure(figsize=(12, 6))
                    coefs = model.params[1:]
                    pvals = model.pvalues[1:]
                    colors = ['red' if p < 0.05 else 'blue' for p in pvals]
                    plt.barh(range(len(coefs)), coefs.values, color=colors)
                    plt.yticks(range(len(coefs)), coefs.index)
                    plt.xlabel('Regressziós együttható')
                    plt.title(f'Többváltozós regresszió: init_mai ~ age + F5 + denture_type\n(R² = {model.rsquared:.3f}, p = {model.f_pvalue:.4f})')
                    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plot_img = plot_to_base64(fig)
                    plt.close(fig)
                    
                    age_coef = {'coefficient': float(model.params['age']), 'p_value': float(model.pvalues['age'])} if 'age' in model.params.index else None
                    
                    results['H4b'] = {
                        'status': 'success',
                        'test_name': 'Multiple linear regression',
                        'r_squared': float(model.rsquared),
                        'f_statistic': float(model.fvalue),
                        'f_p_value': float(model.f_pvalue),
                        'n': len(df),
                        'age_coefficient': age_coef,
                        'plot_img': plot_img,
                        'data_needed': 'init_mai + age + F5 + denture_type',
                        'null_hypothesis': 'Age has no independent effect on init_mai.',
                        'alternative_hypothesis': 'Age remains significant in multivariable models (older age predicts higher init_mai).'
                    }
                else:
                    results['H4b'] = {'status': 'insufficient_data', 'message': 'Not enough variables', 'data_needed': 'init_mai + age + F5 + denture_type', 'n': len(df)}
            else:
                results['H4b'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 15)', 'data_needed': 'init_mai + age + F5 + denture_type', 'n': len(df)}
        else:
            results['H4b'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + age + F5 + denture_type'}
    except Exception as e:
        results['H4b'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + age + F5 + denture_type'}
    
    # H6a: Discordance analysis between init_mai and chewing_today_situation
    try:
        if check_data_availability(['init_mai', 'chewing_today_situation']):
            data = get_data(['init_mai', 'chewing_today_situation'])
            df = pd.DataFrame(data, columns=['init_mai', 'chewing_today_situation'])
            df = df.dropna()
            
            if len(df) >= 10:
                # Define concordance: good subjective (Kiváló/Jó) should have low MAI, bad subjective (Rossz/Nagyon rossz) should have high MAI
                # Create binary: good subjective = 1, bad subjective = 0
                good_subjective = ['Kiváló', 'Jó']
                df['subjective_good'] = df['chewing_today_situation'].isin(good_subjective).astype(int)
                
                # Define objective: low MAI = good (use median as cutoff)
                mai_median = df['init_mai'].median()
                df['objective_good'] = (df['init_mai'] <= mai_median).astype(int)
                
                # Concordance: both good or both bad
                df['concordant'] = (df['subjective_good'] == df['objective_good']).astype(int)
                
                concordance_rate = df['concordant'].mean()
                discordant_n = (df['concordant'] == 0).sum()
                concordant_n = (df['concordant'] == 1).sum()
                
                # Create cross-tabulation visualization
                fig = plt.figure(figsize=(10, 6))
                crosstab = pd.crosstab(df['subjective_good'], df['objective_good'], 
                                      rownames=['Szubjektív jó'], 
                                      colnames=['Objektív jó (alacsony MAI)'])
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Páciensek száma'})
                plt.title(f'Konkordanciamátrix: Szubjektív vs Objektív rágóképesség\n(Konkordancia arány: {concordance_rate:.2%})')
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                results['H6a'] = {
                    'status': 'success',
                    'test_name': 'Cross-tabulation analysis',
                    'concordance_rate': float(concordance_rate),
                    'discordant_n': int(discordant_n),
                    'concordant_n': int(concordant_n),
                    'n': len(df),
                    'plot_img': plot_img,
                    'data_needed': 'init_mai + chewing_today_situation',
                    'null_hypothesis': 'There is no discordance between subjective and objective chewing.',
                    'alternative_hypothesis': f'A subgroup ({discordant_n}/{len(df)} = {(1-concordance_rate):.1%}) shows mismatch between init_mai and self-rated chewing.'
                }
            else:
                results['H6a'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'init_mai + chewing_today_situation', 'n': len(df)}
        else:
            results['H6a'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + chewing_today_situation'}
    except Exception as e:
        results['H6a'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + chewing_today_situation'}
    
    # H6b: QoL differences between concordant and discordant patients
    try:
        # First need to calculate concordance (reuse H6a logic)
        if check_data_availability(['init_mai', 'chewing_today_situation', 'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5']):
            data = get_data(['init_mai', 'chewing_today_situation', 'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5'])
            df = pd.DataFrame(data, columns=['init_mai', 'chewing_today_situation', 'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5'])
            df['OHIP_total'] = df[['OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5']].sum(axis=1)
            df = df.dropna()
            
            if len(df) >= 10:
                good_subjective = ['Kiváló', 'Jó']
                df['subjective_good'] = df['chewing_today_situation'].isin(good_subjective).astype(int)
                mai_median = df['init_mai'].median()
                df['objective_good'] = (df['init_mai'] <= mai_median).astype(int)
                df['concordant'] = (df['subjective_good'] == df['objective_good']).astype(int)
                
                concordant_group = df[df['concordant'] == 1]['OHIP_total'].values
                discordant_group = df[df['concordant'] == 0]['OHIP_total'].values
                
                if len(concordant_group) > 0 and len(discordant_group) > 0:
                    stat, p_value = mannwhitneyu(concordant_group, discordant_group, alternative='two-sided')
                    
                    # Create box plot
                    fig = plt.figure(figsize=(10, 6))
                    data_to_plot = [concordant_group, discordant_group]
                    plt.boxplot(data_to_plot, labels=['Konkordáns', 'Diszkordáns'])
                    plt.ylabel('OHIP_total (magasabb = rosszabb QoL)')
                    plt.title(f'OHIP_total eloszlása konkordáns vs diszkordáns pácienseknél\n(Mann-Whitney U, p = {p_value:.4f})')
                    plt.tight_layout()
                    plot_img = plot_to_base64(fig)
                    plt.close(fig)
                    
                    results['H6b'] = {
                        'status': 'success',
                        'test_name': 'Mann-Whitney U',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'n': len(df),
                        'concordant_n': len(concordant_group),
                        'discordant_n': len(discordant_group),
                        'descriptive_stats': {
                            'concordant': {'mean': float(np.mean(concordant_group)), 'std': float(np.std(concordant_group))},
                            'discordant': {'mean': float(np.mean(discordant_group)), 'std': float(np.std(discordant_group))}
                        },
                        'plot_img': plot_img,
                        'data_needed': 'concordance_group + OHIP_total',
                        'null_hypothesis': 'QoL domains do not differ between concordant and discordant patients.',
                        'alternative_hypothesis': 'Discordant patients have distinct psychosocial OHIP patterns.'
                    }
                else:
                    results['H6b'] = {'status': 'insufficient_data', 'message': 'Not enough concordant/discordant groups', 'data_needed': 'concordance_group + OHIP_total', 'n': len(df)}
            else:
                results['H6b'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'concordance_group + OHIP_total', 'n': len(df)}
        else:
            results['H6b'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'concordance_group + OHIP_total'}
    except Exception as e:
        results['H6b'] = {'status': 'error', 'message': str(e), 'data_needed': 'concordance_group + OHIP_total'}
    
    # H7: Multivariable model predicting init_mai
    try:
        if check_data_availability(['init_mai', 'birthdate', 'gender', 'denture_type', 'F5']):
            data = get_data(['init_mai', 'birthdate', 'gender', 'denture_type', 'F5'])
            df = pd.DataFrame(data, columns=['init_mai', 'birthdate', 'gender', 'denture_type', 'F5'])
            df['age'] = df['birthdate'].apply(calculate_age)
            df = df[['init_mai', 'age', 'gender', 'denture_type', 'F5']].dropna()
            df['F5'] = pd.to_numeric(df['F5'], errors='coerce')
            
            if len(df) >= 20:  # Need more for multivariable model
                # Create dummy variables
                gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
                denture_dummies = pd.get_dummies(df['denture_type'], prefix='denture')
                
                df_reg = pd.concat([df[['init_mai', 'age', 'F5']], gender_dummies, denture_dummies], axis=1)
                
                X_cols = ['age', 'F5'] + [col for col in gender_dummies.columns[1:] if col in df_reg.columns] + [col for col in denture_dummies.columns[1:] if col in df_reg.columns]
                X = sm.add_constant(df_reg[X_cols])
                y = df_reg['init_mai']
                
                model = sm.OLS(y, X).fit()
                
                # Create coefficient plot
                fig = plt.figure(figsize=(12, 8))
                coefs = model.params[1:]  # Exclude intercept
                pvals = model.pvalues[1:]
                colors = ['red' if p < 0.05 else 'blue' for p in pvals]
                y_pos = range(len(coefs))
                plt.barh(y_pos, coefs.values, color=colors)
                plt.yticks(y_pos, coefs.index)
                plt.xlabel('Regressziós együttható')
                plt.title(f'Többváltozós modell: init_mai ~ age + gender + denture_type + F5\n(R² = {model.rsquared:.3f}, F = {model.fvalue:.2f}, p = {model.f_pvalue:.4f})')
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plot_img = plot_to_base64(fig)
                plt.close(fig)
                
                # Extract significant predictors
                significant_vars = [var for var in coefs.index if model.pvalues[var] < 0.05]
                
                results['H7'] = {
                    'status': 'success',
                    'test_name': 'Multiple linear regression',
                    'r_squared': float(model.rsquared),
                    'adjusted_r_squared': float(model.rsquared_adj),
                    'f_statistic': float(model.fvalue),
                    'f_p_value': float(model.f_pvalue),
                    'n': len(df),
                    'significant_predictors': significant_vars,
                    'coefficients': {var: {'coef': float(model.params[var]), 'p_value': float(model.pvalues[var])} for var in coefs.index},
                    'plot_img': plot_img,
                    'data_needed': 'init_mai + age + gender + denture_type + F5',
                    'null_hypothesis': 'Anatomy, age, and denture type do not predict init_mai as a group.',
                    'alternative_hypothesis': f'A multivariable model significantly predicts init_mai (R² = {model.rsquared:.3f}, p = {model.f_pvalue:.4f}).'
                }
            else:
                results['H7'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 20)', 'data_needed': 'init_mai + age + gender + denture_type + F5', 'n': len(df)}
        else:
            results['H7'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'init_mai + age + gender + denture_type + F5'}
    except Exception as e:
        results['H7'] = {'status': 'error', 'message': str(e), 'data_needed': 'init_mai + age + gender + denture_type + F5'}
    
    # H8: Cumulative anatomical risk factors vs baseline OHIP (dose-response effect)
    try:
        # Define anatomical risk variables according to specific rules
        # Upper jaw: F5, F7 (exclude F8 - prosthetic situation)
        # Lower jaw: A1_Kaan, A3-A9 (jobb+bal), A11-A13 (exclude A14 - prosthetic situation)
        required_vars = ['OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5', 
                        'F5', 'F7', 
                        'A1_Kaan',
                        'A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal',
                        'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal', 
                        'A9_jobb', 'A9_bal', 'A11', 'A12', 'A13']
        
        if check_data_availability(required_vars):
            data = get_data(required_vars)
            df = pd.DataFrame(data, columns=required_vars)
            
            # Calculate OHIP_total (baseline)
            df['OHIP_total'] = df[['OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5']].sum(axis=1)
            
            # Initialize risk count
            risk_count = pd.Series(0, index=df.index)
            
            # Upper jaw variables
            # F5: 1 = normal, 2-3 = problematic
            if 'F5' in df.columns:
                df['F5'] = pd.to_numeric(df['F5'], errors='coerce')
                risk_count += ((df['F5'] >= 2) & (df['F5'] <= 3)).astype(int)
            
            # F7: 1 = normal, 2-3 = problematic
            if 'F7' in df.columns:
                df['F7'] = pd.to_numeric(df['F7'], errors='coerce')
                risk_count += ((df['F7'] >= 2) & (df['F7'] <= 3)).astype(int)
            
            # Lower jaw variables
            # A1_Kaan: 1 = normal, 2-5 = problematic
            if 'A1_Kaan' in df.columns:
                df['A1_Kaan'] = pd.to_numeric(df['A1_Kaan'], errors='coerce')
                risk_count += ((df['A1_Kaan'] >= 2) & (df['A1_Kaan'] <= 5)).astype(int)
            
            # A3-A9: paired variables (jobb/bal)
            # Count as problematic if EITHER side is problematic
            a_pairs = [
                ('A3_jobb', 'A3_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A4_jobb', 'A4_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A5_jobb', 'A5_bal', 2, [1, 3]),  # 2 = normal, 1 or 3 = problematic (special case!)
                ('A6_jobb', 'A6_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A7_jobb', 'A7_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A8_jobb', 'A8_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A9_jobb', 'A9_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
            ]
            
            for left_var, right_var, normal_val, problematic_vals in a_pairs:
                if left_var in df.columns and right_var in df.columns:
                    df[left_var] = pd.to_numeric(df[left_var], errors='coerce')
                    df[right_var] = pd.to_numeric(df[right_var], errors='coerce')
                    # Count if either side is problematic
                    left_problematic = df[left_var].isin(problematic_vals)
                    right_problematic = df[right_var].isin(problematic_vals)
                    risk_count += (left_problematic | right_problematic).astype(int)
            
            # A11: 2 = normal, 1 or 3 = problematic (special case!)
            if 'A11' in df.columns:
                df['A11'] = pd.to_numeric(df['A11'], errors='coerce')
                risk_count += ((df['A11'] == 1) | (df['A11'] == 3)).astype(int)
            
            # A12: 1 = normal, 2-3 = problematic
            if 'A12' in df.columns:
                df['A12'] = pd.to_numeric(df['A12'], errors='coerce')
                risk_count += ((df['A12'] >= 2) & (df['A12'] <= 3)).astype(int)
            
            # A13: 1 = normal, 2-3 = problematic
            if 'A13' in df.columns:
                df['A13'] = pd.to_numeric(df['A13'], errors='coerce')
                risk_count += ((df['A13'] >= 2) & (df['A13'] <= 3)).astype(int)
            
            df['anatomical_risk_count'] = risk_count
            
            # Keep only rows where we have OHIP_total and risk_count
            df = df[['OHIP_total', 'anatomical_risk_count']].copy()
            df = df[df['OHIP_total'].notna() & df['anatomical_risk_count'].notna()]
            
            # Find optimal cut-off point
            optimal_cutoff, effect_size, cutoff_p_value = find_optimal_cutoff(
                df['anatomical_risk_count'].values,
                df['OHIP_total'].values,
                outcome_type='continuous'
            )
            
            if optimal_cutoff is None:
                optimal_cutoff = int(df['anatomical_risk_count'].median())
            
            # Create dose-response categories using optimal cut-off
            df['risk_category'] = pd.cut(df['anatomical_risk_count'], 
                                         bins=[-1, optimal_cutoff, 100], 
                                         labels=[f'0-{optimal_cutoff} issues', f'{optimal_cutoff+1}+ issues'])
            
            if len(df) >= 10:
                # Test for monotonic increase using Kruskal-Wallis (non-parametric)
                categories = df['risk_category'].cat.categories
                groups = [df[df['risk_category'] == cat]['OHIP_total'].values for cat in categories]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    stat, p_value = kruskal(*groups)
                    
                    # Test for trend (Jonckheere-Terpstra test approximation using Spearman)
                    # Create numeric risk score for trend test
                    df['risk_numeric'] = df['anatomical_risk_count']
                    trend_corr, trend_p = spearmanr(df['risk_numeric'], df['OHIP_total'])
                    
                    # Descriptive statistics by category
                    desc_stats = df.groupby('risk_category')['OHIP_total'].agg(['mean', 'std', 'count', 'median']).to_dict('index')
                    
                    # Create visualization: box plot + scatter plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Box plot by category
                    categories_present = [cat for cat in categories if cat in df['risk_category'].values]
                    data_to_plot = [df[df['risk_category'] == cat]['OHIP_total'].values for cat in categories_present]
                    ax1.boxplot(data_to_plot, labels=categories_present)
                    ax1.set_xlabel('Anatómiai kockázati kategória')
                    ax1.set_ylabel('OHIP_total (magasabb = rosszabb QoL)')
                    ax1.set_title(f'OHIP_total eloszlása kockázati kategóriák szerint\n(Kruskal-Wallis, p = {p_value:.4f})')
                    ax1.grid(True, alpha=0.3)
                    
                    # Scatter plot showing dose-response
                    scatter_data = df.groupby('anatomical_risk_count')['OHIP_total'].agg(['mean', 'std', 'count']).reset_index()
                    ax2.scatter(scatter_data['anatomical_risk_count'], scatter_data['mean'], 
                               s=scatter_data['count']*10, alpha=0.6, label='Átlag OHIP')
                    ax2.errorbar(scatter_data['anatomical_risk_count'], scatter_data['mean'],
                               yerr=scatter_data['std'], fmt='none', alpha=0.5, capsize=5)
                    
                    # Add trend line
                    z = np.polyfit(df['anatomical_risk_count'], df['OHIP_total'], 1)
                    p = np.poly1d(z)
                    x_trend = np.arange(df['anatomical_risk_count'].min(), df['anatomical_risk_count'].max() + 1)
                    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend line (r = {trend_corr:.3f})')
                    ax2.set_xlabel('Anatómiai kockázati pontszám (unfavourable features száma)')
                    ax2.set_ylabel('OHIP_total (magasabb = rosszabb QoL)')
                    ax2.set_title(f'Dose-response effect: OHIP_total vs kockázati pontszám\n(Spearman r = {trend_corr:.3f}, p = {trend_p:.4f})')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plot_img = plot_to_base64(fig)
                    plt.close(fig)
                    
                    # Calculate mean OHIP for each category
                    category_means = {cat: float(desc_stats[cat]['mean']) for cat in desc_stats.keys()}
                    
                    # Check if monotonic increase exists
                    category_order = sorted(df['risk_category'].cat.categories.tolist())
                    means_ordered = [category_means.get(cat, 0) for cat in category_order if cat in category_means]
                    is_monotonic = len(means_ordered) >= 2 and all(means_ordered[i] <= means_ordered[i+1] for i in range(len(means_ordered)-1))
                    
                    results['H8'] = {
                        'status': 'success',
                        'test_name': 'Kruskal-Wallis + Spearman trend test',
                        'kruskal_statistic': float(stat),
                        'kruskal_p_value': float(p_value),
                        'trend_correlation': float(trend_corr),
                        'trend_p_value': float(trend_p),
                        'n': len(df),
                        'optimal_cutoff': int(optimal_cutoff),
                        'cutoff_effect_size': float(effect_size) if effect_size is not None else None,
                        'cutoff_p_value': float(cutoff_p_value) if cutoff_p_value is not None else None,
                        'is_monotonic_increase': is_monotonic,
                        'category_means': category_means,
                        'descriptive_stats': desc_stats,
                        'plot_img': plot_img,
                        'data_needed': 'OHIP_total + anatomical features (F5, F7, A1_Kaan, A3-A9, A11-A13)',
                        'null_hypothesis': 'Baseline QoL (OHIP) does not worsen cumulatively with the number of anatomical risk factors.',
                        'alternative_hypothesis': 'Baseline QoL worsens cumulatively with the number of anatomical risk factors (dose-response effect).'
                    }
                else:
                    results['H8'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'OHIP_total + anatomical features', 'n': len(df)}
            else:
                results['H8'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'OHIP_total + anatomical features', 'n': len(df)}
        else:
            results['H8'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'OHIP_total + anatomical features'}
    except Exception as e:
        results['H8'] = {'status': 'error', 'message': str(e), 'data_needed': 'OHIP_total + anatomical features'}
    
    # H9: Cumulative anatomical risk factors vs baseline GOHAI (dose-response effect)
    # Note: GOHAI is reversed - higher GOHAI = better QoL (unlike OHIP)
    try:
        # Use the same anatomical risk variables as H8
        required_vars = ['GOHAI_1', 'GOHAI_2', 'GOHAI_3', 'GOHAI_4', 'GOHAI_5', 'GOHAI_6', 
                        'GOHAI_7', 'GOHAI_8', 'GOHAI_9', 'GOHAI_10', 'GOHAI_11', 'GOHAI_12',
                        'F5', 'F7', 
                        'A1_Kaan',
                        'A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal',
                        'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal', 
                        'A9_jobb', 'A9_bal', 'A11', 'A12', 'A13']
        
        if check_data_availability(required_vars):
            data = get_data(required_vars)
            df = pd.DataFrame(data, columns=required_vars)
            
            # Calculate GOHAI_total (baseline)
            df['GOHAI_total'] = df[['GOHAI_1', 'GOHAI_2', 'GOHAI_3', 'GOHAI_4', 'GOHAI_5', 'GOHAI_6',
                                    'GOHAI_7', 'GOHAI_8', 'GOHAI_9', 'GOHAI_10', 'GOHAI_11', 'GOHAI_12']].sum(axis=1)
            
            # Initialize risk count (same calculation as H8)
            risk_count = pd.Series(0, index=df.index)
            
            # Upper jaw variables
            # F5: 1 = normal, 2-3 = problematic
            if 'F5' in df.columns:
                df['F5'] = pd.to_numeric(df['F5'], errors='coerce')
                risk_count += ((df['F5'] >= 2) & (df['F5'] <= 3)).astype(int)
            
            # F7: 1 = normal, 2-3 = problematic
            if 'F7' in df.columns:
                df['F7'] = pd.to_numeric(df['F7'], errors='coerce')
                risk_count += ((df['F7'] >= 2) & (df['F7'] <= 3)).astype(int)
            
            # Lower jaw variables
            # A1_Kaan: 1 = normal, 2-5 = problematic
            if 'A1_Kaan' in df.columns:
                df['A1_Kaan'] = pd.to_numeric(df['A1_Kaan'], errors='coerce')
                risk_count += ((df['A1_Kaan'] >= 2) & (df['A1_Kaan'] <= 5)).astype(int)
            
            # A3-A9: paired variables (jobb/bal)
            # Count as problematic if EITHER side is problematic
            a_pairs = [
                ('A3_jobb', 'A3_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A4_jobb', 'A4_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A5_jobb', 'A5_bal', 2, [1, 3]),  # 2 = normal, 1 or 3 = problematic (special case!)
                ('A6_jobb', 'A6_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A7_jobb', 'A7_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A8_jobb', 'A8_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A9_jobb', 'A9_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
            ]
            
            for left_var, right_var, normal_val, problematic_vals in a_pairs:
                if left_var in df.columns and right_var in df.columns:
                    df[left_var] = pd.to_numeric(df[left_var], errors='coerce')
                    df[right_var] = pd.to_numeric(df[right_var], errors='coerce')
                    # Count if either side is problematic
                    left_problematic = df[left_var].isin(problematic_vals)
                    right_problematic = df[right_var].isin(problematic_vals)
                    risk_count += (left_problematic | right_problematic).astype(int)
            
            # A11: 2 = normal, 1 or 3 = problematic (special case!)
            if 'A11' in df.columns:
                df['A11'] = pd.to_numeric(df['A11'], errors='coerce')
                risk_count += ((df['A11'] == 1) | (df['A11'] == 3)).astype(int)
            
            # A12: 1 = normal, 2-3 = problematic
            if 'A12' in df.columns:
                df['A12'] = pd.to_numeric(df['A12'], errors='coerce')
                risk_count += ((df['A12'] >= 2) & (df['A12'] <= 3)).astype(int)
            
            # A13: 1 = normal, 2-3 = problematic
            if 'A13' in df.columns:
                df['A13'] = pd.to_numeric(df['A13'], errors='coerce')
                risk_count += ((df['A13'] >= 2) & (df['A13'] <= 3)).astype(int)
            
            df['anatomical_risk_count'] = risk_count
            
            # Keep only rows where we have GOHAI_total and risk_count
            df = df[['GOHAI_total', 'anatomical_risk_count']].copy()
            df = df[df['GOHAI_total'].notna() & df['anatomical_risk_count'].notna()]
            
            # Find optimal cut-off point
            optimal_cutoff, effect_size, cutoff_p_value = find_optimal_cutoff(
                df['anatomical_risk_count'].values,
                df['GOHAI_total'].values,
                outcome_type='continuous'
            )
            
            if optimal_cutoff is None:
                optimal_cutoff = int(df['anatomical_risk_count'].median())
            
            # Create dose-response categories using optimal cut-off
            df['risk_category'] = pd.cut(df['anatomical_risk_count'], 
                                         bins=[-1, optimal_cutoff, 100], 
                                         labels=[f'0-{optimal_cutoff} issues', f'{optimal_cutoff+1}+ issues'])
            
            if len(df) >= 10:
                # Test for monotonic decrease using Kruskal-Wallis (non-parametric)
                # Note: For GOHAI, we expect DECREASE with more risk factors (opposite of OHIP)
                categories = df['risk_category'].cat.categories
                groups = [df[df['risk_category'] == cat]['GOHAI_total'].values for cat in categories]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    stat, p_value = kruskal(*groups)
                    
                    # Test for trend (Spearman correlation - negative expected for GOHAI)
                    df['risk_numeric'] = df['anatomical_risk_count']
                    trend_corr, trend_p = spearmanr(df['risk_numeric'], df['GOHAI_total'])
                    
                    # Descriptive statistics by category
                    desc_stats = df.groupby('risk_category')['GOHAI_total'].agg(['mean', 'std', 'count', 'median']).to_dict('index')
                    
                    # Create visualization: box plot + scatter plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Box plot by category
                    categories_present = [cat for cat in categories if cat in df['risk_category'].values]
                    data_to_plot = [df[df['risk_category'] == cat]['GOHAI_total'].values for cat in categories_present]
                    ax1.boxplot(data_to_plot, labels=categories_present)
                    ax1.set_xlabel('Anatómiai kockázati kategória')
                    ax1.set_ylabel('GOHAI_total (magasabb = jobb QoL)')
                    ax1.set_title(f'GOHAI_total eloszlása kockázati kategóriák szerint\n(Kruskal-Wallis, p = {p_value:.4f})')
                    ax1.grid(True, alpha=0.3)
                    
                    # Scatter plot showing dose-response
                    scatter_data = df.groupby('anatomical_risk_count')['GOHAI_total'].agg(['mean', 'std', 'count']).reset_index()
                    ax2.scatter(scatter_data['anatomical_risk_count'], scatter_data['mean'], 
                               s=scatter_data['count']*10, alpha=0.6, label='Átlag GOHAI')
                    ax2.errorbar(scatter_data['anatomical_risk_count'], scatter_data['mean'],
                               yerr=scatter_data['std'], fmt='none', alpha=0.5, capsize=5)
                    
                    # Add trend line
                    z = np.polyfit(df['anatomical_risk_count'], df['GOHAI_total'], 1)
                    p = np.poly1d(z)
                    x_trend = np.arange(df['anatomical_risk_count'].min(), df['anatomical_risk_count'].max() + 1)
                    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend line (r = {trend_corr:.3f})')
                    ax2.set_xlabel('Anatómiai kockázati pontszám (unfavourable features száma)')
                    ax2.set_ylabel('GOHAI_total (magasabb = jobb QoL)')
                    ax2.set_title(f'Dose-response effect: GOHAI_total vs kockázati pontszám\n(Spearman r = {trend_corr:.3f}, p = {trend_p:.4f})')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plot_img = plot_to_base64(fig)
                    plt.close(fig)
                    
                    # Calculate mean GOHAI for each category
                    category_means = {cat: float(desc_stats[cat]['mean']) for cat in desc_stats.keys()}
                    
                    # Check if monotonic decrease exists (opposite of OHIP)
                    category_order = sorted(df['risk_category'].cat.categories.tolist())
                    means_ordered = [category_means.get(cat, 0) for cat in category_order if cat in category_means]
                    is_monotonic_decrease = len(means_ordered) >= 2 and all(means_ordered[i] >= means_ordered[i+1] for i in range(len(means_ordered)-1))
                    
                    results['H9'] = {
                        'status': 'success',
                        'test_name': 'Kruskal-Wallis + Spearman trend test',
                        'kruskal_statistic': float(stat),
                        'kruskal_p_value': float(p_value),
                        'trend_correlation': float(trend_corr),
                        'trend_p_value': float(trend_p),
                        'n': len(df),
                        'optimal_cutoff': int(optimal_cutoff),
                        'cutoff_effect_size': float(effect_size) if effect_size is not None else None,
                        'cutoff_p_value': float(cutoff_p_value) if cutoff_p_value is not None else None,
                        'is_monotonic_decrease': is_monotonic_decrease,
                        'category_means': category_means,
                        'descriptive_stats': desc_stats,
                        'plot_img': plot_img,
                        'data_needed': 'GOHAI_total + anatomical features (F5, F7, A1_Kaan, A3-A9, A11-A13)',
                        'null_hypothesis': 'Baseline QoL (GOHAI) does not worsen cumulatively with the number of anatomical risk factors.',
                        'alternative_hypothesis': 'Baseline QoL worsens cumulatively with the number of anatomical risk factors (dose-response effect: more risk factors → lower GOHAI).'
                    }
                else:
                    results['H9'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'GOHAI_total + anatomical features', 'n': len(df)}
            else:
                results['H9'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'GOHAI_total + anatomical features', 'n': len(df)}
        else:
            results['H9'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'GOHAI_total + anatomical features'}
    except Exception as e:
        results['H9'] = {'status': 'error', 'message': str(e), 'data_needed': 'GOHAI_total + anatomical features'}
    
    # H10: Cumulative anatomical risk factors vs baseline responsiveness_today_situation (dose-response effect)
    # Note: responsiveness is categorical: Kiváló=5, Jó=4, Átlagos=3, Rossz=2, Nagyon rossz=1 (higher = better)
    try:
        # Use the same anatomical risk variables as H8/H9
        required_vars = ['responsiveness_today_situation',
                        'F5', 'F7', 
                        'A1_Kaan',
                        'A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal',
                        'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal', 
                        'A9_jobb', 'A9_bal', 'A11', 'A12', 'A13']
        
        if check_data_availability(required_vars):
            data = get_data(required_vars)
            df = pd.DataFrame(data, columns=required_vars)
            
            # Convert categorical responsiveness to numeric (higher = better)
            responsiveness_map = {
                'Kiváló': 5,
                'Jó': 4,
                'Átlagos': 3,
                'Rossz': 2,
                'Nagyon rossz': 1
            }
            df['responsiveness_numeric'] = df['responsiveness_today_situation'].map(responsiveness_map)
            
            # Initialize risk count (same calculation as H8/H9)
            risk_count = pd.Series(0, index=df.index)
            
            # Upper jaw variables
            # F5: 1 = normal, 2-3 = problematic
            if 'F5' in df.columns:
                df['F5'] = pd.to_numeric(df['F5'], errors='coerce')
                risk_count += ((df['F5'] >= 2) & (df['F5'] <= 3)).astype(int)
            
            # F7: 1 = normal, 2-3 = problematic
            if 'F7' in df.columns:
                df['F7'] = pd.to_numeric(df['F7'], errors='coerce')
                risk_count += ((df['F7'] >= 2) & (df['F7'] <= 3)).astype(int)
            
            # Lower jaw variables
            # A1_Kaan: 1 = normal, 2-5 = problematic
            if 'A1_Kaan' in df.columns:
                df['A1_Kaan'] = pd.to_numeric(df['A1_Kaan'], errors='coerce')
                risk_count += ((df['A1_Kaan'] >= 2) & (df['A1_Kaan'] <= 5)).astype(int)
            
            # A3-A9: paired variables (jobb/bal)
            # Count as problematic if EITHER side is problematic
            a_pairs = [
                ('A3_jobb', 'A3_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A4_jobb', 'A4_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A5_jobb', 'A5_bal', 2, [1, 3]),  # 2 = normal, 1 or 3 = problematic (special case!)
                ('A6_jobb', 'A6_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A7_jobb', 'A7_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A8_jobb', 'A8_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
                ('A9_jobb', 'A9_bal', 1, [2, 3]),  # 1 = normal, 2-3 = problematic
            ]
            
            for left_var, right_var, normal_val, problematic_vals in a_pairs:
                if left_var in df.columns and right_var in df.columns:
                    df[left_var] = pd.to_numeric(df[left_var], errors='coerce')
                    df[right_var] = pd.to_numeric(df[right_var], errors='coerce')
                    # Count if either side is problematic
                    left_problematic = df[left_var].isin(problematic_vals)
                    right_problematic = df[right_var].isin(problematic_vals)
                    risk_count += (left_problematic | right_problematic).astype(int)
            
            # A11: 2 = normal, 1 or 3 = problematic (special case!)
            if 'A11' in df.columns:
                df['A11'] = pd.to_numeric(df['A11'], errors='coerce')
                risk_count += ((df['A11'] == 1) | (df['A11'] == 3)).astype(int)
            
            # A12: 1 = normal, 2-3 = problematic
            if 'A12' in df.columns:
                df['A12'] = pd.to_numeric(df['A12'], errors='coerce')
                risk_count += ((df['A12'] >= 2) & (df['A12'] <= 3)).astype(int)
            
            # A13: 1 = normal, 2-3 = problematic
            if 'A13' in df.columns:
                df['A13'] = pd.to_numeric(df['A13'], errors='coerce')
                risk_count += ((df['A13'] >= 2) & (df['A13'] <= 3)).astype(int)
            
            df['anatomical_risk_count'] = risk_count
            
            # Keep only rows where we have responsiveness and risk_count
            df = df[['responsiveness_numeric', 'responsiveness_today_situation', 'anatomical_risk_count']].copy()
            df = df[df['responsiveness_numeric'].notna() & df['anatomical_risk_count'].notna()]
            
            # Find optimal cut-off point
            optimal_cutoff, effect_size, cutoff_p_value = find_optimal_cutoff(
                df['anatomical_risk_count'].values,
                df['responsiveness_numeric'].values,
                outcome_type='categorical'
            )
            
            if optimal_cutoff is None:
                optimal_cutoff = int(df['anatomical_risk_count'].median())
            
            # Create dose-response categories using optimal cut-off
            df['risk_category'] = pd.cut(df['anatomical_risk_count'], 
                                         bins=[-1, optimal_cutoff, 100], 
                                         labels=[f'0-{optimal_cutoff} issues', f'{optimal_cutoff+1}+ issues'])
            
            if len(df) >= 10:
                # Test for monotonic decrease using Kruskal-Wallis (non-parametric)
                # Note: For responsiveness, we expect DECREASE with more risk factors (more risk → worse responsiveness)
                categories = df['risk_category'].cat.categories
                groups = [df[df['risk_category'] == cat]['responsiveness_numeric'].values for cat in categories]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    stat, p_value = kruskal(*groups)
                    
                    # Test for trend (Spearman correlation - negative expected)
                    df['risk_numeric'] = df['anatomical_risk_count']
                    trend_corr, trend_p = spearmanr(df['risk_numeric'], df['responsiveness_numeric'])
                    
                    # Descriptive statistics by category
                    desc_stats = df.groupby('risk_category')['responsiveness_numeric'].agg(['mean', 'std', 'count', 'median']).to_dict('index')
                    
                    # Create visualization: box plot + scatter plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Box plot by category
                    categories_present = [cat for cat in categories if cat in df['risk_category'].values]
                    data_to_plot = [df[df['risk_category'] == cat]['responsiveness_numeric'].values for cat in categories_present]
                    ax1.boxplot(data_to_plot, labels=categories_present)
                    ax1.set_xlabel('Anatómiai kockázati kategória')
                    ax1.set_ylabel('Responsiveness pontszám (magasabb = jobb)')
                    ax1.set_title(f'Responsiveness eloszlása kockázati kategóriák szerint\n(Kruskal-Wallis, p = {p_value:.4f})')
                    ax1.set_yticks([1, 2, 3, 4, 5])
                    ax1.set_yticklabels(['Nagyon rossz', 'Rossz', 'Átlagos', 'Jó', 'Kiváló'])
                    ax1.grid(True, alpha=0.3)
                    
                    # Scatter plot showing dose-response
                    scatter_data = df.groupby('anatomical_risk_count')['responsiveness_numeric'].agg(['mean', 'std', 'count']).reset_index()
                    ax2.scatter(scatter_data['anatomical_risk_count'], scatter_data['mean'], 
                               s=scatter_data['count']*10, alpha=0.6, label='Átlag responsiveness')
                    ax2.errorbar(scatter_data['anatomical_risk_count'], scatter_data['mean'],
                               yerr=scatter_data['std'], fmt='none', alpha=0.5, capsize=5)
                    
                    # Add trend line
                    z = np.polyfit(df['anatomical_risk_count'], df['responsiveness_numeric'], 1)
                    p = np.poly1d(z)
                    x_trend = np.arange(df['anatomical_risk_count'].min(), df['anatomical_risk_count'].max() + 1)
                    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend line (r = {trend_corr:.3f})')
                    ax2.set_xlabel('Anatómiai kockázati pontszám (unfavourable features száma)')
                    ax2.set_ylabel('Responsiveness pontszám (magasabb = jobb)')
                    ax2.set_title(f'Dose-response effect: Responsiveness vs kockázati pontszám\n(Spearman r = {trend_corr:.3f}, p = {trend_p:.4f})')
                    ax2.set_yticks([1, 2, 3, 4, 5])
                    ax2.set_yticklabels(['Nagyon rossz', 'Rossz', 'Átlagos', 'Jó', 'Kiváló'])
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plot_img = plot_to_base64(fig)
                    plt.close(fig)
                    
                    # Calculate mean responsiveness for each category
                    category_means = {cat: float(desc_stats[cat]['mean']) for cat in desc_stats.keys()}
                    
                    # Check if monotonic decrease exists (more risk → lower responsiveness)
                    category_order = sorted(df['risk_category'].cat.categories.tolist())
                    means_ordered = [category_means.get(cat, 0) for cat in category_order if cat in category_means]
                    is_monotonic_decrease = len(means_ordered) >= 2 and all(means_ordered[i] >= means_ordered[i+1] for i in range(len(means_ordered)-1))
                    
                    results['H10'] = {
                        'status': 'success',
                        'test_name': 'Kruskal-Wallis + Spearman trend test',
                        'kruskal_statistic': float(stat),
                        'kruskal_p_value': float(p_value),
                        'trend_correlation': float(trend_corr),
                        'trend_p_value': float(trend_p),
                        'n': len(df),
                        'optimal_cutoff': int(optimal_cutoff),
                        'cutoff_effect_size': float(effect_size) if effect_size is not None else None,
                        'cutoff_p_value': float(cutoff_p_value) if cutoff_p_value is not None else None,
                        'is_monotonic_decrease': is_monotonic_decrease,
                        'category_means': category_means,
                        'descriptive_stats': desc_stats,
                        'plot_img': plot_img,
                        'data_needed': 'responsiveness_today_situation + anatomical features (F5, F7, A1_Kaan, A3-A9, A11-A13)',
                        'null_hypothesis': 'Baseline responsiveness does not worsen cumulatively with the number of anatomical risk factors.',
                        'alternative_hypothesis': 'Baseline responsiveness worsens cumulatively with the number of anatomical risk factors (dose-response effect: more risk factors → lower responsiveness).'
                    }
                else:
                    results['H10'] = {'status': 'insufficient_data', 'message': 'Not enough groups with data', 'data_needed': 'responsiveness_today_situation + anatomical features', 'n': len(df)}
            else:
                results['H10'] = {'status': 'insufficient_data', 'message': f'Only {len(df)} data points available (need at least 10)', 'data_needed': 'responsiveness_today_situation + anatomical features', 'n': len(df)}
        else:
            results['H10'] = {'status': 'insufficient_data', 'message': 'Missing required data fields', 'data_needed': 'responsiveness_today_situation + anatomical features'}
    except Exception as e:
        results['H10'] = {'status': 'error', 'message': str(e), 'data_needed': 'responsiveness_today_situation + anatomical features'}
    
    return results


@app.route('/results')
def results():
    cursor = get_db_cursor()
    cursor.execute('SELECT COUNT(*) FROM patients WHERE "TAJ" IS NOT NULL')
    patient_count = cursor.fetchone()[0]
    
    # Count dropouts (where dropout = 1 or TRUE)
    cursor.execute('SELECT COUNT(*) FROM patients WHERE "dropout" = TRUE')
    dropout_count = cursor.fetchone()[0]
    
    cursor.execute("""SELECT COUNT(*) FROM patients WHERE ("denture_type" = 'lower' OR "denture_type" = 'both') AND "denture_type" IS NOT NULL""")
    lower_denture_count = cursor.fetchone()[0]
    
    cursor.execute("""SELECT COUNT(*) FROM patients WHERE ("denture_type" = 'upper' OR "denture_type" = 'both') AND "denture_type" IS NOT NULL""")
    upper_denture_count = cursor.fetchone()[0]
    # Denture Type Chart
    fig_dentures = plt.figure(figsize=(8, 6))
    labels = ['Alsó fogsor', 'Felső fogsor']
    values = [lower_denture_count, upper_denture_count]
    plt.bar(labels, values, color=['#4CAF50', '#FFC107'])
    plt.title('Teljes lemezes fogpótlások száma állcsontonként')
    plt.ylabel('Szám')
    # Make sure that the y_axis has only integers as ticks
    plt.yticks(np.arange(0, max(values) + 1, step=1))
    dentures_img = plot_to_base64(fig_dentures)
    
    # Age and gender distribution logic
    cursor.execute('SELECT "gender", "birthdate" FROM patients WHERE "gender" IS NOT NULL AND "birthdate" IS NOT NULL')
    patients = cursor.fetchall()
    male_age_distribution = [calculate_age(row[1]) for row in patients if row[0] == 'Male']
    female_age_distribution = [calculate_age(row[1]) for row in patients if row[0] == 'Female']

    # Plotting
    fig_age_gender = plt.figure(figsize=(10, 6))
    age_bins = range(0, 101, 5)  # age groups
    male_age_hist, _ = np.histogram(male_age_distribution, bins=age_bins)
    female_age_hist, _ = np.histogram(female_age_distribution, bins=age_bins)

    # Create the pyramid
    y = np.arange(len(age_bins) - 1)
    plt.barh(y, -male_age_hist, align='center', color='#2196F3', label='Férfiak')  # Negative values for males
    plt.barh(y, female_age_hist, align='center', color='#E91E63', label='Nők')
    plt.xlabel('Szám')
    plt.ylabel('Kor csoportok')
    plt.title('Kor és nem szerinti megoszlás')
    plt.yticks(y, [f'{age_bins[i]}-{age_bins[i + 1] - 1}' for i in range(len(age_bins) - 1)])

    # Customize x-ticks
    max_hist = max(male_age_hist.max(), female_age_hist.max())
    x_ticks = np.arange(-max_hist, max_hist + 1, step=1)
    plt.xticks(x_ticks, [str(abs(x)) for x in x_ticks])

    plt.legend(loc='upper right')
    plt.grid(axis='x')
    age_gender_img = plot_to_base64(fig_age_gender)

    # Q1 subjective chewing ability
    cursor.execute('SELECT "chewing_today_situation" FROM patients WHERE "chewing_today_situation" IS NOT NULL')
    subjective_chewing = cursor.fetchall()
    subjective_chewing = [row[0] for row in subjective_chewing]
        # Convert the list to a pandas DataFrame
    df = pd.DataFrame(subjective_chewing, columns=['today_situation'])
        # Count the occurrences of each response
    response_counts = df['today_situation'].value_counts().reindex(["Kiváló", "Jó", "Átlagos", "Rossz", "Nagyon rossz"], fill_value=0)
        # Plotting the bar chart
    fig_q1 = plt.figure(figsize=(8, 4))
    response_counts.plot(kind='bar', color='skyblue')
    plt.xlabel(None)
    plt.ylabel('A válaszolók száma')
    plt.title('Szubjektív rágóképességre vonatkozó kérdésre adott válaszok megoszlása')
    plt.xticks(rotation=0)
    # Make sure the y-axis has integers as ticks
    plt.yticks(np.arange(0, max(response_counts) + 1, step=1))
    q1_barchart = plot_to_base64(fig_q1)

    # Q2 subjective CHANGE IN chewing ability
    cursor.execute('SELECT "chewing_change" FROM patients WHERE "chewing_change" IS NOT NULL')
    subjective_chewing_change = cursor.fetchall()
    subjective_chewing_change = [row[0] for row in subjective_chewing_change]
        # Convert the list to a pandas DataFrame
    df_subjective_chewing_change = pd.DataFrame(subjective_chewing_change, columns=['subjective_chewing_change'])
        # Count the occurrences of each response
    response_counts_subjective_chewing_change = df_subjective_chewing_change['subjective_chewing_change'].value_counts().reindex(["Sokat romlott", "Kicsit romlott", "Változatlan maradt", "Kicsit javult", "Sokat javult"], fill_value=0)
        # Plotting the bar chart
    fig_q2 = plt.figure(figsize=(8, 4))
    response_counts_subjective_chewing_change.plot(kind='bar', color='skyblue')
    plt.xlabel(None)
    plt.ylabel('A válaszolók száma')
    plt.title('Szubjektív rágóképességVÁLTOZÁSra vonatkozó kérdésre adott válaszok megoszlása')
    plt.xticks(rotation=0)
    plt.yticks(np.arange(0, max(response_counts_subjective_chewing_change) + 1, step=1))
    q2_barchart = plot_to_base64(fig_q2)
    
    # Initial OHIP and GOHAI calculations
    cursor.execute('SELECT "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5" FROM patients WHERE "OHIP_1" IS NOT NULL AND "OHIP_2" IS NOT NULL AND "OHIP_3" IS NOT NULL AND "OHIP_4" IS NOT NULL AND "OHIP_5" IS NOT NULL')
    initial_ohip_scores = cursor.fetchall()
    ohip_init_scores = [sum(row) for row in initial_ohip_scores]
    ohip_init_mean = np.mean(ohip_init_scores)
    ohip_init_std = np.std(ohip_init_scores)
    
    cursor.execute('SELECT "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12" FROM patients WHERE "GOHAI_1" IS NOT NULL AND "GOHAI_2" IS NOT NULL AND "GOHAI_3" IS NOT NULL AND "GOHAI_4" IS NOT NULL AND "GOHAI_5" IS NOT NULL AND "GOHAI_6" IS NOT NULL AND "GOHAI_7" IS NOT NULL AND "GOHAI_8" IS NOT NULL AND "GOHAI_9" IS NOT NULL AND "GOHAI_10" IS NOT NULL AND "GOHAI_11" IS NOT NULL AND "GOHAI_12" IS NOT NULL')
    gohai_scores = cursor.fetchall()
    gohai_init_scores = [sum(row) for row in gohai_scores]
    gohai_init_mean = np.mean(gohai_init_scores)
    gohai_init_std = np.std(gohai_init_scores)

    # Final OHIP and GOHAI calculations
    cursor.execute('SELECT "OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall" FROM patients WHERE "OHIP_1_recall" IS NOT NULL AND "OHIP_2_recall" IS NOT NULL AND "OHIP_3_recall" IS NOT NULL AND "OHIP_4_recall" IS NOT NULL AND "OHIP_5_recall" IS NOT NULL')
    final_ohip_scores = cursor.fetchall()
    ohip_final_scores = [sum(row) for row in final_ohip_scores]
    ohip_final_mean = np.mean(ohip_final_scores)
    ohip_final_std = np.std(ohip_final_scores)
    
    cursor.execute('SELECT "GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall" FROM patients WHERE "GOHAI_1_recall" IS NOT NULL AND "GOHAI_2_recall" IS NOT NULL AND "GOHAI_3_recall" IS NOT NULL AND "GOHAI_4_recall" IS NOT NULL AND "GOHAI_5_recall" IS NOT NULL AND "GOHAI_6_recall" IS NOT NULL AND "GOHAI_7_recall" IS NOT NULL AND "GOHAI_8_recall" IS NOT NULL AND "GOHAI_9_recall" IS NOT NULL AND "GOHAI_10_recall" IS NOT NULL AND "GOHAI_11_recall" IS NOT NULL AND "GOHAI_12_recall" IS NOT NULL')
    final_gohai_scores = cursor.fetchall()
    gohai_final_scores = [sum(row) for row in final_gohai_scores]
    gohai_final_mean = np.mean(gohai_final_scores)
    gohai_final_std = np.std(gohai_final_scores)

    # Initial MAI calculations (hue degree method)
    cursor.execute('SELECT "init_mai_huedegree" FROM patients WHERE "init_mai_huedegree" IS NOT NULL')
    init_scores = cursor.fetchall()
    init_mai_scores = [row[0] for row in init_scores]
    init_mai_mean = np.mean(init_mai_scores)
    init_mai_std = np.std(init_mai_scores)
    
    # Final MAI calculations (hue degree method)
    cursor.execute('SELECT "final_mai_huedegree" FROM patients WHERE "final_mai_huedegree" IS NOT NULL')
    final_scores = cursor.fetchall()
    final_mai_scores = [row[0] for row in final_scores]
    final_mai_mean = np.mean(final_mai_scores)
    final_mai_std = np.std(final_mai_scores)

    def create_blank_plot(message="Nincs elegendő adat – várjuk a final_mai_huedegree adatokat"):
        fig = plt.figure(figsize=(6, 1))
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, wrap=True)
        plt.axis('off')
        return plot_to_base64(fig)

    # Filter valid data for ROC analysis
    optimal_threshold_mai = None
    cursor.execute('SELECT "TAJ", "init_mai_huedegree", "final_mai_huedegree", "chewing_change" FROM patients WHERE "init_mai_huedegree" IS NOT NULL AND "final_mai_huedegree" IS NOT NULL AND "chewing_change" IS NOT NULL')
    roc_data = cursor.fetchall()
    if len(roc_data) > 0:
        roc_df = pd.DataFrame(roc_data, columns=["TAJ", "init_mai_huedegree", "final_mai_huedegree", "perceived_change"])
        mai_score_difference = roc_df["final_mai_huedegree"] - roc_df["init_mai_huedegree"]
        reported_improvement = roc_df["perceived_change"].apply(lambda x: 1 if x in ['Kicsit javult', 'Sokat javult'] else 0)

        if len(reported_improvement.unique()) > 1:
            fpr, tpr, thresholds = roc_curve(reported_improvement, mai_score_difference)
            roc_auc = roc_auc_score(reported_improvement, mai_score_difference)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold_mai = thresholds[optimal_idx]

            # Plot ROC curve for MAI
            fig_roc_mai = plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Fals pozitívok aránya')
            plt.ylabel('Valódi pozitívok aránya')
            plt.title('Receiver Operating Characteristic (ROC) görbe (MAI hue degree)')
            plt.legend(loc="lower right")
            roc_img_mai = plot_to_base64(fig_roc_mai)

            # Plot Score Difference vs Reported Improvement for MAI
            fig_diff_mai = plt.figure(figsize=(10, 6))
            plt.scatter(mai_score_difference, reported_improvement, alpha=0.5, label='résztvevők')
            plt.axvline(x=optimal_threshold_mai, color='r', linestyle='--', label=f'Az optimális vágópont: {optimal_threshold_mai:.2f}')
            plt.title('Rágóképesség pontkülönbség és a szubjektív javulás (MAI hue degree)')
            plt.xlabel('ΔMAI_huedegree')
            plt.ylabel('Tapasztalt-e változást a \nrágóképességének tekintetében? \n(1 = igen, 0 = nem)')
            plt.legend()
            diff_img_mai = plot_to_base64(fig_diff_mai)
        else:
            roc_img_mai = create_blank_plot("Nem áll rendelkezésre elegendő eltérő szubjektív válasz – várjuk a final_mai_huedegree adatokat")
            diff_img_mai = create_blank_plot("Nincs elegendő adat az összehasonlításhoz – final_mai_huedegree hiányzik")
    else:
        roc_img_mai = create_blank_plot("Nincs elegendő adat – várjuk a final_mai_huedegree adatokat")
        diff_img_mai = create_blank_plot("Nincs elegendő adat – várjuk a final_mai_huedegree adatokat")

    # ROC Analysis for OHIP
    optimal_threshold_ohip = None
    cursor.execute('SELECT "TAJ", "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5", "OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall", "chewing_change" FROM patients WHERE "OHIP_1" IS NOT NULL AND "OHIP_2" IS NOT NULL AND "OHIP_3" IS NOT NULL AND "OHIP_4" IS NOT NULL AND "OHIP_5" IS NOT NULL AND "OHIP_1_recall" IS NOT NULL AND "OHIP_2_recall" IS NOT NULL AND "OHIP_3_recall" IS NOT NULL AND "OHIP_4_recall" IS NOT NULL AND "OHIP_5_recall" IS NOT NULL AND "chewing_change" IS NOT NULL')
    ohip_data = cursor.fetchall()
    if len(ohip_data) > 0:
        ohip_roc_df = pd.DataFrame(ohip_data, columns=["TAJ", "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5", "OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall", "perceived_change"])
        ohip_init_roc = ohip_roc_df[["OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5"]].sum(axis=1)
        ohip_final_roc = ohip_roc_df[["OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall"]].sum(axis=1)
        ohip_score_difference = ohip_final_roc - ohip_init_roc
        reported_improvement_ohip = ohip_roc_df["perceived_change"].apply(lambda x: 1 if x in ['Kicsit javult', 'Sokat javult'] else 0)

        if len(reported_improvement_ohip.unique()) > 1:
            fpr_o, tpr_o, thresholds_o = roc_curve(reported_improvement_ohip, ohip_score_difference)
            roc_auc_o = roc_auc_score(reported_improvement_ohip, ohip_score_difference)
            optimal_idx_o = np.argmax(tpr_o - fpr_o)
            optimal_threshold_ohip = thresholds_o[optimal_idx_o]

            fig_roc_ohip = plt.figure(figsize=(8, 6))
            plt.plot(fpr_o, tpr_o, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_o:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(fpr_o[optimal_idx_o], tpr_o[optimal_idx_o], marker='o', color='red', label='Optimal Threshold')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Fals pozitívok aránya')
            plt.ylabel('Valódi pozitívok aránya')
            plt.title('Receiver Operating Characteristic (ROC) görbe (OHIP)')
            plt.legend(loc="lower right")
            roc_img_ohip = plot_to_base64(fig_roc_ohip)

            fig_diff_ohip = plt.figure(figsize=(10, 6))
            plt.scatter(ohip_score_difference, reported_improvement_ohip, alpha=0.5, label='résztvevők')
            plt.axvline(x=optimal_threshold_ohip, color='r', linestyle='--', label=f'Az optimális vágópont: {optimal_threshold_ohip:.2f}')
            plt.title('OHIP pontkülönbség és a szubjektív javulás (OHIP)')
            plt.xlabel('ΔOHIP')
            plt.ylabel('Tapasztalt-e változást a \nrágóképességének tekintetében? \n(1 = igen, 0 = nem)')
            plt.legend()
            diff_img_ohip = plot_to_base64(fig_diff_ohip)
        else:
            roc_img_ohip = create_blank_plot("Nem áll rendelkezésre elegendő eltérő szubjektív válasz – várjuk az OHIP visszamérési adatokat")
            diff_img_ohip = create_blank_plot("Nincs elegendő adat az összehasonlításhoz – OHIP visszamérés hiányzik")
    else:
        roc_img_ohip = create_blank_plot("Nincs elegendő adat – várjuk az OHIP visszamérési adatokat")
        diff_img_ohip = create_blank_plot("Nincs elegendő adat – várjuk az OHIP visszamérési adatokat")

    # ROC Analysis for GOHAI
    optimal_threshold_gohai = None
    cursor.execute('SELECT "TAJ", "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12", "GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall", "chewing_change" FROM patients WHERE "GOHAI_1" IS NOT NULL AND "GOHAI_2" IS NOT NULL AND "GOHAI_3" IS NOT NULL AND "GOHAI_4" IS NOT NULL AND "GOHAI_5" IS NOT NULL AND "GOHAI_6" IS NOT NULL AND "GOHAI_7" IS NOT NULL AND "GOHAI_8" IS NOT NULL AND "GOHAI_9" IS NOT NULL AND "GOHAI_10" IS NOT NULL AND "GOHAI_11" IS NOT NULL AND "GOHAI_12" IS NOT NULL AND "GOHAI_1_recall" IS NOT NULL AND "GOHAI_2_recall" IS NOT NULL AND "GOHAI_3_recall" IS NOT NULL AND "GOHAI_4_recall" IS NOT NULL AND "GOHAI_5_recall" IS NOT NULL AND "GOHAI_6_recall" IS NOT NULL AND "GOHAI_7_recall" IS NOT NULL AND "GOHAI_8_recall" IS NOT NULL AND "GOHAI_9_recall" IS NOT NULL AND "GOHAI_10_recall" IS NOT NULL AND "GOHAI_11_recall" IS NOT NULL AND "GOHAI_12_recall" IS NOT NULL AND "chewing_change" IS NOT NULL')
    gohai_data = cursor.fetchall()
    if len(gohai_data) > 0:
        gohai_roc_df = pd.DataFrame(gohai_data, columns=["TAJ", "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12", "GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall", "perceived_change"])
        gohai_init_roc = gohai_roc_df[["GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12"]].sum(axis=1)
        gohai_final_roc = gohai_roc_df[["GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall"]].sum(axis=1)
        gohai_score_difference = gohai_final_roc - gohai_init_roc
        reported_improvement_gohai = gohai_roc_df["perceived_change"].apply(lambda x: 1 if x in ['Kicsit javult', 'Sokat javult'] else 0)

        if len(reported_improvement_gohai.unique()) > 1:
            fpr_g, tpr_g, thresholds_g = roc_curve(reported_improvement_gohai, gohai_score_difference)
            roc_auc_g = roc_auc_score(reported_improvement_gohai, gohai_score_difference)
            optimal_idx_g = np.argmax(tpr_g - fpr_g)
            optimal_threshold_gohai = thresholds_g[optimal_idx_g]

            fig_roc_gohai = plt.figure(figsize=(8, 6))
            plt.plot(fpr_g, tpr_g, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_g:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(fpr_g[optimal_idx_g], tpr_g[optimal_idx_g], marker='o', color='red', label='Optimal Threshold')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Fals pozitívok aránya')
            plt.ylabel('Valódi pozitívok aránya')
            plt.title('Receiver Operating Characteristic (ROC) görbe (GOHAI)')
            plt.legend(loc="lower right")
            roc_img_gohai = plot_to_base64(fig_roc_gohai)

            fig_diff_gohai = plt.figure(figsize=(10, 6))
            plt.scatter(gohai_score_difference, reported_improvement_gohai, alpha=0.5, label='résztvevők')
            plt.axvline(x=optimal_threshold_gohai, color='r', linestyle='--', label=f'Az optimális vágópont: {optimal_threshold_gohai:.2f}')
            plt.title('GOHAI pontkülönbség és a szubjektív javulás (GOHAI)')
            plt.xlabel('ΔGOHAI')
            plt.ylabel('Tapasztalt-e változást a \nrágóképességének tekintetében? \n(1 = igen, 0 = nem)')
            plt.legend()
            diff_img_gohai = plot_to_base64(fig_diff_gohai)
        else:
            roc_img_gohai = create_blank_plot("Nem áll rendelkezésre elegendő eltérő szubjektív válasz – várjuk a GOHAI visszamérési adatokat")
            diff_img_gohai = create_blank_plot("Nincs elegendő adat az összehasonlításhoz – GOHAI visszamérés hiányzik")
    else:
        roc_img_gohai = create_blank_plot("Nincs elegendő adat – várjuk a GOHAI visszamérési adatokat")
        diff_img_gohai = create_blank_plot("Nincs elegendő adat – várjuk a GOHAI visszamérési adatokat")

    # Binned odds ratios for low-sample settings (grouped anatomical predictors)
    def _or_from_2x2(a, b, c, d):
        correction = 0.5 if min(a, b, c, d) == 0 else 0.0
        a_c, b_c, c_c, d_c = a + correction, b + correction, c + correction, d + correction
        return (a_c * d_c) / (b_c * c_c)

    def _bootstrap_or_ci(temp_df, exposure_col, outcome_col, n_boot=3000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(temp_df)
        boot_vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            sample = temp_df.iloc[idx]
            a = int(((sample[exposure_col] == 1) & (sample[outcome_col] == 1)).sum())
            b = int(((sample[exposure_col] == 1) & (sample[outcome_col] == 0)).sum())
            c = int(((sample[exposure_col] == 0) & (sample[outcome_col] == 1)).sum())
            d = int(((sample[exposure_col] == 0) & (sample[outcome_col] == 0)).sum())
            boot_vals.append(_or_from_2x2(a, b, c, d))
        return np.percentile(boot_vals, [2.5, 97.5])

    def compute_or_row(df, exposure_col, outcome_col, label):
        temp = df[[exposure_col, outcome_col]].dropna().copy()
        if len(temp) < 8:
            return None
        temp[exposure_col] = temp[exposure_col].astype(int)
        temp[outcome_col] = temp[outcome_col].astype(int)
        if temp[exposure_col].nunique() < 2 or temp[outcome_col].nunique() < 2:
            return None
        a = int(((temp[exposure_col] == 1) & (temp[outcome_col] == 1)).sum())
        b = int(((temp[exposure_col] == 1) & (temp[outcome_col] == 0)).sum())
        c = int(((temp[exposure_col] == 0) & (temp[outcome_col] == 1)).sum())
        d = int(((temp[exposure_col] == 0) & (temp[outcome_col] == 0)).sum())
        odds_ratio = _or_from_2x2(a, b, c, d)
        
        # Default (analytic) CI
        correction = 0.5 if min(a, b, c, d) == 0 else 0.0
        a_c, b_c, c_c, d_c = a + correction, b + correction, c + correction, d + correction
        se = np.sqrt((1 / a_c) + (1 / b_c) + (1 / c_c) + (1 / d_c))
        ci_low_analytic = np.exp(np.log(odds_ratio) - 1.96 * se)
        ci_high_analytic = np.exp(np.log(odds_ratio) + 1.96 * se)

        # For low-sample or sparse tables prefer bootstrap CI
        use_bootstrap = (len(temp) < 30) or (min(a, b, c, d) < 5)
        if use_bootstrap:
            ci_low, ci_high = _bootstrap_or_ci(temp, exposure_col, outcome_col)
            ci_method = 'bootstrap'
        else:
            ci_low, ci_high = ci_low_analytic, ci_high_analytic
            ci_method = 'analytic'
        return {
            'predictor': label,
            'n': len(temp),
            'events_exposed': a,
            'total_exposed': a + b,
            'events_unexposed': c,
            'total_unexposed': c + d,
            'odds_ratio': float(odds_ratio),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'ci_method': ci_method,
        }

    def build_binned_or_table(df, outcome_col):
        rows = []
        # Risk bins: low (0-6), medium (7-11), high (12+)
        risk_bin = pd.cut(
            df['anatomical_risk_score'],
            bins=[-0.1, 6, 11, np.inf],
            labels=['0-6', '7-11', '12+']
        )
        df_local = df.copy()
        df_local['risk_bin'] = risk_bin.astype(str)
        for comp, label in [('7-11', 'Kumulativ kockazat: 7-11 vs 0-6'),
                            ('12+', 'Kumulativ kockazat: 12+ vs 0-6')]:
            subset = df_local[df_local['risk_bin'].isin(['0-6', comp])].copy()
            subset['exposure_tmp'] = (subset['risk_bin'] == comp).astype(int)
            row = compute_or_row(subset, 'exposure_tmp', outcome_col, label)
            if row:
                rows.append(row)

        # F-group aggregation
        row = compute_or_row(df_local, 'f_unfavorable_any', outcome_col, 'F csoport: legalabb 1 kedvezotlen (F5/F7/F8)')
        if row:
            rows.append(row)
        row = compute_or_row(df_local, 'f_unfavorable_heavy', outcome_col, 'F csoport: legalabb 2 kedvezotlen (F5/F7/F8)')
        if row:
            rows.append(row)

        # A-group aggregation
        row = compute_or_row(df_local, 'a_unfavorable_heavy', outcome_col, 'A csoport: median feletti kedvezotlen terheles')
        if row:
            rows.append(row)
        return rows

    anatomy_cols = ['F5', 'F7', 'F8', 'A1_Kaan',
                    'A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal',
                    'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal',
                    'A9_jobb', 'A9_bal', 'A11', 'A12', 'A13', 'A14']
    a_cols = ['A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal',
              'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal',
              'A9_jobb', 'A9_bal', 'A11', 'A12', 'A13']
    cols_sql = ', '.join([f'"{c}"' for c in anatomy_cols])
    cursor.execute(f'''
        WITH latest AS (
            SELECT DISTINCT ON ("TAJ")
                "TAJ", "id", {cols_sql},
                "init_mai_huedegree", "final_mai_huedegree",
                "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5",
                "OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall",
                "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12",
                "GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall"
            FROM patients
            ORDER BY "TAJ", "id" DESC
        )
        SELECT * FROM latest
    ''')
    or_rows = cursor.fetchall()
    or_columns = (
        ['TAJ', 'id'] + anatomy_cols +
        ['init_mai_huedegree', 'final_mai_huedegree'] +
        [f'OHIP_{i}' for i in range(1, 6)] + [f'OHIP_{i}_recall' for i in range(1, 6)] +
        [f'GOHAI_{i}' for i in range(1, 13)] + [f'GOHAI_{i}_recall' for i in range(1, 13)]
    )
    or_df = pd.DataFrame(or_rows, columns=or_columns)
    for col in anatomy_cols + ['init_mai_huedegree', 'final_mai_huedegree'] + \
               [f'OHIP_{i}' for i in range(1, 6)] + [f'OHIP_{i}_recall' for i in range(1, 6)] + \
               [f'GOHAI_{i}' for i in range(1, 13)] + [f'GOHAI_{i}_recall' for i in range(1, 13)]:
        or_df[col] = pd.to_numeric(or_df[col], errors='coerce')

    # Keep only complete anatomy rows for stable risk computation
    or_df = or_df.dropna(subset=anatomy_cols).copy()

    # Anatomical risk score (same principle as H2c)
    risk_score = np.zeros(len(or_df), dtype=int)
    for var in ['F5', 'F7', 'F8']:
        risk_score += (or_df[var] >= 2).astype(int)
    for var in a_cols:
        threshold = or_df[var].median()
        risk_score += (or_df[var] > threshold).astype(int)
    or_df['anatomical_risk_score'] = risk_score

    # Grouped predictors
    or_df['f_unfavorable_any'] = ((or_df['F5'] >= 2) | (or_df['F7'] >= 2) | (or_df['F8'] >= 2)).astype(int)
    or_df['f_unfavorable_heavy'] = (((or_df['F5'] >= 2).astype(int) + (or_df['F7'] >= 2).astype(int) + (or_df['F8'] >= 2).astype(int)) >= 2).astype(int)
    a_unf_counts = np.zeros(len(or_df), dtype=int)
    for var in a_cols:
        a_unf_counts += (or_df[var] > or_df[var].median()).astype(int)
    a_median = np.median(a_unf_counts) if len(a_unf_counts) > 0 else 0
    or_df['a_unfavorable_heavy'] = (a_unf_counts > a_median).astype(int)

    # Binary outcomes from ROC-derived thresholds
    if optimal_threshold_mai is not None:
        or_df['mai_change_binary'] = np.where(
            or_df[['init_mai_huedegree', 'final_mai_huedegree']].notna().all(axis=1),
            (or_df['final_mai_huedegree'] - or_df['init_mai_huedegree'] > optimal_threshold_mai).astype(int),
            np.nan
        )
    else:
        or_df['mai_change_binary'] = np.nan
    ohip_init_cols = [f'OHIP_{i}' for i in range(1, 6)]
    ohip_final_cols = [f'OHIP_{i}_recall' for i in range(1, 6)]
    gohai_init_cols = [f'GOHAI_{i}' for i in range(1, 13)]
    gohai_final_cols = [f'GOHAI_{i}_recall' for i in range(1, 13)]
    ohip_init = or_df[ohip_init_cols].sum(axis=1).where(or_df[ohip_init_cols].notna().all(axis=1), np.nan)
    ohip_final = or_df[ohip_final_cols].sum(axis=1).where(or_df[ohip_final_cols].notna().all(axis=1), np.nan)
    gohai_init = or_df[gohai_init_cols].sum(axis=1).where(or_df[gohai_init_cols].notna().all(axis=1), np.nan)
    gohai_final = or_df[gohai_final_cols].sum(axis=1).where(or_df[gohai_final_cols].notna().all(axis=1), np.nan)
    if optimal_threshold_ohip is not None:
        or_df['ohip_change_binary'] = np.where(ohip_init.notna() & ohip_final.notna(), ((ohip_final - ohip_init) > optimal_threshold_ohip).astype(int), np.nan)
    else:
        or_df['ohip_change_binary'] = np.nan
    if optimal_threshold_gohai is not None:
        or_df['gohai_change_binary'] = np.where(gohai_init.notna() & gohai_final.notna(), ((gohai_final - gohai_init) > optimal_threshold_gohai).astype(int), np.nan)
    else:
        or_df['gohai_change_binary'] = np.nan

    odds_ratios_table_mai = build_binned_or_table(or_df, 'mai_change_binary')
    odds_ratios_table_ohip = build_binned_or_table(or_df, 'ohip_change_binary')
    odds_ratios_table_gohai = build_binned_or_table(or_df, 'gohai_change_binary')

    insufficient_data_mai = len(odds_ratios_table_mai) == 0
    insufficient_data_ohip = len(odds_ratios_table_ohip) == 0
    insufficient_data_gohai = len(odds_ratios_table_gohai) == 0
    odds_ratios_img_mai = None
    odds_ratios_img_ohip = None
    odds_ratios_img_gohai = None

    # Perform cross-sectional analysis
    cross_sectional_results = perform_cross_sectional_analysis(cursor)
    
    # Analyze F1-F9 measurements
    def analyze_f_measurements(cursor):
        """Analyze F1-F9 measurements: descriptive stats, distributions, and correlations."""
        results = {}

        def hedges_g(group_a, group_b):
            """Effect size for two groups with small-sample correction."""
            a = pd.to_numeric(pd.Series(group_a), errors='coerce').dropna().values
            b = pd.to_numeric(pd.Series(group_b), errors='coerce').dropna().values
            n1, n2 = len(a), len(b)
            if n1 < 2 or n2 < 2:
                return np.nan
            s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
            pooled_num = ((n1 - 1) * (s1 ** 2)) + ((n2 - 1) * (s2 ** 2))
            pooled_den = (n1 + n2 - 2)
            if pooled_den <= 0:
                return np.nan
            pooled_sd = np.sqrt(pooled_num / pooled_den)
            if pooled_sd == 0 or np.isnan(pooled_sd):
                return np.nan
            d = (np.mean(b) - np.mean(a)) / pooled_sd
            correction = 1 - (3 / (4 * (n1 + n2) - 9)) if (n1 + n2) > 2 else 1
            return float(d * correction)
        
        # F1-F9 definitions
        f_measurements = {
            'F1': {'type': 'numeric', 'unit': 'mm', 'description': 'A felső állcsontgerinc magassága'},
            'F2': {'type': 'numeric', 'unit': 'mm³', 'description': 'Alámenős területek'},
            'F3': {'type': 'numeric', 'unit': 'mm', 'description': 'A szájpad boltozata'},
            'F4': {'type': 'numeric', 'unit': '°', 'description': 'A felső állcsontgerinc alakja'},
            'F5': {'type': 'categorical', 'values': {1: 'nincs', 2: 'van, „lötyögő" tuberek', 3: 'van, frontális gerincen'}, 'description': 'Lötyögő, csontmag nélküli gerinc', 'higher_code_worse': True},
            'F6': {'type': 'numeric', 'unit': '°', 'description': 'Az interalveoláris vonal és a rágósík által bezárt szög'},
            'F7': {'type': 'categorical', 'values': {1: 'nincs', 2: 'plató alakú', 3: 'orsó alakú'}, 'description': 'A torus palatinus', 'higher_code_worse': True},
            'F8': {'type': 'categorical', 'values': {1: 'nincs, most készül', 2: 'teljes lemezes/overdenture/részleges fémlemezes', 3: 'teljesen megtartott/rögzített'}, 'description': 'Az antagonista fogazati státusz', 'higher_code_worse': False},
            'F9': {'type': 'categorical', 'values': {1: 'a páciens elmondása szerint sincs', 2: 'van, de nem befolyásolta', 3: 'van és jelentősen befolyásolta'}, 'description': 'A garatreflex erőssége', 'higher_code_worse': True}
        }
        a_measurements = {
            'A1_Kaan': {'type': 'categorical', 'description': 'A mandibulagerinc topográfiája', 'values': {1: 'egészében megtartott', 2: 'elöl megtartott, oldalt lapos', 3: 'egészében lapos', 4: 'negatív', 5: 'mélyült negatív'}, 'higher_code_worse': True},
            'A3_any_side': {'type': 'categorical', 'description': 'A3 buccinator tasak (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A4_any_side': {'type': 'categorical', 'description': 'A4 torus mandibularis (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A5_any_side': {'type': 'categorical', 'description': 'A5 lingualis tasak (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A6_any_side': {'type': 'categorical', 'description': 'A6 feszes ínyborítás (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A7_any_side': {'type': 'categorical', 'description': 'A7 tuberculum forma (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A8_any_side': {'type': 'categorical', 'description': 'A8 inklinációs szög (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A9_any_side': {'type': 'categorical', 'description': 'A9 alakváltozás nyitás-záráskor (jobb+bal összevont)', 'values': {0: 'nincs eltérés egyik oldalon sem', 1: 'legalább egy oldalon eltérés'}, 'higher_code_worse': True},
            'A10': {'type': 'numeric', 'unit': '°', 'description': 'Állcsontreláció szögértéke'},
            'A11': {'type': 'categorical', 'description': 'Nyelvhelyzet (A11)', 'values': {1: 'nem elődomborodó', 2: 'puhán elődomborodó', 3: 'tömött, elődomborodó'}},
            'A12': {'type': 'categorical', 'description': 'Mucosalis állapot (A12)', 'values': {1: 'nem tapintható', 2: 'tapintható', 3: 'nyomásra érzékeny'}, 'higher_code_worse': True},
            'A13': {'type': 'categorical', 'description': 'Mandibuláris szegmentális rizikó (A13)', 'values': {1: 'panaszmentes', 2: 'hangjelenség fájdalom nélkül', 3: 'fájdalom/mzg.korlát'}, 'higher_code_worse': True},
            'A14': {'type': 'categorical', 'description': 'Protetikai szituáció (A14)', 'values': {1: 'nincs', 2: 'teljes lemezes/overdenture/részleges fémlemezes', 3: 'teljesen megtartott/rögzített'}, 'higher_code_worse': False}
        }
        all_measurements = {**f_measurements, **a_measurements}
        
        # Fetch F/A data
        cursor.execute("""
            SELECT "id", "TAJ", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                   "A1_Kaan", "A3_jobb", "A3_bal", "A4_jobb", "A4_bal", "A5_jobb", "A5_bal",
                   "A6_jobb", "A6_bal", "A7_jobb", "A7_bal", "A8_jobb", "A8_bal", "A9_jobb", "A9_bal",
                   "A10", "A11", "A12", "A13", "A14",
                   "init_mai_huedegree", "final_mai_huedegree",
                   "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5",
                   "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6",
                   "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12"
            FROM patients
            WHERE ("F1" IS NOT NULL OR "F2" IS NOT NULL OR "F3" IS NOT NULL OR "F4" IS NOT NULL
                   OR "F5" IS NOT NULL OR "F6" IS NOT NULL OR "F7" IS NOT NULL OR "F8" IS NOT NULL OR "F9" IS NOT NULL
                   OR "A1_Kaan" IS NOT NULL OR "A3_jobb" IS NOT NULL OR "A3_bal" IS NOT NULL OR "A4_jobb" IS NOT NULL
                   OR "A4_bal" IS NOT NULL OR "A5_jobb" IS NOT NULL OR "A5_bal" IS NOT NULL OR "A6_jobb" IS NOT NULL
                   OR "A6_bal" IS NOT NULL OR "A7_jobb" IS NOT NULL OR "A7_bal" IS NOT NULL OR "A8_jobb" IS NOT NULL
                   OR "A8_bal" IS NOT NULL OR "A9_jobb" IS NOT NULL OR "A9_bal" IS NOT NULL OR "A10" IS NOT NULL
                   OR "A11" IS NOT NULL OR "A12" IS NOT NULL OR "A13" IS NOT NULL OR "A14" IS NOT NULL)
            ORDER BY "id" DESC
        """)
        data = cursor.fetchall()
        
        if len(data) == 0:
            return {'status': 'insufficient_data', 'message': 'Nincs F/A anatómiai adat'}
        
        df = pd.DataFrame(data, columns=[
            'id', 'TAJ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9',
            'A1_Kaan', 'A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal',
            'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal', 'A9_jobb', 'A9_bal',
            'A10', 'A11', 'A12', 'A13', 'A14',
            'init_mai_huedegree', 'final_mai_huedegree',
            'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5',
            'GOHAI_1', 'GOHAI_2', 'GOHAI_3', 'GOHAI_4', 'GOHAI_5', 'GOHAI_6',
            'GOHAI_7', 'GOHAI_8', 'GOHAI_9', 'GOHAI_10', 'GOHAI_11', 'GOHAI_12'
        ])
        
        # Convert numeric columns from Decimal to float (NUMERIC types cause issues with scipy)
        numeric_cols = ['F1', 'F2', 'F3', 'F4', 'F6', 'A10', 'init_mai_huedegree', 'final_mai_huedegree',
                       'OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5',
                       'GOHAI_1', 'GOHAI_2', 'GOHAI_3', 'GOHAI_4', 'GOHAI_5', 'GOHAI_6',
                       'GOHAI_7', 'GOHAI_8', 'GOHAI_9', 'GOHAI_10', 'GOHAI_11', 'GOHAI_12']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert categorical columns to int (they might be Decimal too)
        categorical_cols = ['F5', 'F7', 'F8', 'F9', 'A1_Kaan', 'A3_jobb', 'A3_bal', 'A4_jobb', 'A4_bal',
                            'A5_jobb', 'A5_bal', 'A6_jobb', 'A6_bal', 'A7_jobb', 'A7_bal', 'A8_jobb', 'A8_bal',
                            'A9_jobb', 'A9_bal', 'A11', 'A12', 'A13', 'A14']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Int64 allows NaN

        # Combine right/left lower-jaw fields into single clinical flags (any-side abnormality)
        for base in ['A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']:
            right_col = f'{base}_jobb'
            left_col = f'{base}_bal'
            combined_col = f'{base}_any_side'
            right_vals = pd.to_numeric(df[right_col], errors='coerce')
            left_vals = pd.to_numeric(df[left_col], errors='coerce')
            both_missing = right_vals.isna() & left_vals.isna()
            any_abnormal = (right_vals >= 2).fillna(False) | (left_vals >= 2).fillna(False)
            combined = pd.Series(pd.NA, index=df.index, dtype='Int64')
            has_any_side_data = ~both_missing
            combined.loc[has_any_side_data] = 0
            combined.loc[has_any_side_data & any_abnormal] = 1
            df[combined_col] = combined
        
        # Calculate OHIP and GOHAI totals
        df['OHIP_total'] = df[['OHIP_1', 'OHIP_2', 'OHIP_3', 'OHIP_4', 'OHIP_5']].sum(axis=1)
        df['GOHAI_total'] = df[['GOHAI_1', 'GOHAI_2', 'GOHAI_3', 'GOHAI_4', 'GOHAI_5', 'GOHAI_6',
                                'GOHAI_7', 'GOHAI_8', 'GOHAI_9', 'GOHAI_10', 'GOHAI_11', 'GOHAI_12']].sum(axis=1)
        
        # Descriptive statistics for each F measurement
        descriptive_stats = {}
        distribution_plots = {}
        
        for f_name, f_info in f_measurements.items():
            f_data = df[f_name].dropna()
            
            if len(f_data) == 0:
                descriptive_stats[f_name] = {'status': 'no_data'}
                continue
            
            if f_info['type'] == 'numeric':
                descriptive_stats[f_name] = {
                    'mean': float(f_data.mean()),
                    'std': float(f_data.std()),
                    'median': float(f_data.median()),
                    'min': float(f_data.min()),
                    'max': float(f_data.max()),
                    'n': len(f_data),
                    'unit': f_info['unit']
                }
                
                # Create histogram
                fig = plt.figure(figsize=(8, 5))
                plt.hist(f_data, bins=15, edgecolor='black', alpha=0.7, color='#427aa1')
                plt.xlabel(f"{f_name} ({f_info['unit']})")
                plt.ylabel('Gyakoriság')
                plt.title(f"{f_name}: {f_info['description']}\nÁtlag: {f_data.mean():.2f} ± {f_data.std():.2f} {f_info['unit']} (n={len(f_data)})")
                plt.grid(True, alpha=0.3)
                distribution_plots[f_name] = plot_to_base64(fig)
                
            else:  # categorical
                value_counts = f_data.value_counts().sort_index()
                descriptive_stats[f_name] = {
                    'value_counts': {int(k): int(v) for k, v in value_counts.items()},
                    'percentages': {int(k): float(v/len(f_data)*100) for k, v in value_counts.items()},
                    'n': len(f_data)
                }
                
                # Create bar chart
                fig = plt.figure(figsize=(10, 5))
                labels = [f_info['values'].get(int(k), f'Érték {k}') for k in value_counts.index]
                bars = plt.bar(range(len(value_counts)), value_counts.values, color='#427aa1', alpha=0.7)
                plt.xlabel('Kategória')
                plt.ylabel('Gyakoriság')
                plt.title(f"{f_name}: {f_info['description']} (n={len(f_data)})")
                plt.xticks(range(len(value_counts)), labels, rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom')
                
                plt.tight_layout()
                distribution_plots[f_name] = plot_to_base64(fig)
        
        # Correlation analyses: F1-F9 vs outcomes (init_mai, final_mai, OHIP_total, GOHAI_total)
        correlations = {}
        
        outcomes = {
            'init_mai_huedegree': 'Kezdeti MAI (hue degree)',
            'final_mai_huedegree': 'Végső MAI (hue degree)',
            'OHIP_total': 'OHIP_total',
            'GOHAI_total': 'GOHAI_total'
        }
        
        for f_name, f_info in f_measurements.items():
            correlations[f_name] = {}
            
            for outcome_name, outcome_label in outcomes.items():
                # Get data where both F and outcome are available
                valid_data = df[[f_name, outcome_name]].dropna()
                
                if len(valid_data) < 10:  # Need at least 10 data points
                    correlations[f_name][outcome_name] = {'status': 'insufficient_data', 'n': len(valid_data)}
                    continue
                
                f_values = valid_data[f_name]
                outcome_values = valid_data[outcome_name]
                
                if f_info['type'] == 'numeric':
                    # Pearson correlation for numeric-numeric
                    corr, p_value = spearmanr(f_values, outcome_values)  # Using Spearman for robustness
                    correlations[f_name][outcome_name] = {
                        'correlation': float(corr),
                        'p_value': float(p_value),
                        'n': len(valid_data),
                        'type': 'Spearman'
                    }
                else:
                    # For categorical F, compare groups
                    groups = [outcome_values[f_values == val] for val in f_values.unique() if len(outcome_values[f_values == val]) > 0]
                    
                    if len(groups) >= 2:
                        # Kruskal-Wallis test for group comparison
                        stat, p_value = kruskal(*groups)
                        group_means = {int(val): float(outcome_values[f_values == val].mean()) 
                                     for val in f_values.unique() if len(outcome_values[f_values == val]) > 0}
                        
                        correlations[f_name][outcome_name] = {
                            'test': 'Kruskal-Wallis',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'group_means': group_means,
                            'n': len(valid_data)
                        }
                    else:
                        correlations[f_name][outcome_name] = {'status': 'insufficient_groups', 'n': len(valid_data)}
        
        # Create correlation heatmap for numeric F measurements
        numeric_f = [f for f, info in f_measurements.items() if info['type'] == 'numeric']
        if len(numeric_f) > 0:
            numeric_df = df[numeric_f + ['init_mai_huedegree', 'final_mai_huedegree', 'OHIP_total', 'GOHAI_total']].dropna()
            
            if len(numeric_df) >= 10:
                corr_matrix = numeric_df.corr(method='spearman')
                
                fig = plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           square=True, linewidths=1, cbar_kws={"shrink": 0.8})
                plt.title('Korrelációs mátrix: F1-F9 mérések és kimeneti változók\n(Spearman korreláció)')
                plt.tight_layout()
                correlation_heatmap = plot_to_base64(fig)
            else:
                correlation_heatmap = None
        else:
            correlation_heatmap = None

        # Effect-size heatmap (clinical signal focus): F1-F9 vs MAI/OHIP outcomes
        effect_size_heatmap = None
        top_effects = []
        try:
            effect_df = df.copy()
            effect_df['delta_mai_huedegree'] = effect_df['final_mai_huedegree'] - effect_df['init_mai_huedegree']
            outcomes_for_effect = ['init_mai_huedegree', 'delta_mai_huedegree', 'OHIP_total']
            outcome_labels = {
                'init_mai_huedegree': 'init_mai_huedegree',
                'delta_mai_huedegree': 'delta_mai_huedegree',
                'OHIP_total': 'OHIP_total'
            }

            effect_rows = []
            row_labels = []
            for f_name, f_info in f_measurements.items():
                row_vals = []
                for out in outcomes_for_effect:
                    sub = effect_df[[f_name, out]].dropna()
                    if len(sub) < 8:
                        row_vals.append(np.nan)
                        continue

                    if f_info['type'] == 'categorical':
                        # Favorable = 1, unfavorable = 2-3 for F5/F7/F8/F9
                        g0 = sub[sub[f_name] == 1][out].values
                        g1 = sub[sub[f_name] >= 2][out].values
                    else:
                        # Numeric F variables: median split for robust low-n comparison
                        thr = sub[f_name].median()
                        g0 = sub[sub[f_name] <= thr][out].values
                        g1 = sub[sub[f_name] > thr][out].values

                    g_val = hedges_g(g0, g1)
                    row_vals.append(g_val)
                    if np.isfinite(g_val):
                        top_effects.append({
                            'feature': f_name,
                            'outcome': outcome_labels[out],
                            'hedges_g': float(g_val),
                            'abs_g': float(abs(g_val)),
                            'n': int(len(sub))
                        })
                effect_rows.append(row_vals)
                row_labels.append(f_name)

            effect_matrix = pd.DataFrame(effect_rows, index=row_labels, columns=[outcome_labels[o] for o in outcomes_for_effect])
            if effect_matrix.notna().sum().sum() > 0:
                fig_eff = plt.figure(figsize=(9, 6))
                sns.heatmap(
                    effect_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    mask=effect_matrix.isna(),
                    cbar_kws={"label": "Hedges' g (unfavourable - favourable)"}
                )
                plt.title("F1-F9 effektusméret hőtérkép (p-érték helyett)\nPozitív g: kedvezőtlen anatómia -> magasabb (rosszabb) kimenet")
                plt.xlabel("Kimeneti változók")
                plt.ylabel("Anatómiai változók")
                plt.tight_layout()
                effect_size_heatmap = plot_to_base64(fig_eff)
                plt.close(fig_eff)

            top_effects = sorted(top_effects, key=lambda x: x['abs_g'], reverse=True)[:8]
        except Exception:
            effect_size_heatmap = None
            top_effects = []

        # Clinical relevance summary:
        # do "difference from reference patient (919-191-919)" flags link to QoL/mixing?
        clinical_links = []
        clinical_summary = {
            'tested_pairs': 0,
            'significant_pairs': 0,
            'moderate_or_stronger_pairs': 0
        }
        clinical_heatmap = None
        reference_taj = '919-191-919'
        reference_found = False
        reference_usable = False
        reference_message = ''
        try:
            clinical_df = df.copy()
            clinical_df['delta_mai_huedegree'] = clinical_df['final_mai_huedegree'] - clinical_df['init_mai_huedegree']
            clinical_df['TAJ_normalized'] = clinical_df['TAJ'].astype(str).str.replace(r'\D', '', regex=True)
            reference_taj_normalized = str(reference_taj).replace('-', '')

            # Use the latest row for the reference patient as etalon
            reference_rows = clinical_df[clinical_df['TAJ_normalized'] == reference_taj_normalized].copy()
            if len(reference_rows) == 0:
                # The TAJ may exist, but without F/A anatomy values (thus filtered out above)
                cursor.execute('SELECT "TAJ" FROM patients WHERE "TAJ" IS NOT NULL')
                all_tajs = [str(r[0]) for r in cursor.fetchall()]
                all_tajs_norm = [t.replace('-', '').replace(' ', '') for t in all_tajs]
                if reference_taj_normalized in all_tajs_norm:
                    reference_found = True
                    reference_message = f'Az etalon páciens ({reference_taj}) megtalálható, de nincs rögzített F/A anatómiai adata, ezért nem használható referenciaként.'
                else:
                    reference_message = f'Az etalon páciens ({reference_taj}) nem található az adatbázisban.'
                raise ValueError(reference_message)
            reference_found = True
            # Pick the reference row with the most available anatomy values
            ref_feature_cols = [c for c in all_measurements.keys() if c in reference_rows.columns]
            if len(ref_feature_cols) == 0:
                reference_message = f'Az etalon páciens ({reference_taj}) sorában nincs elemezhető anatómiai mező.'
                raise ValueError(reference_message)
            reference_rows['_nonnull_features'] = reference_rows[ref_feature_cols].notna().sum(axis=1)
            reference_row = reference_rows.sort_values(['_nonnull_features', 'id'], ascending=[False, False]).iloc[0]
            if int(reference_row.get('_nonnull_features', 0)) == 0:
                reference_message = f'Az etalon páciens ({reference_taj}) megtalálható, de nincs rögzített F/A anatómiai adata, ezért nem használható referenciaként.'
                raise ValueError(reference_message)
            reference_usable = True

            # Exclude reference patient rows from association testing
            analysis_df = clinical_df[clinical_df['TAJ_normalized'] != reference_taj_normalized].copy()
            clinical_outcomes = ['init_mai_huedegree', 'final_mai_huedegree', 'delta_mai_huedegree', 'OHIP_total', 'GOHAI_total']
            outcome_labels = {
                'init_mai_huedegree': 'Mixing ability (init MAI)',
                'final_mai_huedegree': 'Mixing ability (final MAI)',
                'delta_mai_huedegree': 'Mixing változás (delta MAI)',
                'OHIP_total': 'OHIP (magasabb = rosszabb QoL)',
                'GOHAI_total': 'GOHAI (magasabb = jobb QoL)'
            }

            heatmap_rows = []
            heatmap_idx = []
            for feature_name, feature_info in all_measurements.items():
                ref_value = reference_row.get(feature_name, np.nan)
                if pd.isna(ref_value):
                    continue

                # Binary exposure: differs from reference (1) vs matches reference (0)
                differs_col = f'{feature_name}_differs_from_reference'
                feature_vals = analysis_df[feature_name]
                differs_bool = (feature_vals != ref_value)
                if hasattr(differs_bool, 'fillna'):
                    differs_bool = differs_bool.fillna(False)
                differs_series = pd.Series(np.nan, index=analysis_df.index, dtype='float64')
                has_feature = feature_vals.notna()
                differs_series.loc[has_feature] = differs_bool.loc[has_feature].astype(int)
                analysis_df[differs_col] = differs_series

                row_vals = []
                for outcome_name in clinical_outcomes:
                    pair_df = analysis_df[[differs_col, outcome_name]].dropna()
                    if len(pair_df) < 10:
                        row_vals.append(np.nan)
                        continue
                    if pair_df[differs_col].nunique() < 2 or pair_df[outcome_name].nunique() < 2:
                        row_vals.append(np.nan)
                        continue

                    rho, p_value = spearmanr(pair_df[differs_col], pair_df[outcome_name])
                    if not np.isfinite(rho) or not np.isfinite(p_value):
                        row_vals.append(np.nan)
                        continue

                    abs_rho = float(abs(rho))
                    if abs_rho < 0.20:
                        effect_class = 'gyenge'
                    elif abs_rho < 0.40:
                        effect_class = 'kisebb-közepes'
                    elif abs_rho < 0.60:
                        effect_class = 'közepes'
                    else:
                        effect_class = 'erős'

                    if outcome_name == 'OHIP_total':
                        direction_hint = (
                            'eltérés az etalontól -> rosszabb QoL'
                            if rho > 0 else
                            'eltérés az etalontól -> jobb QoL'
                        )
                    elif outcome_name == 'GOHAI_total':
                        direction_hint = (
                            'eltérés az etalontól -> rosszabb QoL'
                            if rho < 0 else
                            'eltérés az etalontól -> jobb QoL'
                        )
                    else:
                        direction_hint = (
                            'eltérés az etalontól -> rosszabb objektív mixing'
                            if rho > 0 else
                            'eltérés az etalontól -> jobb objektív mixing'
                        )

                    values_map = feature_info.get('values', {})
                    if values_map and ref_value in values_map:
                        coding_note = f'etalon érték: {ref_value} ({values_map.get(ref_value)})'
                    else:
                        coding_note = f'etalon érték: {ref_value}'

                    clinical_links.append({
                        'feature': feature_name,
                        'feature_label': feature_info.get('description', feature_name),
                        'coding_note': coding_note,
                        'outcome': outcome_name,
                        'outcome_label': outcome_labels[outcome_name],
                        'rho': float(rho),
                        'p_value': float(p_value),
                        'n': int(len(pair_df)),
                        'abs_rho': abs_rho,
                        'effect_class': effect_class,
                        'is_significant': bool(p_value < 0.05),
                        'direction_hint': direction_hint
                    })
                    clinical_summary['tested_pairs'] += 1
                    if p_value < 0.05:
                        clinical_summary['significant_pairs'] += 1
                    if abs_rho >= 0.30:
                        clinical_summary['moderate_or_stronger_pairs'] += 1

                    row_vals.append(float(rho))
                heatmap_rows.append(row_vals)
                heatmap_idx.append(feature_name)

            clinical_heatmap_matrix = pd.DataFrame(
                heatmap_rows,
                index=heatmap_idx,
                columns=[outcome_labels[o] for o in clinical_outcomes]
            )
            if clinical_heatmap_matrix.notna().sum().sum() > 0:
                fig_clin = plt.figure(figsize=(10, 12))
                sns.heatmap(
                    clinical_heatmap_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    mask=clinical_heatmap_matrix.isna(),
                    cbar_kws={"label": "Spearman rho"}
                )
                plt.title('F + A anatómiai változók klinikai kapcsolata\n(QoL és mixing ability kimenetek)')
                plt.xlabel('Kimenetek')
                plt.ylabel('Anatómiai változók')
                plt.tight_layout()
                clinical_heatmap = plot_to_base64(fig_clin)
                plt.close(fig_clin)

            clinical_links = sorted(clinical_links, key=lambda x: x['abs_rho'], reverse=True)
        except Exception:
            clinical_links = []
            clinical_heatmap = None
            clinical_summary = {
                'tested_pairs': 0,
                'significant_pairs': 0,
                'moderate_or_stronger_pairs': 0
            }
            if not reference_message:
                reference_message = 'Az etalon referencia elemzése sikertelen.'
        
        return {
            'status': 'success',
            'descriptive_stats': descriptive_stats,
            'distribution_plots': distribution_plots,
            'correlations': correlations,
            'correlation_heatmap': correlation_heatmap,
            'effect_size_heatmap': effect_size_heatmap,
            'top_effects': top_effects,
            'clinical_links': clinical_links,
            'clinical_summary': clinical_summary,
            'clinical_heatmap': clinical_heatmap,
            'reference_taj': reference_taj,
            'reference_found': reference_found,
            'reference_usable': reference_usable,
            'reference_message': reference_message,
            'total_n': len(df)
        }
    
    f_analysis = analyze_f_measurements(cursor)

    return render_template('results.html',
                        patient_count=patient_count,
                        dropout_count=dropout_count,
                        lower_denture_count=lower_denture_count,
                        upper_denture_count=upper_denture_count,
                        male_age_distribution=male_age_distribution,
                        female_age_distribution=female_age_distribution,
                        dentures_img=dentures_img,
                        age_gender_img=age_gender_img,
                        q1_barchart = q1_barchart,
                        q2_barchart = q2_barchart,
                        ohip_init_mean=ohip_init_mean,
                        ohip_init_std=ohip_init_std,
                        ohip_final_mean=ohip_final_mean,
                        ohip_final_std=ohip_final_std,
                        gohai_init_mean=gohai_init_mean,
                        gohai_init_std=gohai_init_std,
                        gohai_final_mean=gohai_final_mean,
                        gohai_final_std=gohai_final_std,
                        init_mai_mean=init_mai_mean,
                        init_mai_std=init_mai_std,
                        final_mai_mean=final_mai_mean,
                        final_mai_std=final_mai_std,
                        roc_img_mai=roc_img_mai,
                        diff_img_mai=diff_img_mai,
                        roc_img_ohip=roc_img_ohip,
                        diff_img_ohip=diff_img_ohip,
                        roc_img_gohai=roc_img_gohai,
                        diff_img_gohai=diff_img_gohai,
                        optimal_threshold_mai=optimal_threshold_mai,
                        optimal_threshold_ohip=optimal_threshold_ohip,
                        optimal_threshold_gohai=optimal_threshold_gohai,
                        insufficient_data_mai=insufficient_data_mai,
                        insufficient_data_ohip=insufficient_data_ohip,
                        insufficient_data_gohai=insufficient_data_gohai,
                        odds_ratios_img_mai=odds_ratios_img_mai,
                        odds_ratios_img_ohip=odds_ratios_img_ohip,
                        odds_ratios_img_gohai=odds_ratios_img_gohai,
                        odds_ratios_table_mai=odds_ratios_table_mai,
                        odds_ratios_table_ohip=odds_ratios_table_ohip,
                        odds_ratios_table_gohai=odds_ratios_table_gohai,
                        cross_sectional_results=cross_sectional_results,
                        f_analysis=f_analysis
                        )


BLENDER_API_KEY = os.getenv("BLENDER_API_KEY")


def _float_or_none(val):
    """Cast to float, return None if missing or non-numeric."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return False  # sentinel: field present but invalid


@app.route('/api/morphometria', methods=['POST'])
def api_morphometria():
    """Blender addon végpont: morphometriai adatok feltöltése TAJ szerint."""
    api_key = request.headers.get('X-API-Key', '')
    if not BLENDER_API_KEY or not secrets.compare_digest(api_key, BLENDER_API_KEY):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body szükséges'}), 400

    TAJ = data.get('TAJ', '').strip()
    if not TAJ:
        return jsonify({'error': 'Hiányzó TAJ'}), 400

    numeric_fields = ('F1', 'F2', 'F3', 'F4', 'F6', 'A10', 'A2_mag_mm')
    converted = {}
    for field in numeric_fields:
        val = _float_or_none(data.get(field))
        if val is False:
            return jsonify({'error': f'{field} nem szám'}), 400
        converted[field] = val

    a2_modszer = data.get('A2_modszer')
    if a2_modszer not in ('A', 'B'):
        a2_modszer = None

    # Profile point arrays (list of dicts) → stored as JSON text in dedicated
    # columns. Must be a list (or None/missing); reject anything else.
    profiles = {}
    for field in ('F1_profil', 'A2_profil'):
        val = data.get(field)
        if val is None:
            profiles[field] = None
        elif isinstance(val, list):
            profiles[field] = json.dumps(val, ensure_ascii=False) if val else None
        else:
            return jsonify({'error': f'{field} nem lista'}), 400

    cursor = get_db_cursor()
    try:
        cursor.execute('SELECT COUNT(*) FROM patients WHERE "TAJ" = %s', (TAJ,))
        if cursor.fetchone()[0] == 0:
            return jsonify({'error': f'TAJ ({TAJ}) nem található a rendszerben'}), 404

        cursor.execute(
            """UPDATE patients SET
               "F1"                        = %s,
               "F2"                        = %s,
               "F3"                        = %s,
               "F4"                        = %s,
               "F6"                        = %s,
               "A10"                       = %s,
               "A2_mag_mm"                 = %s,
               "A2_modszer"                = %s,
               "F1_profil"                 = COALESCE(%s, "F1_profil"),
               "A2_profil"                 = COALESCE(%s, "A2_profil"),
               "modellanalizis_megtortent" = TRUE
               WHERE "TAJ" = %s""",
            (
                converted['F1'], converted['F2'], converted['F3'],
                converted['F4'], converted['F6'], converted['A10'],
                converted['A2_mag_mm'], a2_modszer,
                profiles['F1_profil'], profiles['A2_profil'],
                TAJ,
            )
        )
        db.commit()
        return jsonify({'success': True, 'TAJ': TAJ})
    except Exception as e:
        db.rollback()
        app.logger.error("api_morphometria hiba (TAJ=%s): %s", TAJ, e)
        return jsonify({'error': 'Adatbázis hiba. Kérlek próbáld újra.'}), 500
    finally:
        cursor.close()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)  # Change to an available port
