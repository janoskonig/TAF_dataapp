from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import mysql.connector
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import time
import threading
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime, date
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

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Retrieve environment variables
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")
# Ensure all variables are retrieved correctly
if not all([host, port, user, password, database]):
    raise ValueError("Missing one or more environment variables")
# Print environment variables
print("DB_HOST:", host)
print("DB_PORT:", port)
print("DB_USER:", user)
print("DB_PASSWORD:", password)
print("DB_NAME:", database)

# MySQL database connection
def create_db_connection():
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )

db = create_db_connection()

def ping_db():
    global db
    while True:
        time.sleep(600)  # Sleep for 10 minutes
        try:
            db.ping(reconnect=True, attempts=3, delay=5)
        except mysql.connector.Error as err:
            print(f"Error pinging MySQL: {err}")
            db = create_db_connection()

# Start the background thread to ping the database
thread = threading.Thread(target=ping_db)
thread.daemon = True
thread.start()

def get_db_cursor():
    global db
    try:
        db.ping(reconnect=True, attempts=3, delay=5)
    except mysql.connector.Error:
        db = create_db_connection()
    return db.cursor()

def calculate_standard_deviation(histogram):
    intensity_levels = np.arange(256)
    mean_intensity = np.average(intensity_levels, weights=histogram)
    variance = np.average((intensity_levels - mean_intensity)**2, weights=histogram)
    return np.sqrt(variance)

def calculate_mixing_index(std_dev_red, std_dev_blue):
    return std_dev_red + std_dev_blue

def process_image(image_path):
    image = Image.open(image_path)
    rgb_image = image.convert('RGB')
    r, g, b = rgb_image.split()
    
    histogram_r = r.histogram()
    histogram_b = b.histogram()
    
    std_dev_red = calculate_standard_deviation(histogram_r)
    std_dev_blue = calculate_standard_deviation(histogram_b)
    
    mai = calculate_mixing_index(std_dev_red, std_dev_blue)
    
    return mai

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

@app.route('/submit_questionnaire1', methods=['POST'])
def submit_questionnaire1():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    birthdate = request.form['birthdate']
    gender = request.form['gender']
    denture_type = request.form['denture_type']
    today_situation = request.form['today_situation']

    # Fetch responses for GOHAI questions
    GOHAI_questions = [request.form[f'GOHAI_{i}'] for i in range(1, 13)]

    # Fetch responses for OHIP questions
    OHIP_questions = [request.form[f'OHIP_{i}'] for i in range(1, 6)]

    sql = """
    INSERT INTO patients (TAJ, birthdate, gender, denture_type, GOHAI_1, GOHAI_2, GOHAI_3, GOHAI_4, GOHAI_5, GOHAI_6, GOHAI_7, GOHAI_8, GOHAI_9, GOHAI_10, GOHAI_11, GOHAI_12, OHIP_1, OHIP_2, OHIP_3, OHIP_4, OHIP_5, today_situation)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (TAJ, birthdate, gender, denture_type, *GOHAI_questions, *OHIP_questions, today_situation)
    cursor.execute(sql, values)
    db.commit()

    return render_template('confirmation.html')

@app.route('/submit_questionnaire2', methods=['POST'])
def submit_questionnaire2():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    # Function to fetch form data safely
    def get_form_data(field_name):
        value = request.form.get(field_name, None)
        return value if value else None
    # Fetch responses for F1 and F2 questions
    F1_jobb = get_form_data('F1_jobb')
    F1_bal = get_form_data('F1_bal')
    F2_jobb = get_form_data('F2_jobb')
    F2_bal = get_form_data('F2_bal')
    
    # Fetch responses for F3 to F9 questions
    F3 = get_form_data('F3')
    F4 = get_form_data('F4')
    F5 = get_form_data('F5')
    F6 = get_form_data('F6')
    F7 = get_form_data('F7')
    F8 = get_form_data('F8')
    F9 = get_form_data('F9')
    
    # Fetch responses for A1 and new questions
    A1 = get_form_data('A1')
    A1a_jobb = get_form_data('A1a_jobb')
    A1a_bal = get_form_data('A1a_bal')
    A1b_jobb = get_form_data('A1b_jobb')
    A1b_bal = get_form_data('A1b_bal')
    A2_jobb = get_form_data('A2_jobb')
    A2_bal = get_form_data('A2_bal')
    A3_jobb = get_form_data('A3_jobb')
    A3_bal = get_form_data('A3_bal')
    A4_jobb = get_form_data('A4_jobb')
    A4_bal = get_form_data('A4_bal')
    A5_jobb = get_form_data('A5_jobb')
    A5_bal = get_form_data('A5_bal')
    A6_jobb = get_form_data('A6_jobb')
    A6_bal = get_form_data('A6_bal')
    A7_jobb = get_form_data('A7_jobb')
    A7_bal = get_form_data('A7_bal')
    A8_jobb = get_form_data('A8_jobb')
    A8_bal = get_form_data('A8_bal')
    A9 = get_form_data('A9')
    A10 = get_form_data('A10')
    A11 = get_form_data('A11')
    A12 = get_form_data('A12')
    A13 = get_form_data('A13')

    
    # Check if TAJ exists
    cursor.execute("SELECT COUNT(*) FROM patients WHERE TAJ = %s", (TAJ,))
    result = cursor.fetchone()
    
    if result[0] == 0:
        # TAJ does not exist
        db.close()
        return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

    sql = """
        UPDATE patients SET 
        F1_jobb = %s, F1_bal = %s, 
        F2_jobb = %s, F2_bal = %s,
        F3 = %s, F4 = %s, F5 = %s, F6 = %s, F7 = %s, F8 = %s, F9 = %s, 
        A1 = %s, A1a_jobb = %s, A1a_bal = %s, A1b_jobb = %s, A1b_bal = %s, A2_jobb = %s, A2_bal = %s, A3_jobb = %s, A3_bal = %s,
        A4_jobb = %s, A4_bal = %s, A5_jobb = %s, A5_bal = %s, A6_jobb = %s, A6_bal = %s, A7_jobb = %s, A7_bal = %s, A8_jobb = %s, A8_bal = %s,
        A9 = %s, A10 = %s, A11 = %s, A12 = %s, A13 = %s
        WHERE TAJ = %s
        """
    values = (F1_jobb, F1_bal, F2_jobb, F2_bal, F3, F4, F5, F6, F7, F8, F9, 
              A1, A1a_jobb, A1a_bal, A1b_jobb, A1b_bal, A2_jobb, A2_bal, A3_jobb, A3_bal, 
              A4_jobb, A4_bal, A5_jobb, A5_bal, A6_jobb, A6_bal, A7_jobb, A7_bal, A8_jobb, A8_bal, 
              A9, A10, A11, A12, A13,
              TAJ)
    cursor.execute(sql, values)
    db.commit()

    return render_template('confirmation.html')

@app.route('/submit_questionnaire3', methods=['POST'])
def submit_questionnaire3():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    
    # Function to fetch form data safely
    def get_form_data(field_name):
        return request.form.get(field_name, '')

    # Fetch responses for today's situation
    today_situation_recall = get_form_data('today_situation_recall')
    change = get_form_data('change')
    
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

    # Fetch responses for MFIQ questions
    MFIQ_questions = [get_form_data(f'MFIQ_{i}') for i in range(1, 18)]

    # Check if TAJ exists
    cursor.execute("SELECT COUNT(*) FROM patients WHERE TAJ = %s", (TAJ,))
    result = cursor.fetchone()
    
    if result[0] == 0:
        # TAJ does not exist
        db.close()
        return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

    sql = """
    UPDATE patients SET 
    today_situation_recall = %s, perceived_change = %s,
    OHIP_1_recall = %s, OHIP_2_recall = %s, OHIP_3_recall = %s, OHIP_4_recall = %s, OHIP_5_recall = %s,
    GOHAI_1_recall = %s, GOHAI_2_recall = %s, GOHAI_3_recall = %s, GOHAI_4_recall = %s, GOHAI_5_recall = %s,
    GOHAI_6_recall = %s, GOHAI_7_recall = %s, GOHAI_8_recall = %s, GOHAI_9_recall = %s, GOHAI_10_recall = %s,
    GOHAI_11_recall = %s, GOHAI_12_recall = %s,
    MFIQ_1 = %s, MFIQ_2 = %s, MFIQ_3 = %s, MFIQ_4 = %s, MFIQ_5 = %s,
    MFIQ_6 = %s, MFIQ_7 = %s, MFIQ_8 = %s, MFIQ_9 = %s, MFIQ_10 = %s,
    MFIQ_11 = %s, MFIQ_12 = %s, MFIQ_13 = %s, MFIQ_14 = %s, MFIQ_15 = %s,
    MFIQ_16 = %s, MFIQ_17 = %s
    WHERE TAJ = %s
    """
    values = (today_situation_recall, change,
              OHIP_1_recall, OHIP_2_recall, OHIP_3_recall, OHIP_4_recall, OHIP_5_recall,
              GOHAI_1_recall, GOHAI_2_recall, GOHAI_3_recall, GOHAI_4_recall, GOHAI_5_recall,
              GOHAI_6_recall, GOHAI_7_recall, GOHAI_8_recall, GOHAI_9_recall, GOHAI_10_recall,
              GOHAI_11_recall, GOHAI_12_recall, *MFIQ_questions, TAJ)
    cursor.execute(sql, values)
    db.commit()
    return render_template('confirmation.html')

@app.route('/upload_init_mai')
def upload_init_mai():
    return render_template('upload_init_mai.html')

@app.route('/upload_final_mai')
def upload_final_mai():
    return render_template('upload_final_mai.html')

@app.route('/submit_init_mai', methods=['POST'])
def submit_init_mai():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    
    # Check if TAJ exists
    cursor.execute("SELECT COUNT(*) FROM patients WHERE TAJ = %s", (TAJ,))
    result = cursor.fetchone()
    
    if result[0] == 0:
        # TAJ does not exist
        return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

    # Save the image
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

        # Calculate MAI
        mai = process_image(file_path)

        # Update the database
        sql = """
        UPDATE patients SET 
        init_mai = %s, init_image_path = %s
        WHERE TAJ = %s
        """
        values = (mai, filename, TAJ)
        cursor.execute(sql, values)
        db.commit()

        return render_template('confirmation.html')
    else:
        flash('Allowed file types are tiff, tif')
        return redirect(request.url)

@app.route('/submit_final_mai', methods=['POST'])
def submit_final_mai():
    cursor = get_db_cursor()
    TAJ = request.form['TAJ']
    
    # Check if TAJ exists
    cursor.execute("SELECT COUNT(*) FROM patients WHERE TAJ = %s", (TAJ,))
    result = cursor.fetchone()
    
    if result[0] == 0:
        # TAJ does not exist
        return render_template('error.html', message="Ilyen TAJ még nem található a rendszerben! Kérlek előbb az első kérdőívet töltsd ki!")

    # Save the image
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

        # Calculate MAI
        mai = process_image(file_path)

        # Update the database
        sql = """
        UPDATE patients SET 
        final_mai = %s, final_image_path = %s
        WHERE TAJ = %s
        """
        values = (mai, filename, TAJ)
        cursor.execute(sql, values)
        db.commit()

        return render_template('confirmation.html')
    else:
        flash('Allowed file types are tiff, tif')
        return redirect(request.url)


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


@app.route('/results')
def results():
    cursor = get_db_cursor()
    cursor.execute("SELECT COUNT(*) FROM patients WHERE TAJ IS NOT NULL")
    patient_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patients WHERE (denture_type = 'lower' OR denture_type = 'both') AND denture_type IS NOT NULL")
    lower_denture_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM patients WHERE (denture_type = 'upper' OR denture_type = 'both') AND denture_type IS NOT NULL")
    upper_denture_count = cursor.fetchone()[0]
    # Denture Type Chart
    fig_dentures = plt.figure(figsize=(8, 6))
    labels = ['Alsó fogsor', 'Felső fogsor']
    values = [lower_denture_count, upper_denture_count]
    plt.bar(labels, values, color=['#4CAF50', '#FFC107'])
    plt.title('Teljes lemezes fogpótlások száma állcsontonként')
    plt.ylabel('Szám')
    dentures_img = plot_to_base64(fig_dentures)
    
    # Age and gender distribution logic
    cursor.execute("SELECT gender, birthdate FROM patients WHERE gender IS NOT NULL AND birthdate IS NOT NULL")
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
    cursor.execute("SELECT today_situation FROM patients WHERE today_situation IS NOT NULL")
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
    q1_barchart = plot_to_base64(fig_q1)

    # Q2 subjective CHANGE IN chewing ability
    cursor.execute("SELECT perceived_change FROM patients WHERE perceived_change IS NOT NULL")
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
    q2_barchart = plot_to_base64(fig_q2)
    
    # Initial OHIP and GOHAI calculations
    cursor.execute("SELECT OHIP_1, OHIP_2, OHIP_3, OHIP_4, OHIP_5 FROM patients WHERE OHIP_1 IS NOT NULL AND OHIP_2 IS NOT NULL AND OHIP_3 IS NOT NULL AND OHIP_4 IS NOT NULL AND OHIP_5 IS NOT NULL")
    initial_ohip_scores = cursor.fetchall()
    ohip_init_scores = [sum(row) for row in initial_ohip_scores]
    ohip_init_mean = np.mean(ohip_init_scores)
    ohip_init_std = np.std(ohip_init_scores)
    
    cursor.execute("SELECT GOHAI_1, GOHAI_2, GOHAI_3, GOHAI_4, GOHAI_5, GOHAI_6, GOHAI_7, GOHAI_8, GOHAI_9, GOHAI_10, GOHAI_11, GOHAI_12 FROM patients WHERE GOHAI_1 IS NOT NULL AND GOHAI_2 IS NOT NULL AND GOHAI_3 IS NOT NULL AND GOHAI_4 IS NOT NULL AND GOHAI_5 IS NOT NULL AND GOHAI_6 IS NOT NULL AND GOHAI_7 IS NOT NULL AND GOHAI_8 IS NOT NULL AND GOHAI_9 IS NOT NULL AND GOHAI_10 IS NOT NULL AND GOHAI_11 IS NOT NULL AND GOHAI_12 IS NOT NULL")
    gohai_scores = cursor.fetchall()
    gohai_init_scores = [sum(row) for row in gohai_scores]
    gohai_init_mean = np.mean(gohai_init_scores)
    gohai_init_std = np.std(gohai_init_scores)

    # Final OHIP and GOHAI calculations
    cursor.execute("SELECT OHIP_1_recall, OHIP_2_recall, OHIP_3_recall, OHIP_4_recall, OHIP_5_recall FROM patients WHERE OHIP_1_recall IS NOT NULL AND OHIP_2_recall IS NOT NULL AND OHIP_3_recall IS NOT NULL AND OHIP_4_recall IS NOT NULL AND OHIP_5_recall IS NOT NULL")
    final_ohip_scores = cursor.fetchall()
    ohip_final_scores = [sum(row) for row in final_ohip_scores]
    ohip_final_mean = np.mean(ohip_final_scores)
    ohip_final_std = np.std(ohip_final_scores)
    
    cursor.execute("SELECT GOHAI_1_recall, GOHAI_2_recall, GOHAI_3_recall, GOHAI_4_recall, GOHAI_5_recall, GOHAI_6_recall, GOHAI_7_recall, GOHAI_8_recall, GOHAI_9_recall, GOHAI_10_recall, GOHAI_11_recall, GOHAI_12_recall FROM patients WHERE GOHAI_1_recall IS NOT NULL AND GOHAI_2_recall IS NOT NULL AND GOHAI_3_recall IS NOT NULL AND GOHAI_4_recall IS NOT NULL AND GOHAI_5_recall IS NOT NULL AND GOHAI_6_recall IS NOT NULL AND GOHAI_7_recall IS NOT NULL AND GOHAI_8_recall IS NOT NULL AND GOHAI_9_recall IS NOT NULL AND GOHAI_10_recall IS NOT NULL AND GOHAI_11_recall IS NOT NULL AND GOHAI_12_recall IS NOT NULL")
    final_gohai_scores = cursor.fetchall()
    gohai_final_scores = [sum(row) for row in final_gohai_scores]
    gohai_final_mean = np.mean(gohai_final_scores)
    gohai_final_std = np.std(gohai_final_scores)

    # MAI calculations
    cursor.execute("SELECT init_mai, final_mai FROM patients WHERE init_mai IS NOT NULL AND final_mai IS NOT NULL")
    mai_scores = cursor.fetchall()
    init_mai_scores = [row[0] for row in mai_scores]
    final_mai_scores = [row[1] for row in mai_scores]
    init_mai_mean = np.mean(init_mai_scores)
    init_mai_std = np.std(init_mai_scores)
    final_mai_mean = np.mean(final_mai_scores)
    final_mai_std = np.std(final_mai_scores)

# Filter valid data for ROC analysis
    cursor.execute("SELECT TAJ, init_mai, final_mai, perceived_change FROM patients WHERE init_mai IS NOT NULL AND final_mai IS NOT NULL AND perceived_change IS NOT NULL")
    roc_data = cursor.fetchall()
    if len(roc_data) > 0:
        roc_df = pd.DataFrame(roc_data, columns=["TAJ", "init_mai", "final_mai", "perceived_change"])
        mai_score_difference = roc_df["final_mai"] - roc_df["init_mai"]
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
            plt.title('Receiver Operating Characteristic (ROC) görbe (MAI)')
            plt.legend(loc="lower right")
            roc_img_mai = plot_to_base64(fig_roc_mai)

            # Plot Score Difference vs Reported Improvement for MAI
            fig_diff_mai = plt.figure(figsize=(10, 6))
            plt.scatter(mai_score_difference, reported_improvement, alpha=0.5, label='résztvevők')
            plt.axvline(x=optimal_threshold_mai, color='r', linestyle='--', label=f'Az optimális vágópont: {optimal_threshold_mai:.2f}')
            plt.title('Rágóképesség pontkülönbség és a szubjektív javulás (MAI)')
            plt.xlabel('ΔMAI')
            plt.ylabel('Tapasztalt-e változást a \nrágóképességének tekintetében? \n(1 = igen, 0 = nem)')
            plt.legend()
            diff_img_mai = plot_to_base64(fig_diff_mai)
        else:
            roc_img_mai = None
            diff_img_mai = None
    else:
        roc_img_mai = None
        diff_img_mai = None

    # ROC Analysis for OHIP
    cursor.execute("SELECT TAJ, OHIP_1, OHIP_2, OHIP_3, OHIP_4, OHIP_5, OHIP_1_recall, OHIP_2_recall, OHIP_3_recall, OHIP_4_recall, OHIP_5_recall, perceived_change FROM patients WHERE OHIP_1 IS NOT NULL AND OHIP_2 IS NOT NULL AND OHIP_3 IS NOT NULL AND OHIP_4 IS NOT NULL AND OHIP_5 IS NOT NULL AND OHIP_1_recall IS NOT NULL AND OHIP_2_recall IS NOT NULL AND OHIP_3_recall IS NOT NULL AND OHIP_4_recall IS NOT NULL AND OHIP_5_recall IS NOT NULL AND perceived_change IS NOT NULL")
    ohip_data = cursor.fetchall()
    if len(ohip_data) > 0:
        ohip_df = pd.DataFrame(ohip_data, columns=["TAJ", "OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5", "OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall", "perceived_change"])
        ohip_init_scores = ohip_df[["OHIP_1", "OHIP_2", "OHIP_3", "OHIP_4", "OHIP_5"]].sum(axis=1)
        ohip_final_scores = ohip_df[["OHIP_1_recall", "OHIP_2_recall", "OHIP_3_recall", "OHIP_4_recall", "OHIP_5_recall"]].sum(axis=1)
        ohip_score_difference = ohip_final_scores - ohip_init_scores
        reported_improvement = ohip_df["perceived_change"].apply(lambda x: 1 if x in ['Kicsit javult', 'Sokat javult'] else 0)

        if len(reported_improvement.unique()) > 1:
            fpr, tpr, thresholds = roc_curve(reported_improvement, ohip_score_difference)
            roc_auc = roc_auc_score(reported_improvement, ohip_score_difference)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold_ohip = thresholds[optimal_idx]

            # Plot ROC curve for OHIP
            fig_roc_ohip = plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Fals pozitívok aránya')
            plt.ylabel('Valódi pozitívok aránya')
            plt.title('Receiver Operating Characteristic (ROC) görbe (OHIP)')
            plt.legend(loc="lower right")
            roc_img_ohip = plot_to_base64(fig_roc_ohip)

            # Plot Score Difference vs Reported Improvement for OHIP
            fig_diff_ohip = plt.figure(figsize=(10, 6))
            plt.scatter(ohip_score_difference, reported_improvement, alpha=0.5, label='résztvevők')
            plt.axvline(x=optimal_threshold_ohip, color='r', linestyle='--', label=f'Az optimális vágópont: {optimal_threshold_ohip:.2f}')
            plt.title('OHIP pontkülönbség és a szubjektív javulás (OHIP)')
            plt.xlabel('ΔOHIP')
            plt.ylabel('Tapasztalt-e változást a \nrágóképességének tekintetében? \n(1 = igen, 0 = nem)')
            plt.legend()
            diff_img_ohip = plot_to_base64(fig_diff_ohip)
        else:
            roc_img_ohip = None
            diff_img_ohip = None
    else:
        roc_img_ohip = None
        diff_img_ohip = None

    # ROC Analysis for GOHAI
    cursor.execute("SELECT TAJ, GOHAI_1, GOHAI_2, GOHAI_3, GOHAI_4, GOHAI_5, GOHAI_6, GOHAI_7, GOHAI_8, GOHAI_9, GOHAI_10, GOHAI_11, GOHAI_12, GOHAI_1_recall, GOHAI_2_recall, GOHAI_3_recall, GOHAI_4_recall, GOHAI_5_recall, GOHAI_6_recall, GOHAI_7_recall, GOHAI_8_recall, GOHAI_9_recall, GOHAI_10_recall, GOHAI_11_recall, GOHAI_12_recall, perceived_change FROM patients WHERE GOHAI_1 IS NOT NULL AND GOHAI_2 IS NOT NULL AND GOHAI_3 IS NOT NULL AND GOHAI_4 IS NOT NULL AND GOHAI_5 IS NOT NULL AND GOHAI_6 IS NOT NULL AND GOHAI_7 IS NOT NULL AND GOHAI_8 IS NOT NULL AND GOHAI_9 IS NOT NULL AND GOHAI_10 IS NOT NULL AND GOHAI_11 IS NOT NULL AND GOHAI_12 IS NOT NULL AND GOHAI_1_recall IS NOT NULL AND GOHAI_2_recall IS NOT NULL AND GOHAI_3_recall IS NOT NULL AND GOHAI_4_recall IS NOT NULL AND GOHAI_5_recall IS NOT NULL AND GOHAI_6_recall IS NOT NULL AND GOHAI_7_recall IS NOT NULL AND GOHAI_8_recall IS NOT NULL AND GOHAI_9_recall IS NOT NULL AND GOHAI_10_recall IS NOT NULL AND GOHAI_11_recall IS NOT NULL AND GOHAI_12_recall IS NOT NULL AND perceived_change IS NOT NULL")
    gohai_data = cursor.fetchall()
    if len(gohai_data) > 0:
        gohai_df = pd.DataFrame(gohai_data, columns=["TAJ", "GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12", "GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall", "perceived_change"])
        gohai_init_scores = gohai_df[["GOHAI_1", "GOHAI_2", "GOHAI_3", "GOHAI_4", "GOHAI_5", "GOHAI_6", "GOHAI_7", "GOHAI_8", "GOHAI_9", "GOHAI_10", "GOHAI_11", "GOHAI_12"]].sum(axis=1)
        gohai_final_scores = gohai_df[["GOHAI_1_recall", "GOHAI_2_recall", "GOHAI_3_recall", "GOHAI_4_recall", "GOHAI_5_recall", "GOHAI_6_recall", "GOHAI_7_recall", "GOHAI_8_recall", "GOHAI_9_recall", "GOHAI_10_recall", "GOHAI_11_recall", "GOHAI_12_recall"]].sum(axis=1)
        gohai_score_difference = gohai_final_scores - gohai_init_scores
        reported_improvement = gohai_df["perceived_change"].apply(lambda x: 1 if x in ['Kicsit javult', 'Sokat javult'] else 0)

        if len(reported_improvement.unique()) > 1:
            fpr, tpr, thresholds = roc_curve(reported_improvement, gohai_score_difference)
            roc_auc = roc_auc_score(reported_improvement, gohai_score_difference)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold_gohai = thresholds[optimal_idx]

            # Plot ROC curve for GOHAI
            fig_roc_gohai = plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Fals pozitívok aránya')
            plt.ylabel('Valódi pozitívok aránya')
            plt.title('Receiver Operating Characteristic (ROC) görbe (GOHAI)')
            plt.legend(loc="lower right")
            roc_img_gohai = plot_to_base64(fig_roc_gohai)

            # Plot Score Difference vs Reported Improvement for GOHAI
            fig_diff_gohai = plt.figure(figsize=(10, 6))
            plt.scatter(gohai_score_difference, reported_improvement, alpha=0.5, label='résztvevők')
            plt.axvline(x=optimal_threshold_gohai, color='r', linestyle='--', label=f'Az optimális vágópont: {optimal_threshold_gohai:.2f}')
            plt.title('GOHAI pontkülönbség és a szubjektív javulás (GOHAI)')
            plt.xlabel('ΔGOHAI')
            plt.ylabel('Tapasztalt-e változást a \nrágóképességének tekintetében? \n(1 = igen, 0 = nem)')
            plt.legend()
            diff_img_gohai = plot_to_base64(fig_diff_gohai)
        else:
            roc_img_gohai = None
            diff_img_gohai = None
    else:
        roc_img_gohai = None
        diff_img_gohai = None
    
    def safe_float_conversion(x):
        try:
            return np.nan if x == "N/A" or x is None else float(x.split()[0])
        except AttributeError:
            return np.nan

    """
    # Define the range of cut-off points for MAI
    cutoff_points = range(start_value, end_value + 1)  # Adjust start_value and end_value based on your study

    # Initialize variables to store the best cut-off point and its corresponding metrics
    best_cutoff_point = None
    best_sensitivity = 0
    best_specificity = 0
    best_proportion_correct = 0

    # Loop through each cut-off point
    for cutoff in cutoff_points:
        # Initialize counts for the 2x2 contingency table
        TP = 0  # True Positives
        FP = 0  # False Positives
        TN = 0  # True Negatives
        FN = 0  # False Negatives

        # Loop through each patient's data
        for patient in patients_data:
            # Determine if the patient is restricted or non-restricted based on the current cut-off point
            if patient.MAI <= cutoff:
                if patient.perceived_restriction:  # Patient perceives restriction
                    TP += 1
                else:
                    FP += 1
            else:
                if patient.perceived_restriction:  # Patient perceives restriction
                    FN += 1
                else:
                    TN += 1

        # Calculate sensitivity, specificity, and proportion correctly predicted for the current cut-off point
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        proportion_correct = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

        # Store the metrics if they are better than the previous best
        if proportion_correct > best_proportion_correct:
            best_cutoff_point = cutoff
            best_sensitivity = sensitivity
            best_specificity = specificity
            best_proportion_correct = proportion_correct

    # Output the best cut-off point and its corresponding metrics
    print("Best Cut-off Point for MAI:", best_cutoff_point)
    print("Sensitivity:", best_sensitivity)
    print("Specificity:", best_specificity)
    print("Proportion Correctly Predicted:", best_proportion_correct)

    # Additional analysis for functional impairment scores if needed
    # For cut-off points from specific range (e.g., 25 to 45)
    for cutoff in specific_range:
        restricted_group = [patient for patient in patients_data if patient.MAI <= cutoff]
        non_restricted_group = [patient for patient in patients_data if patient.MAI > cutoff]

        # Calculate mean and standard deviation of functional impairment scores for each group
        mean_restricted = mean([patient.functional_score for patient in restricted_group])
        mean_non_restricted = mean([patient.functional_score for patient in non_restricted_group])
        std_restricted = std([patient.functional_score for patient in restricted_group])
        std_non_restricted = std([patient.functional_score for patient in non_restricted_group])

        # Perform t-test to compare functional impairment scores between the groups
        t_statistic, p_value = t_test(restricted_group.functional_score, non_restricted_group.functional_score)

        # Output the results of the t-test
        print("Cut-off Point:", cutoff)
        print("Mean Functional Score (Restricted):", mean_restricted)
        print("Mean Functional Score (Non-Restricted):", mean_non_restricted)
        print("T-test Statistic:", t_statistic)
        print("P-value:", p_value)
    """

    # Helper function to calculate heatmap data
    def calculate_heatmap_data(data_df, binary_col):
        heatmap_data = []
        no_valid_odds_ratios = True

        for feature in features:
            feature_data_for_heatmap = []
            for score in range(1, feature_scales[feature] + 1):  # Ensure proper range for scoring
                data_copy = data_df.copy()
                data_copy[feature] = (data_copy[feature] == score).astype(int)
                odds_ratio, (ci_lower, ci_upper) = calculate_odds_ratios_and_ci(data_copy.assign(MAI_változás_binary=data_copy[binary_col]), feature)
                feature_data_for_heatmap.append(f"{odds_ratio:.2f} ({ci_lower:.2f}, {ci_upper:.2f})")
                if not np.isnan(odds_ratio):
                    no_valid_odds_ratios = False
            heatmap_data.append(feature_data_for_heatmap)

        if no_valid_odds_ratios:
            heatmap_data = [["N/A" for _ in range(1, max(feature_scales.values()) + 1)] for _ in features]
        
        heatmap_df = pd.DataFrame(heatmap_data, index=features, columns=[f"Score {i}" for i in range(1, max(feature_scales.values()) + 1)])
        return heatmap_df, no_valid_odds_ratios
    # Features to analyze with scales
    features = [
        'F1_jobb', 'F1_bal', 'F2_jobb', 'F2_bal', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 
        'A1', 'A1a_jobb', 'A1a_bal', 'A1b_jobb', 'A1b_bal', 'A2_jobb', 'A2_bal', 'A3_jobb', 
        'A3_bal', 'A4_jobb', 'A4_bal', 'A5_jobb', 'A5_bal', 'A6_jobb', 'A6_bal', 'A7_jobb', 
        'A7_bal', 'A8_jobb', 'A8_bal', 'A9', 'A10', 'A11', 'A12', 'A13'
    ]
    feature_scales = {
        'F1_jobb': 3, 'F1_bal': 3, 'F2_jobb': 3, 'F2_bal': 3, 'F3': 3, 'F4': 3, 'F5': 3, 'F6': 3, 'F7': 3, 'F8': 3, 'F9': 3, 
        'A1': 5, 'A1a_jobb': 3, 'A1a_bal': 3, 'A1b_jobb': 3, 'A1b_bal': 3, 'A2_jobb': 3, 'A2_bal': 3, 'A3_jobb': 3, 'A3_bal': 3, 
        'A4_jobb': 3, 'A4_bal': 3, 'A5_jobb': 3, 'A5_bal': 3, 'A6_jobb': 3, 'A6_bal': 3, 'A7_jobb': 3, 'A7_bal': 3, 'A8_jobb': 3, 
        'A8_bal': 3, 'A9': 3, 'A10': 3, 'A11': 3, 'A12': 3, 'A13': 3
    }
    columns = ["TAJ", "birthdate"] + features + ["init_mai", "final_mai"]

    # Build the WHERE clause to exclude rows with NULL values for relevant columns
    where_clause = " AND ".join([f"{column} IS NOT NULL" for column in columns])

    # Fetch the patient count excluding rows with NULL values
    cursor.execute(f"SELECT {', '.join(columns)} FROM patients WHERE {where_clause}")
    all_data = cursor.fetchall()

    data_df = pd.DataFrame(all_data, columns=columns)
    # Calculate heatmap data for MAI
    data_df["MAI_változás_binary"] = (data_df["final_mai"] - data_df["init_mai"] > optimal_threshold_mai).astype(int)
    heatmap_df_mai, no_valid_odds_ratios_mai = calculate_heatmap_data(data_df, "MAI_változás_binary")
    fig_heatmap_mai = plt.figure(figsize=(12, 20))
    sns.heatmap(heatmap_df_mai.applymap(safe_float_conversion), annot=heatmap_df_mai, cmap="YlGnBu", fmt='')
    plt.title('Esélyhányadosok hőtérképe konfidenciaintervallumokkal a rágóképességváltozás tekintetében (MAI)')
    odds_ratios_img_mai = plot_to_base64(fig_heatmap_mai)

    # Calculate heatmap data for OHIP
    if len(data_df) == len(ohip_final_scores):
        data_df["OHIP_változás_binary"] = (np.array(ohip_final_scores) - np.array(ohip_init_scores) > optimal_threshold_ohip).astype(int)
        heatmap_df_ohip, no_valid_odds_ratios_ohip = calculate_heatmap_data(data_df.assign(MAI_változás_binary=data_df["OHIP_változás_binary"]), "OHIP_változás_binary")
        fig_heatmap_ohip = plt.figure(figsize=(12, 20))
        sns.heatmap(heatmap_df_ohip.applymap(safe_float_conversion), annot=heatmap_df_ohip, cmap="YlGnBu", fmt='')
        plt.title('Esélyhányadosok hőtérképe konfidenciaintervallumokkal a OHIP-pontszámváltozás tekintetében')
        odds_ratios_img_ohip = plot_to_base64(fig_heatmap_ohip)
    else:
        no_valid_odds_ratios_ohip = True
        odds_ratios_img_ohip = None

    # Calculate heatmap data for GOHAI
    if len(data_df) == len(gohai_final_scores):
        data_df["GOHAI_változás_binary"] = (np.array(gohai_final_scores) - np.array(gohai_init_scores) > optimal_threshold_gohai).astype(int)
        heatmap_df_gohai, no_valid_odds_ratios_gohai = calculate_heatmap_data(data_df.assign(MAI_változás_binary=data_df["GOHAI_változás_binary"]), "GOHAI_változás_binary")
        fig_heatmap_gohai = plt.figure(figsize=(12, 20))
        sns.heatmap(heatmap_df_gohai.applymap(safe_float_conversion), annot=heatmap_df_gohai, cmap="YlGnBu", fmt='')
        plt.title('Esélyhányadosok hőtérképe konfidenciaintervallumokkal a GOHAI-pontszámváltozás tekintetében')
        odds_ratios_img_gohai = plot_to_base64(fig_heatmap_gohai)
    else:
        no_valid_odds_ratios_gohai = True
        odds_ratios_img_gohai = None

    return render_template('results.html',
                        patient_count=patient_count,
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
                        fpr=fpr,
                        tpr=tpr,
                        optimal_threshold_mai=optimal_threshold_mai,
                        optimal_threshold_ohip=optimal_threshold_ohip,
                        optimal_threshold_gohai=optimal_threshold_gohai,
                        roc_auc=roc_auc,
                        roc_img_mai=roc_img_mai,
                        diff_img_mai=diff_img_mai,
                        roc_img_ohip=roc_img_ohip,
                        diff_img_ohip=diff_img_ohip,
                        roc_img_gohai=roc_img_gohai,
                        diff_img_gohai=diff_img_gohai,
                        insufficient_data_mai=no_valid_odds_ratios_mai,
                        insufficient_data_ohip=no_valid_odds_ratios_ohip,
                        insufficient_data_gohai=no_valid_odds_ratios_gohai,
                        odds_ratios_img_mai=odds_ratios_img_mai,
                        odds_ratios_img_ohip=odds_ratios_img_ohip,
                        odds_ratios_img_gohai=odds_ratios_img_gohai)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)  # Change to an available port
