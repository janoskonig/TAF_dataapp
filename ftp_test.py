from flask import Flask, request, redirect, url_for, flash
import os
from ftplib import FTP

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/'  # Temporary local storage

NAS_HOST = 'konigjanos.synology.me'  # Synology NAS IP address
NAS_USER = 'janos'
NAS_PASS = 'jiimbo_Jimb@109'
NAS_DIR = 'home'  # Folder on your NAS

def upload_to_nas(file_path, filename):
    with FTP(NAS_HOST) as ftp:
        ftp.login(NAS_USER, NAS_PASS)
        ftp.cwd(NAS_DIR)
        with open(file_path, 'rb') as file:
            ftp.storbinary(f'STOR {filename}', file)
        ftp.quit()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            upload_to_nas(file_path, filename)
            os.remove(file_path)  # Clean up temporary file
            return 'File uploaded successfully to NAS'
    return '''
    <!doctype html>
    <title>Upload STL File</title>
    <h1>Upload new STL File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)