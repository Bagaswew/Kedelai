from flask import Flask, request, render_template, url_for, redirect, session
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='Templates', static_folder='Static')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 1. Panggil "Otak" Model .h5
# (Pastikan path foldernya benar)
MODEL_PATH = os.path.join('Model', 'model_tanah_kedelai_skripsi.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'File model tidak ditemukan: {MODEL_PATH}')

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['Sesuai', 'Tidak_Sesuai']

UPLOAD_FOLDER = os.path.join('Static', 'Uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
TARGET_IMAGE_SIZE = (224, 224)


def is_allowed_image(filename):
    _, extension = os.path.splitext(filename.lower())
    return extension in ALLOWED_IMAGE_EXTENSIONS

# 2. Halaman Utama Website
@app.route('/', methods=['GET'])
def index():
    hasil = session.pop('hasil', None)
    akurasi = session.pop('akurasi', None)
    uploaded_image = session.pop('uploaded_image', None)
    warning = session.pop('warning', None)
    return render_template(
        'index.html',
        hasil=hasil,
        akurasi=akurasi,
        uploaded_image=uploaded_image,
        warning=warning,
    ) # Menampilkan desain web

# 3. Logika Saat Tombol Upload Ditekan
# ... (bagian atas app.py biarkan saja) ...
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        session['hasil'] = 'Tidak ada file yang diunggah'
        session['akurasi'] = 0
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file and file.filename:
        if not is_allowed_image(file.filename):
            session['warning'] = 'File yang diunggah bukan gambar. Silakan pilih JPG, PNG, WEBP, atau format gambar lain yang didukung.'
            return redirect(url_for('index'))

        # Simpan gambar sementara
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        uploaded_image = url_for('static', filename=f'Uploads/{filename}')
        
        # Buka dan kecilkan gambar sebelum membuat array untuk prediksi.
        try:
            with Image.open(filepath) as source_image:
                source_image = ImageOps.exif_transpose(source_image)
                source_image.thumbnail(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
                img = Image.new('RGB', TARGET_IMAGE_SIZE)
                offset = (
                    (TARGET_IMAGE_SIZE[0] - source_image.width) // 2,
                    (TARGET_IMAGE_SIZE[1] - source_image.height) // 2,
                )
                img.paste(source_image.convert('RGB'), offset)
        except (UnidentifiedImageError, OSError):
            session['warning'] = 'File gambar tidak dapat dibaca. Silakan pilih gambar yang valid.'
            return redirect(url_for('index'))

        img_array = np.asarray(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Prediksi menggunakan model
        classes = model.predict(img_array)
        result_index = np.argmax(classes[0])
        confidence = classes[0][result_index] * 100
        
        hasil_prediksi = class_names[result_index]
        
        # Kembalikan hasil ke index.html menggunakan Jinja2
        if hasil_prediksi == 'Sesuai':
            hasil_text = 'Lahan sesuai untuk tanaman kedelai'
        else:
            hasil_text = 'Lahan tidak sesuai untuk tanaman kedelai'

        session['hasil'] = hasil_text
        session['akurasi'] = f'{confidence:.2f}'
        session['uploaded_image'] = uploaded_image
        return redirect(url_for('index'))

    session['warning'] = 'File tidak valid. Pastikan file yang dipilih adalah gambar.'
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)