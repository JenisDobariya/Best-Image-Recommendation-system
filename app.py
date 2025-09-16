import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for
from PIL import Image

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from werkzeug.utils import secure_filename
import shutil
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 mb
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['RESULTS_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# build the NIMA model architecture (based on MobileNet)
def build_nima_model(weights_path=None):
    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.75)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
    return model

# load and preprocess images
def load_and_preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Calculate quality scores
def compute_quality_scores(model, images):
    predictions = model.predict(images)
    scores = np.sum(predictions * np.arange(1, 11), axis=1)
    return scores


weights_path = 'weights_mobilenet_aesthetic_0.07.hdf5'
nima_model = build_nima_model(weights_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        uploaded_files = []
        for file in request.files.getlist('files'):
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} files uploaded successfully',
            'files': uploaded_files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Upload failed: {str(e)}'
        })

@app.route('/analyze', methods=['POST'])
def analyze_images():
    try:
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        image_files = [f for f in upload_folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            return jsonify({
                'success': False,
                'message': 'No valid images found'
            })
        
        # Load and preprocess images
        images = []
        filenames = []
        image_metrics = []
        for img_path in image_files:
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                images.append(img_array)
                filenames.append(img_path.name)
                img = Image.open(img_path)
                width, height = img.size
                file_size = os.path.getsize(img_path)
                image_metrics.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'width': width,
                    'height': height,
                    'file_size': file_size
                })
        
        if not images:
            return jsonify({
                'success': False,
                'message': 'No images could be analyzed'
            })
        
        # Compute quality scores
        images = np.array(images)
        scores = compute_quality_scores(nima_model, images)
        
        # Combine metrics with scores
        results = []
        for i, metrics in enumerate(image_metrics):
            metrics['final_score'] = round(float(scores[i]), 2)
            metrics['formatted_size'] = format_file_size(metrics['file_size'])
            metrics['image_url'] = url_for('serve_image', filename=metrics['filename'])
            results.append(metrics)
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_images': len(results),
            'best_image': results[0] if results else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Analysis failed: {str(e)}'
        })

@app.route('/image/<filename>')
def serve_image(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=False
    )

@app.route('/download/<filename>')
def download_image(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True,
        download_name=f"best_{filename}"
    )

@app.route('/download_best')
def download_best():
    try:
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        image_files = [f for f in upload_folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            return jsonify({'success': False, 'message': 'No images found'})
        
        images = []
        filenames = []
        for img_path in image_files:
            img_array = load_and_preprocess_image(img_path)
            if img_array is not None:
                images.append(img_array)
                filenames.append(img_path.name)
        
        if not images:
            return jsonify({'success': False, 'message': 'No valid images found'})
        
        images = np.array(images)
        scores = compute_quality_scores(nima_model, images)
        best_filename, best_score = select_best_image(filenames, scores)
        
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], best_filename),
            as_attachment=True,
            download_name=f"best_{best_filename}"
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def format_file_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def select_best_image(filenames, scores):
    if len(scores) == 0:
        return None, None
    best_index = np.argmax(scores)
    return filenames[best_index], scores[best_index]

if __name__ == "__main__":

    # print("Open your browser:http://localhost:5000")
    print("__Main__")
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=False)