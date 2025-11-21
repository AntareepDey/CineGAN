"""
Run: python server.py
Run with CUDA: python server.py --cuda
"""

import os
import sys
import argparse

# Suppress TensorFlow warnings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs (INFO, WARNING, ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops messages

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import uuid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import time
from PIL import Image
import numpy as np

# Additional TensorFlow logging suppression
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)


def configure_gpu(force_cuda=False):
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if force_cuda:
        if not gpus:
            print("WARNING: --cuda flag set but no CUDA GPUs found. Using CPU.")
            return False
        
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[*] Using CUDA GPU: {len(gpus)} device(s) found")
            return True
        except RuntimeError as e:
            print(f"WARNING: GPU configuration failed: {e}")
            return False
    else:
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[*] GPU detected: {len(gpus)} device(s) available")
                return True
            except RuntimeError:
                pass
        return False


class SEBlock(layers.Layer):
    def __init__(self, filters, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
    
    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.filters // self.ratio, activation='relu')
        self.dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.filters))
    
    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return inputs * se
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "ratio": self.ratio,
        })
        return config


class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
    
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, 3, padding='same')
        self.norm1 = tf.keras.layers.GroupNormalization(groups=-1)
        self.conv2 = layers.Conv2D(self.filters, 3, padding='same')
        self.norm2 = tf.keras.layers.GroupNormalization(groups=-1)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return inputs + x
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


def pixel_shuffle(x, scale=2):
    return tf.nn.depth_to_space(x, scale)


def build_generator(img_size=512):
    """Build generator architecture"""
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Encoder
    e1 = layers.Conv2D(64, 7, padding='same')(inputs)
    e1 = tf.keras.layers.GroupNormalization(groups=-1)(e1)
    e1 = layers.ReLU()(e1)
    
    e2 = tf.keras.layers.SpectralNormalization(layers.Conv2D(128, 3, strides=2, padding='same'))(e1)
    e2 = tf.keras.layers.GroupNormalization(groups=-1)(e2)
    e2 = layers.ReLU()(e2)
    
    e3 = tf.keras.layers.SpectralNormalization(layers.Conv2D(256, 3, strides=2, padding='same'))(e2)
    e3 = tf.keras.layers.GroupNormalization(groups=-1)(e3)
    e3 = layers.ReLU()(e3)
    
    e4 = tf.keras.layers.SpectralNormalization(layers.Conv2D(512, 3, strides=2, padding='same'))(e3)
    e4 = tf.keras.layers.GroupNormalization(groups=-1)(e4)
    e4 = layers.ReLU()(e4)
    
    # Bottleneck with 9 residual blocks
    x = e4
    for _ in range(9):
        x = ResidualBlock(512)(x)
    
    # Decoder with attention skip connections
    # Upsample 1
    x = layers.Conv2D(1024, 3, padding='same')(x)
    x = layers.Lambda(lambda t: pixel_shuffle(t, 2))(x)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    x = layers.ReLU()(x)
    
    e3_att = SEBlock(256)(e3)
    x = layers.Concatenate()([x, e3_att])
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    x = layers.ReLU()(x)
    
    # Upsample 2
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.Lambda(lambda t: pixel_shuffle(t, 2))(x)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    x = layers.ReLU()(x)
    
    e2_att = SEBlock(128)(e2)
    x = layers.Concatenate()([x, e2_att])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    x = layers.ReLU()(x)
    
    # Upsample 3
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.Lambda(lambda t: pixel_shuffle(t, 2))(x)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    x = layers.ReLU()(x)
    
    e1_att = SEBlock(64)(e1)
    x = layers.Concatenate()([x, e1_att])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x)
    x = layers.ReLU()(x)
    
    # Output
    outputs = layers.Conv2D(3, 7, padding='same', activation='tanh')(x)
    
    return keras.Model(inputs, outputs, name='generator')

def load_and_preprocess(image_path, img_size=512):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)

    img_resized = img.resize((img_size, img_size), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0

    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_size


def calculate_output_size(original_size, processing_size):

    orig_w, orig_h = original_size
    if orig_w >= processing_size or orig_h >= processing_size:
        return original_size
    aspect_ratio = orig_w / orig_h
    
    if aspect_ratio > 1:
        output_w = processing_size
        output_h = int(processing_size / aspect_ratio)
    elif aspect_ratio < 1:
        output_h = processing_size
        output_w = int(processing_size * aspect_ratio)
    else:
        output_w = output_h = processing_size
    
    return (output_w, output_h)


def postprocess_and_save(output_tensor, save_path, original_size, processing_size=512):

    img = output_tensor[0]
    img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)

    img = Image.fromarray(img)

    final_size = calculate_output_size(original_size, processing_size)
    if final_size != img.size:
        img = img.resize(final_size, Image.LANCZOS)

    img.save(save_path, quality=95, optimize=True)
    
    return img, final_size


def parse_args():
    parser = argparse.ArgumentParser(
        description='CineGAN Local Web Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start server (CPU/GPU auto-detect)
  python server.py
  
  # Force CUDA GPU usage
  python server.py --cuda
  
  # Custom port
  python server.py --port 8080
  
  # Custom model path
  python server.py --model ./my_model/generator.keras
        '''
    )
    parser.add_argument('--cuda', action='store_true',
                        help='Force use of CUDA GPU if available')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port (default: 5000)')
    parser.add_argument('--model', default='./models/generator.keras',
                        help='Model path (default: ./models/generator.keras)')
    parser.add_argument('--size', type=int, default=512,
                        help='Processing size (default: 512)')
    
    return parser.parse_args()


args = parse_args()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['MODEL_PATH'] = args.model
app.config['IMG_SIZE'] = args.size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

print("=" * 60)
print("CineGAN Server - Starting Up")
print("=" * 60)
configure_gpu(args.cuda)

print(f"Loading model from: {app.config['MODEL_PATH']}")

try:
    generator = build_generator(app.config['IMG_SIZE'])
    generator.load_weights(app.config['MODEL_PATH'])
    
    # Warm up
    print("Warming up model...")
    dummy = tf.random.normal((1, app.config['IMG_SIZE'], app.config['IMG_SIZE'], 3))
    _ = generator(dummy, training=False)
    
    print("✓ Model loaded and ready!")
    print("=" * 60)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your model path and try again.")
    sys.exit(1)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        file.save(input_path)

        output_filename = f"cinematic_{unique_id}_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        print(f"Processing: {filename}")
        start_time = time.time()
        
        input_tensor, original_size = load_and_preprocess(input_path, app.config['IMG_SIZE'])
        output_tensor = generator(tf.constant(input_tensor), training=False).numpy()
        _, final_size = postprocess_and_save(
            output_tensor, 
            output_path, 
            original_size,
            processing_size=app.config['IMG_SIZE']
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f}s - Output: {final_size[0]}x{final_size[1]}")
        
        # Clean up input
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_url': f'/output/{output_filename}',
            'processing_time': f"{elapsed:.2f}s",
            'output_resolution': f"{final_size[0]}x{final_size[1]}"
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/output/<filename>')
def get_output(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))


@app.route('/health')
def health():
    return jsonify({'status': 'ready', 'model': 'loaded'})


if __name__ == '__main__':
    print(f"\nServer starting at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
