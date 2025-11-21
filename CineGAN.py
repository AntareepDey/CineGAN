import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import numpy as np
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Additional TensorFlow logging suppression
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)


def configure_gpu(force_cuda=False):
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
    original_size = img.size
    
    img = img.resize((img_size, img_size), Image.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_size


def calculate_upscale_size(original_size, processing_size, upscale_factor):

    orig_w, orig_h = original_size
    
    if upscale_factor == 1.0:
        if orig_w > processing_size or orig_h > processing_size:
            return original_size
        else:
            return (processing_size, processing_size)
    
    aspect_ratio = orig_w / orig_h
    base_size = int(processing_size * upscale_factor)
    
    if aspect_ratio > 1:
        output_w = base_size
        output_h = int(base_size / aspect_ratio)
    elif aspect_ratio < 1:
        output_h = base_size
        output_w = int(base_size * aspect_ratio)
    else:
        output_w = output_h = base_size
    
    return (output_w, output_h)


def postprocess_and_save(output_tensor, save_path, original_size, processing_size=512, upscale_factor=1.0):
    img = output_tensor[0]
    img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)
    
    img = Image.fromarray(img)

    final_size = calculate_upscale_size(original_size, processing_size, upscale_factor)
    

    if final_size != img.size:
        img = img.resize(final_size, Image.LANCZOS)
    img.save(save_path, quality=95, optimize=True)
    
    return img, final_size


class ProgressSpinner:

    def __init__(self, message="Processing", style="dots"):
        self.message = message
        self.is_running = False
        self.thread = None
        self.start_time = None
        self.styles = {
            "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        }
        
        self.frames = self.styles.get(style, self.styles["dots"])
        
    def _get_elapsed_time(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def _format_time(self, seconds):
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
    
    def _animate(self):
        idx = 0
        while self.is_running:
            frame = self.frames[idx % len(self.frames)]
            elapsed = self._get_elapsed_time()
            
            if elapsed > 0.5:  # Show time after 0.5s
                time_str = f" [{self._format_time(elapsed)}]"
            else:
                time_str = ""

            bar_chars = ["▱", "▰"]
            bar_length = 20
            filled = int((idx % (bar_length * 2)) / 2)
            if idx % (bar_length * 2) >= bar_length:
                filled = bar_length - (idx % bar_length)
            
            bar = bar_chars[1] * filled + bar_chars[0] * (bar_length - filled)

            sys.stdout.write(f"\r  {frame}  {self.message}... {time_str}")
            sys.stdout.flush()
            
            time.sleep(0.08)
            idx += 1
    
    def start(self):
        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self, final_message=None, success=True):
        self.is_running = False
        if self.thread:
            self.thread.join()
        
        sys.stdout.write("\r" + " " * 100 + "\r")
        
        if final_message:
            icon = "✓" if success else "✗"
            elapsed = self._format_time(self._get_elapsed_time())
            print(f"  {icon}  {final_message} [{elapsed}]")
        
        sys.stdout.flush()


@tf.function
def generate_fast(model, input_tensor):
    return model(input_tensor, training=False)


def load_model_cached(model_path, img_size=512, quiet=False):
    if not hasattr(load_model_cached, 'generator'):
        if not quiet:
            spinner = ProgressSpinner("Loading model", style="braille")
            spinner.start()
        
        try:
            generator = build_generator(img_size)
            generator.load_weights(model_path)
            
            # Warm up
            dummy = tf.random.normal((1, img_size, img_size, 3))
            _ = generate_fast(generator, dummy)
            
            load_model_cached.generator = generator
            
            if not quiet:
                spinner.stop("Model loaded and ready", success=True)
        except Exception as e:
            if not quiet:
                spinner.stop(f"Failed to load model: {e}", success=False)
            raise e
    
    return load_model_cached.generator


def print_header():
    print("\n" + "─" * 60)
    print("   ╔══════════════════════════════════════════════════╗")
    print("   ║                     CineGAN                      ║")
    print("   ╚══════════════════════════════════════════════════╝")
    print("─" * 60 + "\n")


def print_result(input_path, output_path, elapsed, output_size, upscale_factor):
    print(f"\n{'─' * 60}")
    print(f"  ╔═══ PROCESSING COMPLETE ═══╗")
    print(f"  ║")
    print(f"  ║  Input:       {Path(input_path).name}")
    print(f"  ║  Output:      {Path(output_path).name}")
    print(f"  ║  Resolution:  {output_size[0]} × {output_size[1]} px")
    if upscale_factor > 1.0:
        print(f"  ║  Upscale:     {upscale_factor}x")
    print(f"  ║  Time:        {elapsed:.2f}s")
    print(f"  ║")
    print(f"  ╚{'═' * 26}╝")
    print(f"{'─' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Apply cinematic effect to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage (outputs at 512x512 or original size if larger)
  python cinematic_cli.py input.jpg
  
  # Upscale 2x while preserving aspect ratio
  python cinematic_cli.py input.jpg --upscale 2
  
  # Use CUDA GPU
  python cinematic_cli.py input.jpg --cuda
  
  # Quiet mode for scripting
  python cinematic_cli.py input.jpg --quiet
  
  # Batch processing with 2x upscale
  for img in photos/*.jpg; do
      python cinematic_cli.py "$img" --upscale 2 --quiet
  done
        '''
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', nargs='?', help='Output image path (optional)')
    parser.add_argument('--size', type=int, default=512, 
                        help='Processing size (default: 512)')
    parser.add_argument('--model', default='./models/generator.keras', 
                        help='Model path')
    parser.add_argument('--upscale', type=float, default=1.0, metavar='FACTOR',
                        help='Upscale output by factor, preserving aspect ratio (e.g., 2 for 2x)')
    parser.add_argument('--cuda', action='store_true',
                        help='Force use of CUDA GPU if available')
    parser.add_argument('--quiet', '-q', action='store_true', 
                        help='Suppress header and animations')
    
    args = parser.parse_args()
    
    # Configure GPU
    if args.cuda or not args.quiet:
        configure_gpu(args.cuda)
    
    if args.output is None:
        input_path = Path(args.input)
        suffix = f"_cinematic_{args.upscale}x" if args.upscale > 1.0 else "_cinematic"
        args.output = str(input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}")
    
    if not args.quiet:
        print_header()
    
    if not os.path.exists(args.input):
        print(f"  ✗  Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"  ✗  Error: Model not found: {args.model}")
        print("     Please specify correct model path with --model")
        sys.exit(1)
    
    # Load model
    try:
        generator = load_model_cached(args.model, args.size, args.quiet)
    except Exception as e:
        print(f"  ✗  Error loading model: {e}")
        sys.exit(1)
    
    # Process image with animated progress
    if not args.quiet:
        print(f"  ▸  Input: {args.input}\n")
        spinner = ProgressSpinner("Generating cinematic effect", style="braille")
        spinner.start()
    
    start_time = time.time()
    
    try:
        input_tensor, original_size = load_and_preprocess(args.input, args.size)
        output_tensor = generate_fast(generator, tf.constant(input_tensor)).numpy()
        
        _, final_size = postprocess_and_save(
            output_tensor, 
            args.output, 
            original_size,
            processing_size=args.size,
            upscale_factor=args.upscale
        )
        
        elapsed = time.time() - start_time
        
        if not args.quiet:
            spinner.stop("Cinematic effect applied", success=True)
            print_result(args.input, args.output, elapsed, final_size, args.upscale)
        else:
            print(f"  ✓  {args.output} [{final_size[0]}×{final_size[1]}] ({elapsed:.2f}s)")
            
    except Exception as e:
        if not args.quiet:
            spinner.stop(f"Processing failed: {e}", success=False)
        else:
            print(f"  ✗  Error processing image: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
