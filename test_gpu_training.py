"""
Quick test to verify GPU is being used for training
"""

import tensorflow as tf
import numpy as np
from datetime import datetime

print("="*80)
print("🧪 QUICK GPU TRAINING TEST")
print("="*80)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detected: {len(gpus)}")

if gpus:
    for gpu in gpus:
        print(f"  {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("  ❌ No GPUs found!")
    exit(1)

# Create a simple model
print("\n📊 Creating test model...")
with tf.device('/GPU:0'):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(60, 10), return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')

print(f"✅ Model created with {model.count_params():,} parameters")

# Generate dummy data
print("\n📦 Generating test data...")
X_train = np.random.random((1000, 60, 10)).astype(np.float32)
y_train = np.random.random((1000, 1)).astype(np.float32)

print(f"✅ Data shape: {X_train.shape}")

# Train
print("\n🎓 Training on GPU...")
print("⚡ CHECK TASK MANAGER NOW!")
print("   Look at: Performance -> GPU -> 3D or Compute_0")

start = datetime.now()

with tf.device('/GPU:0'):
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        verbose=1
    )

end = datetime.now()
duration = (end - start).total_seconds()

print(f"\n✅ Training completed in {duration:.2f} seconds")
print(f"   Final loss: {history.history['loss'][-1]:.6f}")

# Verify device placement
print("\n🔍 Verifying device placement...")
for layer in model.layers:
    print(f"   {layer.name}: {layer.weights[0].device if layer.weights else 'No weights'}")

print("\n" + "="*80)
print("If GPU usage was shown in Task Manager, GPU is working! ✅")
print("="*80)