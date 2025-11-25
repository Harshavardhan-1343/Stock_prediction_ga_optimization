"""
View and analyze prediction plots
"""

import os
import logging
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging. getLogger(__name__)

try:
    import config
    results_dir = config.RESULTS_DIR
except:
    results_dir = 'results'

logger.info("="*80)
logger.info("📊 PREDICTION PLOTS VIEWER")
logger.info("="*80)

# Check for prediction plots
prediction_plots = [
    'baseline_predictions.png',
    'optimized_test_predictions.png',
    'optimized_predictions.png'
]

logger.info(f"\n📂 Looking in: {os.path.abspath(results_dir)}\n")

available_plots = []
for plot_name in prediction_plots:
    plot_path = os.path.join(results_dir, plot_name)
    if os.path. exists(plot_path):
        logger.info(f"✅ Found: {plot_name}")
        available_plots.append((plot_name, plot_path))
    else:
        logger.info(f"❌ Missing: {plot_name}")

if not available_plots:
    logger. error("\n❌ No prediction plots found!")
    exit(1)

logger.info(f"\n✅ Found {len(available_plots)} prediction plot(s)")

# Display plots
logger.info("\n" + "="*80)
logger.info("📈 DISPLAYING PLOTS")
logger.info("="*80)

for plot_name, plot_path in available_plots:
    logger.info(f"\n📊 Opening: {plot_name}")
    
    try:
        # Open and display the image
        img = Image.open(plot_path)
        
        # Get image info
        logger.info(f"   Size: {img.size[0]}x{img.size[1]} pixels")
        logger.info(f"   Format: {img.format}")
        logger.info(f"   Mode: {img.mode}")
        
        # Show the image
        img.show()
        
        logger.info(f"   ✅ Image displayed successfully")
        
    except Exception as e:
        logger. error(f"   ❌ Error opening image: {e}")

logger.info("\n" + "="*80)
logger.info("✅ ALL PLOTS OPENED")
logger.info("="*80)

# Explain what each plot shows
logger.info("\n" + "="*80)
logger.info("📖 PLOT EXPLANATIONS")
logger.info("="*80)

logger.info("""
📊 baseline_predictions.png:
   - Shows predictions from the baseline model (before GA optimization)
   - Compare this with optimized predictions to see improvement
   - Top: Time series comparison (actual vs predicted)
   - Bottom: Scatter plot (shows correlation)

📊 optimized_test_predictions.png:
   - Shows predictions on TEST SET ONLY from GA-optimized model
   - This is the most important plot - shows performance on unseen data
   - Top: Time series of actual vs predicted prices
   - Bottom: Scatter plot showing prediction accuracy
   - If points are close to the diagonal line = good predictions

📊 optimized_predictions.png:
   - Shows predictions on ALL THREE datasets:
     * Training set (data model was trained on)
     * Validation set (data used during training for tuning)
     * Test set (completely unseen data)
   - Compare all three to check for overfitting
   - If train is much better than test = overfitting
   - If all three are similar = good generalization

📈 Key Metrics from your GA results:
   - R² Score: 0.9399 (93.99% of variance explained - EXCELLENT!)
   - RMSE: 0.0242 (very low error)
   - MAE: 0.0173 (predictions off by ~1. 73% on average)
   - MAPE: 2.12% (only 2. 12% error - GREAT!)
   - Directional Accuracy: 54.86% (predicts direction correctly >50%)
""")

logger.info("="*80)