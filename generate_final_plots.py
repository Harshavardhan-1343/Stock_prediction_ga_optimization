"""
Generate final plots from saved GA results
"""

import json
import pandas as pd
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import config, if fails use default
try:
    import config
    results_dir = config. RESULTS_DIR
except:
    results_dir = 'results'

logger.info(f"📂 Looking for GA results in: {os.path.abspath(results_dir)}")

# Check if results directory exists
if not os.path.exists(results_dir):
    logger.error(f"❌ Results directory does not exist: {results_dir}")
    logger.info("Creating results directory...")
    os.makedirs(results_dir, exist_ok=True)

# List all files in results directory
all_files = os.listdir(results_dir)
logger.info(f"\n📄 Files in results directory ({len(all_files)} total):")
for f in all_files:
    logger.info(f"   - {f}")

# Find GA results files
ga_files = [f for f in all_files if f.startswith('ga_results_') and f.endswith('.json')]

if not ga_files:
    logger.error("\n❌ No GA results file found!")
    logger.info("\n💡 Looking for files matching pattern: ga_results_*.json")
    
    # Try to find any JSON files
    json_files = [f for f in all_files if f.endswith('.json')]
    if json_files:
        logger.info(f"\n📋 Found {len(json_files)} JSON file(s):")
        for f in json_files:
            logger.info(f"   - {f}")
        
        # Ask user which file to use
        logger.info("\n🤔 Would you like to use one of these files? ")
        logger.info("Please update the script or rename your GA results file to match 'ga_results_*.json'")
    
    sys.exit(1)

# Sort by timestamp and get latest
ga_files.sort(reverse=True)
latest_file = ga_files[0]

logger.info(f"\n✅ Found {len(ga_files)} GA results file(s)")
logger.info(f"📂 Using latest: {latest_file}")

# Load results
filepath = os.path.join(results_dir, latest_file)
logger.info(f"📖 Reading from: {os.path.abspath(filepath)}")

try:
    with open(filepath, 'r') as f:
        results = json. load(f)
    
    logger.info("✅ Successfully loaded GA results")
    
    # Check what's in the results
    logger.info(f"\n📦 Results contains:")
    for key in results.keys():
        logger. info(f"   - {key}")
    
    # Create DataFrame from history
    if 'history' not in results:
        logger.error("❌ No 'history' key found in results!")
        logger.info(f"Available keys: {list(results.keys())}")
        sys.exit(1)
    
    history_df = pd.DataFrame(results['history'])
    
    logger.info(f"\n✅ Loaded history with {len(history_df)} generations")
    logger.info(f"   Columns: {list(history_df.columns)}")
    
    # Display best chromosome
    best_chromosome = results. get('best_chromosome')
    if best_chromosome:
        logger.info("\n" + "="*80)
        logger.info("🏆 BEST CHROMOSOME FOUND BY GA")
        logger.info("="*80)
        logger.info(f"   Fitness Score: {best_chromosome['fitness']:.6f}")
        
        logger.info("\n📋 Optimal Hyperparameters:")
        genes = best_chromosome['genes']
        for key, value in genes.items():
            logger.info(f"   {key:25s}: {value}")
        
        if best_chromosome. get('metrics'):
            logger.info("\n📊 Best Model Performance Metrics:")
            for key, value in best_chromosome['metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"   {key:25s}: {value:.6f}")
        
        if best_chromosome.get('model_params'):
            logger.info(f"\n🔢 Model Parameters: {best_chromosome['model_params']:,}")
    
    # Generate evolution plot
    from utils.visualization import plot_ga_evolution
    
    plot_path = os.path.join(results_dir, 'ga_evolution.png')
    plot_ga_evolution(history_df, save_path=plot_path)
    
    logger.info("\n" + "="*80)
    logger.info("✅ PLOTS GENERATED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"📊 Evolution plot saved to: {os.path.abspath(plot_path)}")
    
    # Display summary statistics
    logger.info("\n" + "="*80)
    logger.info("📈 EVOLUTION SUMMARY")
    logger.info("="*80)
    
    # Find the correct column name for best fitness
    best_fit_col = None
    for col in history_df.columns:
        if 'best' in col.lower() and 'fitness' in col.lower():
            best_fit_col = col
            break
    
    if best_fit_col:
        initial_fitness = history_df[best_fit_col].iloc[0]
        final_fitness = history_df[best_fit_col].iloc[-1]
        improvement = (initial_fitness - final_fitness) / initial_fitness * 100
        
        logger.info(f"   Initial Best Fitness: {initial_fitness:.6f}")
        logger.info(f"   Final Best Fitness:   {final_fitness:.6f}")
        logger.info(f"   Improvement:          {improvement:.2f}%")
        
        if improvement > 10:
            logger.info("\n   🎉 Excellent improvement! GA optimization was highly successful!")
        elif improvement > 5:
            logger.info("\n   ✅ Good improvement!  GA found better hyperparameters!")
        elif improvement > 0:
            logger.info("\n   ✅ Positive improvement! GA optimization helped!")
        else:
            logger. info("\n   ⚠️  Limited improvement.  Consider running more generations.")
    
    # Display GA parameters
    params = results.get('parameters', {})
    if params:
        logger.info("\n" + "="*80)
        logger.info("⚙️  GA CONFIGURATION")
        logger.info("="*80)
        for key, value in params.items():
            logger.info(f"   {key:20s}: {value}")
    
    logger.info("\n" + "="*80)
    logger.info("🎊 ALL DONE!")
    logger.info("="*80)
    
except FileNotFoundError:
    logger. error(f"❌ File not found: {filepath}")
    sys.exit(1)
except json.JSONDecodeError as e:
    logger.error(f"❌ Error parsing JSON: {e}")
    sys. exit(1)
except Exception as e:
    logger.error(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)