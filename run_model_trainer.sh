#!/bin/bash

# ════════════════════════════════════════════════════════════════════════════
#  Model Trainer Launcher - Interactive ML Model Training Interface
# ════════════════════════════════════════════════════════════════════════════

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║        🚀 MODEL TRAINER - Interactive ML Experimentation Platform         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if models are trained
if [ ! -f "artifacts/results/structured_model_results.csv" ]; then
    echo "⚠️  Model results not found."
    echo ""
    echo "You have two options:"
    echo ""
    echo "Option 1: Use the Interactive Trainer (RECOMMENDED)"
    echo "  - Dynamically select and train models"
    echo "  - Configure custom parameters"
    echo "  - See real-time evaluation results"
    echo ""
    echo "Option 2: Pre-train all models"
    echo "  Run: python run_ecommerce_training.py"
    echo ""
    echo "Starting Interactive Model Trainer..."
    echo ""
else
    echo "✅ Pre-trained models found in artifacts/results/"
    echo ""
    echo "Features:"
    echo "  ✓ Dynamic model selection (6 models)"
    echo "  ✓ Interactive parameter tuning"
    echo "  ✓ Real-time training and evaluation"
    echo "  ✓ Live confusion matrices and ROC curves"
    echo "  ✓ Automatic metrics calculation"
    echo ""
fi

echo "────────────────────────────────────────────────────────────────────────────"
echo ""

# Activate virtual environment if it exists
if [ -d ".ml-venv" ]; then
    source .ml-venv/bin/activate
fi

# Launch the model trainer
.ml-venv/bin/python -m streamlit run model_trainer.py \
    --server.headless true \
    --logger.level=error \
    --browser.gatherUsageStats false

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Server stopped."
echo "════════════════════════════════════════════════════════════════════════════"
