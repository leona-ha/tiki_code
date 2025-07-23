#!/bin/bash

# TIKI Analysis Container Usage Script
# This script provides examples of how to use the Singularity container

CONTAINER="tiki_analysis.sif"
PROJECT_DIR="/home/milu10/src/tiki_code"
DATA_DIR="/sc-projects/sc-proj-cc15-preact/SP6/tiki_data"

echo "============================================"
echo "TIKI Analysis Container Usage Examples"
echo "============================================"
echo ""

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "❌ Container $CONTAINER not found!"
    echo "Please build it first using: ./build_container.sh"
    echo ""
    exit 1
fi

echo "✅ Container found: $CONTAINER"
echo "📁 Project directory: $PROJECT_DIR"
echo "📊 Data directory: $DATA_DIR"
echo ""

echo "============================================"
echo "Available Commands"
echo "============================================"
echo ""

echo "1. Test the container:"
echo "   singularity exec $CONTAINER python --version"
echo ""

echo "2. Interactive Python session:"
echo "   singularity exec --bind \$PWD:/app --bind $DATA_DIR:/data $CONTAINER python"
echo ""

echo "3. Interactive shell:"
echo "   singularity shell --bind \$PWD:/app --bind $DATA_DIR:/data $CONTAINER"
echo ""

echo "4. Start Jupyter Lab (interactive):"
echo "   singularity exec --bind \$PWD:/app --bind $DATA_DIR:/data $CONTAINER \\"
echo "     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""

echo "5. Run a specific notebook:"
echo "   singularity exec --bind \$PWD:/app --bind $DATA_DIR:/data $CONTAINER \\"
echo "     jupyter nbconvert --to notebook --execute /app/notebooks/04_Passive_Preprocess.ipynb"
echo ""

echo "6. Run Python script:"
echo "   singularity exec --bind \$PWD:/app --bind $DATA_DIR:/data $CONTAINER \\"
echo "     python /app/your_script.py"
echo ""

echo "============================================"
echo "SLURM Job Submission"
echo "============================================"
echo ""

echo "1. Submit Jupyter Lab job:"
echo "   sbatch run_jupyter.slurm"
echo ""

echo "2. Submit batch processing job:"
echo "   sbatch run_processing.slurm"
echo ""

echo "3. Check job status:"
echo "   squeue -u \$USER"
echo ""

echo "4. View job output:"
echo "   tail -f jupyter_JOBID.out"
echo "   tail -f processing_JOBID.out"
echo ""

echo "============================================"
echo "Quick Test"
echo "============================================"
echo ""

read -p "Would you like to test the container now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing container..."
    echo ""
    
    echo "Python version:"
    singularity exec "$CONTAINER" python --version
    echo ""
    
    echo "Installed packages (sample):"
    singularity exec "$CONTAINER" python -c "
import pandas as pd
import numpy as np
import sklearn
print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
print(f'sklearn: {sklearn.__version__}')
"
    echo ""
    
    echo "Testing data path access:"
    if [ -d "$DATA_DIR" ]; then
        echo "✅ Data directory exists: $DATA_DIR"
        singularity exec --bind "$DATA_DIR:/data" "$CONTAINER" ls -la /data | head -5
    else
        echo "❌ Data directory not found: $DATA_DIR"
    fi
    
    echo ""
    echo "✅ Container test completed!"
fi

echo ""
echo "For more information, run:"
echo "  singularity help $CONTAINER"
