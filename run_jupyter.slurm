#!/bin/bash
#SBATCH --job-name=tiki_jupyter
#SBATCH --partition=normal  # Change to your available partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jupyter_%j.out
#SBATCH --error=jupyter_%j.err

# SLURM job script for running Jupyter Lab with TIKI Analysis container

echo "============================================"
echo "TIKI Jupyter Lab Job"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo ""

# Load Singularity module (adjust if needed)
module load singularity
echo "Singularity version: $(singularity --version)"

# Set up variables
CONTAINER="tiki_analysis.sif"
PORT=8888
NODE=$(hostname -s)
PROJECT_DIR="/home/milu10/src/tiki_code"
DATA_DIR="/sc-projects/sc-proj-cc15-preact/SP6/tiki_data"

# Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container $CONTAINER not found!"
    echo "Please build the container first using: ./build_container.sh"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

echo "Container: $CONTAINER"
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "Port: $PORT"
echo ""

echo "============================================"
echo "Starting Jupyter Lab"
echo "============================================"
echo ""
echo "To connect to Jupyter Lab:"
echo "1. On your local machine, run:"
echo "   ssh -L $PORT:$NODE:$PORT $USER@your_login_node"
echo "2. Open your browser and go to:"
echo "   http://localhost:$PORT"
echo ""
echo "Jupyter Lab is starting..."
echo ""

# Set environment variables for the container
export PYTHONPATH="$PROJECT_DIR/functions:$PROJECT_DIR"

# Run Jupyter Lab in the container
singularity exec \
    --bind "$PROJECT_DIR:/app" \
    --bind "$DATA_DIR:/data" \
    --bind "$HOME:$HOME" \
    --pwd /app \
    "$CONTAINER" \
    jupyter lab \
        --ip=0.0.0.0 \
        --port=$PORT \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password=''

echo ""
echo "Jupyter Lab session ended at: $(date)"
