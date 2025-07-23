#!/bin/bash
#SBATCH --job-name=tiki_process
#SBATCH --partition=normal  # Change to your available partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=processing_%j.out
#SBATCH --error=processing_%j.err

# SLURM job script for batch processing with TIKI Analysis container

echo "============================================"
echo "TIKI Data Processing Job"
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
PROJECT_DIR="/home/milu10/src/tiki_code"
DATA_DIR="/sc-projects/sc-proj-cc15-preact/SP6/tiki_data"
NOTEBOOK_TO_RUN="notebooks/04_Passive_Preprocess.ipynb"  # Change as needed

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
echo "Notebook to run: $NOTEBOOK_TO_RUN"
echo ""

# Set environment variables for the container
export PYTHONPATH="$PROJECT_DIR/functions:$PROJECT_DIR"
export DATA_PATH="$DATA_DIR"
export PREPROCESSED_PATH="$DATA_DIR/preprocessed"
export PREPROCESSED_PATH_FREEZED="$DATA_DIR/preprocessed_freezed"
export REDCAP_PATH="$DATA_DIR/redcap"

echo "Environment variables:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  DATA_PATH: $DATA_PATH"
echo "  PREPROCESSED_PATH: $PREPROCESSED_PATH"
echo "  PREPROCESSED_PATH_FREEZED: $PREPROCESSED_PATH_FREEZED"
echo "  REDCAP_PATH: $REDCAP_PATH"
echo ""

echo "============================================"
echo "Starting Data Processing"
echo "============================================"
echo ""

# Run the notebook in the container
singularity exec \
    --bind "$PROJECT_DIR:/app" \
    --bind "$DATA_DIR:/data" \
    --bind "$HOME:$HOME" \
    --pwd /app \
    "$CONTAINER" \
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=7200 \
        --output-dir /app/output \
        "/app/$NOTEBOOK_TO_RUN"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✓ Processing completed successfully!"
    echo "Output saved to: output/$(basename $NOTEBOOK_TO_RUN)"
else
    echo "✗ Processing failed with exit code: $exit_code"
fi

echo "Job ended at: $(date)"
exit $exit_code
