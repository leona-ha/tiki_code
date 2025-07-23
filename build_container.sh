#!/bin/bash

# Build script for TIKI Analysis Singularity Container
# This script builds the Singularity container from the definition file

set -e  # Exit on any error

CONTAINER_NAME="tiki_analysis.sif"
DEF_FILE="tiki_analysis.def"

echo "============================================"
echo "Building TIKI Analysis Singularity Container"
echo "============================================"

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "Error: Definition file $DEF_FILE not found!"
    exit 1
fi

# Check if apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer is not installed or not in PATH"
    echo "Please load the apptainer module first:"
    echo "  module load apptainer"
    exit 1
fi

echo "Definition file: $DEF_FILE"
echo "Output container: $CONTAINER_NAME"
echo ""

# Remove existing container if it exists
if [ -f "$CONTAINER_NAME" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    rm "$CONTAINER_NAME"
fi

echo "Building container..."
echo "This may take 10-15 minutes..."
echo ""

# Try building with fakeroot first, fallback to regular build
if apptainer build --fakeroot "$CONTAINER_NAME" "$DEF_FILE" 2>/dev/null; then
    echo "✓ Container built successfully with fakeroot!"
elif apptainer build "$CONTAINER_NAME" "$DEF_FILE"; then
    echo "✓ Container built successfully!"
else
    echo "✗ Build failed!"
    echo ""
    echo "If you don't have fakeroot access, you may need to:"
    echo "1. Build on a system where you have sudo access"
    echo "2. Use the --remote option to build on Sylabs Cloud"
    echo "3. Ask your HPC admin to build it for you"
    exit 1
fi

echo ""
echo "============================================"
echo "Build completed successfully!"
echo "============================================"
echo ""
echo "Container file: $CONTAINER_NAME"
echo "Size: $(du -h $CONTAINER_NAME | cut -f1)"
echo ""
echo "Test the container:"
echo "  apptainer exec $CONTAINER_NAME python --version"
echo ""
echo "Start Jupyter Lab:"
echo "  apptainer exec --bind \$PWD:/app --bind /sc-projects/sc-proj-cc15-preact/SP6/tiki_data:/data $CONTAINER_NAME \\"
echo "    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "Get help:"
echo "  apptainer help $CONTAINER_NAME"
