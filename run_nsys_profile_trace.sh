#!/bin/bash
export PATH="/local-dumpfolders/students/ivkets/nsight-systems-2025.2.1/bin:$PATH"
export JULIA_NUM_THREADS=16

# Program to profile
PROGRAM="/local-dumpfolders/students/ivkets/opt/julia/julia-1.11.4/bin/julia"
SCRIPT="run.jl"
# $PROGRAM -e "using Pkg; Pkg.activate(\"env\"); include(\"$SCRIPT\")"
#PROGRAM="./run.jl"  # Replace with your actual program path
#ARGS=""                   # Add any arguments your program needs

# Directory to store results
RESULT_DIR="nsight_results"

next_index=$(printf "%03d" $(( 
  $(ls "$RESULT_DIR"/profile_*.nsys-rep 2>/dev/null | 
    sed -E 's/.*profile_([0-9]+)\.nsys-rep/\1/' | 
    sort -n | tail -n 1 | awk '{print ($0+1)}' || echo 1) 
)))

# Output report name
REPORT_NAME="$RESULT_DIR/profile_${next_index}"

# Run nsys
nsys profile \
  --output "$REPORT_NAME" \
  --trace=cuda \
  $PROGRAM -e "using Pkg; Pkg.activate(\"env\"); include(\"$SCRIPT\")"