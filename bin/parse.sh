#!/bin/bash

# Default values (only for output dir and language)
OUTPUT_DIR="parsed"
LANG="en"
MD_DIR=""  # New parameter for markdown files

# Parse command line arguments
while getopts "i:o:l:m:rh" opt; do
  case $opt in
    i) INPUT_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    l) LANG="$OPTARG" ;;
    m) MD_DIR="$OPTARG" ;;    # New case for markdown dir
    r) RUN_AS_ROOT=true ;;    # New flag for root
    h)
      echo "Usage: $0 -i input_dir [-o output_dir] [-l language] [-m markdown_dir] [-r]"
      echo "Options:"
      echo "  -i: Input directory (required)"
      echo "  -o: Output directory (default: parsed)"
      echo "  -l: Language (default: en)"
      echo "  -m: Markdown directory (optional, to collect all .md files)"
      echo "  -r: Run as root (optional, default: run as current user)"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
  echo "MinerU PDF Parser"
  echo "----------------"
  echo "This script uses the MinerU docker container to parse a folder containing PDF files."
  echo "MinerU container is huge, it may pull 14GB of data to your machine."
  echo "It extracts text and metadata from PDFs using GPU acceleration."
  echo
  echo "Usage: $0 -i input_dir [-o output_dir] [-l language] [-m markdown_dir] [-r]"
  echo
  echo "Options:"
  echo "  -i: Input directory containing PDF files (required)"
  echo "  -o: Output directory for parsed results (default: parsed)"
  echo "  -l: Language of the PDFs (default: en)"
  echo "  -m: Markdown directory (optional, to collect all .md files)"
  echo "  -r: Run as root (optional, default: run as current user)"
  echo
  echo "Example:"
  echo "  $0 -i ./pdfs -o ./results -l en -m ./markdown"
  exit 1
fi

# Convert to absolute paths
INPUT_ABS=$(readlink -f "$INPUT_DIR")
OUTPUT_ABS=$(readlink -f "$OUTPUT_DIR")

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_ABS" ]; then
  mkdir -p "$OUTPUT_ABS"
  echo "Created output directory: $OUTPUT_ABS"
fi

# Create and handle markdown directory if specified
if [ ! -z "$MD_DIR" ]; then
  MD_ABS=$(readlink -f "$MD_DIR")
  mkdir -p "$MD_ABS"
  echo "Created markdown directory: $MD_ABS"
  # Add markdown dir as a volume mount
  MD_MOUNT="-v $MD_ABS:/data/markdown"
  # After processing, copy markdown files
  COPY_CMD="&& find /data/output -name '*.md' -exec cp {} /data/markdown/ \;"
else
  MD_MOUNT=""
  COPY_CMD=""
fi

# Set user parameters for docker
if [ "$RUN_AS_ROOT" = true ]; then
  USER_PARAMS=""
else
  USER_PARAMS="--user $(id -u):$(id -g)"
fi

# Run the docker command with the specified arguments
docker run --gpus=all \
  $USER_PARAMS \
  -v "$INPUT_ABS:/data/input" \
  -v "$OUTPUT_ABS:/data/output" \
  $MD_MOUNT \
  raychanan/mineru /bin/bash -c "magic-pdf \
  -p '/data/input' \
  -o '/data/output' \
  -l '$LANG' \
  $COPY_CMD"