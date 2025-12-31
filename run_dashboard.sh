#!/bin/bash
# Quick start script for v7 dashboard

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}v7 Channel Prediction Dashboard${NC}"
echo -e "${CYAN}========================================${NC}\n"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check data directory
if [ ! -d "data" ]; then
    echo -e "${RED}Error: data/ directory not found${NC}"
    exit 1
fi

# Check required CSV files
REQUIRED_FILES=("data/TSLA_1min.csv" "data/SPY_1min.csv" "data/VIX_History.csv")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}Warning: $file not found${NC}"
    else
        echo -e "${GREEN}✓ Found $file${NC}"
    fi
done

# Check for model checkpoint
MODEL_PATH=""
if [ -d "checkpoints" ]; then
    MODEL_FILE=$(find checkpoints -name "*.pt" -o -name "*.pth" | head -1)
    if [ -n "$MODEL_FILE" ]; then
        echo -e "${GREEN}✓ Found model: $MODEL_FILE${NC}"
        MODEL_PATH="--model $MODEL_FILE"
    else
        echo -e "${YELLOW}Warning: No model checkpoint found in checkpoints/${NC}"
        echo -e "${YELLOW}Running in features-only mode${NC}"
    fi
else
    echo -e "${YELLOW}Warning: No checkpoints/ directory${NC}"
    echo -e "${YELLOW}Running in features-only mode${NC}"
fi

echo ""

# Parse command line arguments
DASHBOARD_TYPE="terminal"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --visual)
            DASHBOARD_TYPE="visual"
            shift
            ;;
        --save)
            DASHBOARD_TYPE="visual"
            EXTRA_ARGS="$EXTRA_ARGS --save $2"
            shift 2
            ;;
        --refresh)
            EXTRA_ARGS="$EXTRA_ARGS --refresh $2"
            shift 2
            ;;
        --export)
            EXTRA_ARGS="$EXTRA_ARGS --export $2"
            shift 2
            ;;
        --tf)
            shift
            EXTRA_ARGS="$EXTRA_ARGS --tf"
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
            done
            ;;
        --help|-h)
            echo "Usage: ./run_dashboard.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --visual              Use matplotlib visual dashboard"
            echo "  --save FILE           Save visual dashboard to FILE"
            echo "  --refresh SECONDS     Auto-refresh terminal dashboard"
            echo "  --export DIR          Export predictions to DIR"
            echo "  --tf TF1 TF2 ...      Show specific timeframes (visual only)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_dashboard.sh                          # Terminal dashboard"
            echo "  ./run_dashboard.sh --visual                 # Visual dashboard"
            echo "  ./run_dashboard.sh --visual --save out.png  # Save to file"
            echo "  ./run_dashboard.sh --refresh 300            # Auto-refresh every 5min"
            echo "  ./run_dashboard.sh --export results/        # Export predictions"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Run dashboard
if [ "$DASHBOARD_TYPE" = "terminal" ]; then
    echo -e "${CYAN}Starting terminal dashboard...${NC}\n"
    python3 dashboard.py $MODEL_PATH $EXTRA_ARGS
else
    echo -e "${CYAN}Starting visual dashboard...${NC}\n"
    python3 dashboard_visual.py $MODEL_PATH $EXTRA_ARGS
fi
