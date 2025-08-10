#!/bin/bash

# RETGEN Training Launcher with Dashboard
# This script launches the training in the background and opens the monitoring dashboard

echo "=========================================="
echo "🚀 RETGEN Training Launcher"
echo "=========================================="

# Configuration
MAX_SAMPLES=1000000
BATCH_SIZE=64
CHECKPOINT_INTERVAL=50000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if training is already running
if pgrep -f "train_retgen_complete.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Training is already running!${NC}"
    echo "Opening dashboard to monitor existing training..."
else
    echo -e "${GREEN}✓ Starting new training session${NC}"
    
    # Create necessary directories
    mkdir -p models/checkpoints
    mkdir -p training_status
    mkdir -p logs
    
    # Clean up old status file
    rm -f training_status/status.json
    
    # Start training in background using nohup
    echo -e "${GREEN}📊 Starting training with:${NC}"
    echo "   - Max samples: $MAX_SAMPLES"
    echo "   - Batch size: $BATCH_SIZE"
    echo "   - Checkpoint interval: $CHECKPOINT_INTERVAL"
    echo ""
    
    nohup python3 train_retgen_complete.py \
        --max-samples $MAX_SAMPLES \
        --batch-size $BATCH_SIZE \
        --checkpoint-interval $CHECKPOINT_INTERVAL \
        > logs/training_output.log 2>&1 &
    
    TRAINING_PID=$!
    echo -e "${GREEN}✓ Training started with PID: $TRAINING_PID${NC}"
    echo "   Log file: logs/training_output.log"
    
    # Wait a moment for training to initialize
    sleep 3
    
    # Check if training started successfully
    if ps -p $TRAINING_PID > /dev/null; then
        echo -e "${GREEN}✓ Training is running successfully${NC}"
    else
        echo -e "${RED}✗ Training failed to start. Check logs/training_output.log${NC}"
        exit 1
    fi
fi

# Start a simple HTTP server for the dashboard
echo ""
echo -e "${GREEN}🌐 Starting web server for dashboard...${NC}"

# Kill any existing Python HTTP servers on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null

# Start HTTP server in background
cd /home/bud/Desktop/bud-dev/retgen/retgen/retgen_implementation
python3 -m http.server 8080 --bind 127.0.0.1 > /dev/null 2>&1 &
SERVER_PID=$!

echo -e "${GREEN}✓ Web server started on port 8080${NC}"

# Open dashboard in browser
DASHBOARD_URL="http://localhost:8080/training_dashboard.html"
echo ""
echo -e "${GREEN}📊 Opening training dashboard...${NC}"
echo "   URL: $DASHBOARD_URL"

# Try to open in default browser
if command -v xdg-open > /dev/null; then
    xdg-open "$DASHBOARD_URL" 2>/dev/null
elif command -v open > /dev/null; then
    open "$DASHBOARD_URL" 2>/dev/null
else
    echo -e "${YELLOW}Please open your browser and navigate to: $DASHBOARD_URL${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Training system launched successfully!${NC}"
echo "=========================================="
echo ""
echo "📝 Useful commands:"
echo "   - View training log: tail -f logs/training_output.log"
echo "   - Check training status: cat training_status/status.json | python3 -m json.tool"
echo "   - Stop training: pkill -f train_retgen_complete.py"
echo "   - Stop web server: kill $SERVER_PID"
echo ""
echo "The dashboard will auto-refresh every 2 seconds."
echo "Press Ctrl+C to stop the web server (training will continue in background)"
echo ""

# Keep the web server running
wait $SERVER_PID