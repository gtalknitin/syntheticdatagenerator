#!/bin/bash
# V10 Generation Monitor
# Monitors progress and validates data quality in real-time

LOG_FILE="v10_final.log"
CHECK_INTERVAL=120  # Check every 2 minutes

echo "========================================"
echo "V10 GENERATION MONITOR"
echo "========================================"
echo "Log file: $LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
    # Check if process is running
    if ! pgrep -f "python.*v10/generator.py" > /dev/null; then
        echo "[$(date +%H:%M:%S)] ⚠️  Process not running!"

        # Check if completed
        if tail -50 "$LOG_FILE" 2>/dev/null | grep -q "V10 DATA GENERATION COMPLETE"; then
            echo ""
            echo "✅ GENERATION COMPLETE!"
            echo ""
            tail -30 "$LOG_FILE" | grep -A 20 "V10 DATA GENERATION COMPLETE"
            break
        fi

        # Check for errors
        if tail -100 "$LOG_FILE" 2>/dev/null | grep -q "Traceback\|Error\|Exception"; then
            echo ""
            echo "❌ GENERATION FAILED WITH ERROR:"
            tail -50 "$LOG_FILE" | grep -A 10 "Error\|Exception\|Traceback"
            break
        fi
    fi

    # Get current progress
    current=$(tail -500 "$LOG_FILE" 2>/dev/null | grep "^\s*\[" | tail -1)

    # Count files generated
    file_count=$(ls data/generated/v10_real_enhanced/hourly/*.csv 2>/dev/null | wc -l | tr -d ' ')

    # Show progress
    echo "[$(date +%H:%M:%S)] Files: $file_count/438 | $current"

    sleep $CHECK_INTERVAL
done

# Final validation
echo ""
echo "========================================"
echo "Running Data Quality Validation..."
echo "========================================"

source venv/bin/activate
python scripts/validation/validate_v10_quality.py

echo ""
echo "Monitor complete."
