#!/bin/bash
# Record Gradient Recall Agent demo video (mock mode)
# Output: gradient_demo_video.mp4

OUTPUT="$HOME/Documents/TsubasaWorkspace/gradient_recall_agent/gradient_demo_video.mp4"

echo "Recording Gradient Recall Agent demo..."
echo "Output: $OUTPUT"
echo ""

# Start screen recording (macOS built-in)
# Recording for ~150 seconds (2.5 minutes)
osascript - "$OUTPUT" <<'APPLESCRIPT'
on run {outputPath}
    -- Open Terminal and run the demo
    tell application "Terminal"
        activate
        set w to do script ""
        delay 1

        -- Clear screen and set font size
        do script "printf '\\033[2J\\033[H'" in w
        delay 0.5

        -- Run the demo in mock mode (all 4 scenarios, no pause between)
        do script "cd ~/Documents/TsubasaWorkspace && python3 gradient_recall_agent/demo_showcase.py --mock 2>/dev/null" in w
    end tell

    delay 2

    -- Start screen capture
    do shell script "screencapture -V 140 " & quoted form of outputPath & " &"

    delay 145
    return outputPath
end run
APPLESCRIPT

echo "Demo recording complete: $OUTPUT"
ls -lh "$OUTPUT" 2>/dev/null || echo "File not found - check recording"
