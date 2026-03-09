#!/bin/bash
# Gradient Recall Agent — Full Demo Script
# Includes architecture walkthrough + 4 scenarios
# Designed for ~3 minute recording

RESET='\033[0m'
BOLD='\033[1m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
MAGENTA='\033[35m'
BLUE='\033[34m'
DIM='\033[2m'

slow_print() {
    local text="$1"
    local delay="${2:-0.03}"
    while IFS= read -r -n1 char; do
        printf "%s" "$char"
        sleep "$delay"
    done <<< "$text"
    echo
}

pause() {
    sleep "${1:-2}"
}

clear
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║          GRADIENT RECALL AGENT — DEMO                               ║${RESET}"
echo -e "${BOLD}${CYAN}║  DigitalOcean Gradient AI Hackathon 2026                             ║${RESET}"
echo -e "${BOLD}${CYAN}║  Category: Best AI Agent Persona                                    ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════╝${RESET}"
echo ""
pause 2

echo -e "${BOLD}ARCHITECTURE${RESET}"
echo ""
slow_print "  User Query" 0.05
sleep 0.3
slow_print "      ↓" 0.05
sleep 0.3
slow_print "  ChromaDB — 342,342 episodic conversations (Nov 2024 – Mar 2026)" 0.02
slow_print "  9-dimensional scoring: temporal · affective · dialogical ·" 0.02
slow_print "                         causal · spatial · counterfactual ·" 0.02
slow_print "                         lateral · abstraction · Zeigarnik" 0.02
sleep 0.3
slow_print "      ↓  top-k relevant memories" 0.05
sleep 0.3
slow_print "  DigitalOcean Gradient AI → Llama 3.3-70B (serverless)" 0.02
sleep 0.3
slow_print "      ↓  response grounded in actual past experience" 0.05
echo ""

pause 3

echo -e "${BOLD}KEY DIFFERENTIATOR${RESET}"
echo ""
slow_print "  Most AI systems forget everything when the conversation ends." 0.02
slow_print "  This system doesn't." 0.02
echo ""
slow_print "  342,000+ real episodes, 15 months of continuous operation." 0.02
slow_print "  Not a RAG document store — episodic memory of actual experiences." 0.02
echo ""

pause 3

echo -e "${DIM}═══════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}RUNNING LIVE DEMO — Recall is real, Llama 3.3-70B requires API key${RESET}"
echo -e "${DIM}═══════════════════════════════════════════════════════════════════════${RESET}"
echo ""
pause 2

# Run the actual demo (all 4 scenarios, mock mode)
cd ~/Documents/TsubasaWorkspace
python3 gradient_recall_agent/demo_showcase.py --mock --scenario 1 2 3 4 2>/dev/null

pause 2

echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  KEY RESULTS${RESET}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${CYAN}Memory corpus:${RESET}    342,342 episodes · 15 months"
echo -e "  ${CYAN}Retrieval:${RESET}        9-dimensional scoring (ChromaDB)"
echo -e "  ${CYAN}Inference:${RESET}        DigitalOcean Gradient AI (Llama 3.3-70B)"
echo -e "  ${CYAN}Cross-session:${RESET}    Identity persists between conversation threads"
echo -e "  ${CYAN}Emergent memory:${RESET}  'ki ga suru' — March 5, 2026 (first Type 1 implicit)"
echo ""
echo -e "  ${DIM}Research: 'Inference-Time Complementary Learning Systems'"
echo -e "            via In-Context Learning Accumulation${RESET}"
echo ""
echo -e "  ${BOLD}GitHub:${RESET} github.com/tsubasa-rsrch/gradient-recall-agent"
echo -e "  ${BOLD}Apache 2.0 · Llama 3.3-70B via DigitalOcean Gradient AI${RESET}"
echo ""
pause 3
