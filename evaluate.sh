#!/bin/bash

# ============================================================
# Operations Agent System - Complete Evaluation Suite
# ============================================================
# 
# This script runs all evaluation scripts in sequence:
# 1. Basic System Evaluation
# 2. Net Benefit Optimization
# 3. Dynamic Scenarios Evaluation
# 4. Claude Agent System Comparison
#
# Usage:
#   ./evaluate.sh           # Run all evaluations
#   ./evaluate.sh --quick   # Run quick evaluations only
#   ./evaluate.sh --agent   # Run only Claude Agent evaluation
#   ./evaluate.sh --help    # Show help
#
# Environment:
#   ANTHROPIC_API_KEY       # Optional: Enable Claude full reasoning
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║     🚀 Operations Agent System - Complete Evaluation Suite       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Help
print_help() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./evaluate.sh           Run all evaluations"
    echo "  ./evaluate.sh --quick   Run quick evaluations (skip slow training)"
    echo "  ./evaluate.sh --agent   Run only Claude Agent evaluation"
    echo "  ./evaluate.sh --system  Run only basic system evaluation"
    echo "  ./evaluate.sh --benefit Run only net benefit evaluation"
    echo "  ./evaluate.sh --dynamic Run only dynamic scenarios evaluation"
    echo "  ./evaluate.sh --help    Show this help message"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  ANTHROPIC_API_KEY       Set to enable Claude full reasoning"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./evaluate.sh"
    echo "  ANTHROPIC_API_KEY=sk-xxx ./evaluate.sh --agent"
}

# Check Python environment
check_environment() {
    echo -e "${BLUE}📦 Checking environment...${NC}"
    
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}✅ Using virtual environment: $VIRTUAL_ENV${NC}"
    elif [[ -d "venv" ]]; then
        echo -e "${YELLOW}⚠️  Virtual environment found but not activated${NC}"
        echo -e "${YELLOW}   Activating venv...${NC}"
        source venv/bin/activate
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    fi
    
    # Check for API key
    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        echo -e "${GREEN}✅ ANTHROPIC_API_KEY is set - Claude full reasoning enabled${NC}"
    else
        echo -e "${YELLOW}⚠️  ANTHROPIC_API_KEY not set - Claude will use fallback mode${NC}"
    fi
    
    # Create results directory
    mkdir -p results
    echo -e "${GREEN}✅ Results directory ready${NC}"
    echo ""
}

# Run a single evaluation
run_evaluation() {
    local script=$1
    local name=$2
    local icon=$3
    
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${icon} Running: ${name}${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    start_time=$(date +%s)
    
    if python "$script"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo ""
        echo -e "${GREEN}✅ ${name} completed in ${duration}s${NC}"
    else
        echo ""
        echo -e "${RED}❌ ${name} failed${NC}"
        return 1
    fi
    
    echo ""
}

# Summary
print_summary() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    📊 Evaluation Complete                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${YELLOW}Results saved to:${NC}"
    ls -la results/*.csv 2>/dev/null | while read line; do
        echo -e "  ${GREEN}📄${NC} $line"
    done
    
    echo ""
    echo -e "${YELLOW}Plots saved to:${NC}"
    ls -la results/*.png 2>/dev/null | while read line; do
        echo -e "  ${GREEN}📊${NC} $line"
    done
    
    echo ""
    echo -e "${GREEN}🎉 All evaluations completed successfully!${NC}"
}

# Main execution
main() {
    cd "$(dirname "$0")"
    
    print_banner
    
    # Parse arguments
    case "${1:-all}" in
        --help|-h)
            print_help
            exit 0
            ;;
        --quick)
            check_environment
            run_evaluation "evaluate_system.py" "Basic System Evaluation" "📊"
            run_evaluation "evaluate_agent_system.py" "Claude Agent Comparison" "🤖"
            print_summary
            ;;
        --agent)
            check_environment
            run_evaluation "evaluate_agent_system.py" "Claude Agent Comparison" "🤖"
            print_summary
            ;;
        --system)
            check_environment
            run_evaluation "evaluate_system.py" "Basic System Evaluation" "📊"
            print_summary
            ;;
        --benefit)
            check_environment
            run_evaluation "evaluate_net_benefit.py" "Net Benefit Optimization" "💰"
            print_summary
            ;;
        --dynamic)
            check_environment
            run_evaluation "evaluate_dynamic_scenarios.py" "Dynamic Scenarios Evaluation" "🌊"
            print_summary
            ;;
        all|"")
            check_environment
            
            echo -e "${YELLOW}Running all evaluations...${NC}"
            echo ""
            
            # 1. Basic System Evaluation
            run_evaluation "evaluate_system.py" "Basic System Evaluation" "📊"
            
            # 2. Net Benefit Optimization
            run_evaluation "evaluate_net_benefit.py" "Net Benefit Optimization" "💰"
            
            # 3. Dynamic Scenarios
            run_evaluation "evaluate_dynamic_scenarios.py" "Dynamic Scenarios Evaluation" "🌊"
            
            # 4. Claude Agent Comparison
            run_evaluation "evaluate_agent_system.py" "Claude Agent Comparison" "🤖"
            
            print_summary
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo ""
            print_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"

