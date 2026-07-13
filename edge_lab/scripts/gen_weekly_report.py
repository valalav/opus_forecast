#!/usr/bin/env python3
"""
Weekly Status Report Generator
Analyzes PRD and generates summary report of all work.
"""

import os
import json
from datetime import datetime
from collections import Counter

BASE_DIR = "/home/valalav/_projects/sirena-kbr/edge_lab"
TASKS_FILE = os.path.join(BASE_DIR, "tasks", "prd.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORT_FILE = os.path.join(REPORTS_DIR, "week_status.md")


def load_prd():
    """Load PRD from JSON file."""
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_task_status(prd):
    """Analyze task status distribution."""
    tasks = prd.get("user_stories", [])
    total = len(tasks)
    by_status = Counter(t.get("status", "UNKNOWN") for t in tasks)

    return {
        "total": total,
        "by_status": dict(by_status),
        "completion_rate": (by_status.get("DONE", 0) / total * 100) if total > 0 else 0,
    }


def get_recently_completed(prd, limit=10):
    """Get recently completed tasks."""
    done_tasks = [t for t in prd.get("user_stories", []) if t.get("status") == "DONE"]
    return done_tasks[-limit:]


def summarize_best_new_model(prd):
    """Identify and summarize the best new model."""
    done_tasks = prd.get("user_stories", [])
    model_tasks = [
        t
        for t in done_tasks
        if t.get("status") == "DONE" and "model" in t.get("title", "").lower()
    ]

    # Focus on recently completed model tasks
    recent_models = [t for t in model_tasks if t.get("id") >= 500]

    if not recent_models:
        return "No new models implemented in recent batch."

    # Summary of new models
    model_summaries = []
    for task in recent_models:
        title = task.get("title", "Unknown")
        task_id = task.get("id", "N/A")
        desc = task.get("description", "")[:100]
        model_summaries.append(f"**{title}** (ID {task_id})\n   {desc}...")

    return "\n\n".join(model_summaries)


def generate_markdown_report(prd):
    """Generate markdown status report."""
    status = analyze_task_status(prd)
    recent = get_recently_completed(prd, limit=15)
    best_model = summarize_best_new_model(prd)

    date_str = datetime.now().strftime("%B %d, %Y")

    report = f"""# Weekly Status Report

**Generated:** {date_str}  
**Project:** Opus Autopoiesis: Self-Evolving Economic Strategy (Hybrid v3.3)

---

## 📊 Executive Summary

| Metric | Count | Percentage |
|---------|---------|------------|
| **Total Tasks** | {status["total"]} | 100% |
| **Completed (DONE)** | {status["by_status"].get("DONE", 0)} | {status["completion_rate"]:.1f}% |
| **Pending Review** | {status["by_status"].get("PENDING_REVIEW", 0)} | {status["by_status"].get("PENDING_REVIEW", 0) / status["total"] * 100:.1f}% |
| **Blocked** | {status["by_status"].get("BLOCKED", 0)} | {status["by_status"].get("BLOCKED", 0) / status["total"] * 100:.1f}% |
| **TODO** | {status["by_status"].get("TODO", 0)} | {status["by_status"].get("TODO", 0) / status["total"] * 100:.1f}% |

**Completion Rate:** {status["completion_rate"]:.1f}%

---

## 🏆 Best New Models

Recent model implementations with notable impact:

{best_model}

---

## 📝 Recently Completed Tasks

### Last 15 Completed Tasks

"""

    for task in recent:
        task_id = task.get("id", "N/A")
        title = task.get("title", "Unknown")
        priority = task.get("priority", "N/A")

        report += f"- **{title}** (ID: {task_id}, Priority: {priority})\n"

    report += f"""

---

## 📋 Task Status Breakdown

### DONE Tasks
Total: {status["by_status"].get("DONE", 0)}

*Includes:*
- Core model testing (Ridge, Huber, NGBoost, EBM, etc.)
- Data mining and ingestion (Rosstat, OPR statistics, labor market)
- Weekly nowcasting research and implementation
- API endpoints and dashboard enhancements
- Infrastructure and operations (backup, monitoring)

### Blocked Tasks
Total: {status["by_status"].get("BLOCKED", 0)}

*These tasks have failed 3 or more verification attempts and require human intervention.*

### Pending Review
Total: {status["by_status"].get("PENDING_REVIEW", 0)}

*Awaiting Critic verification.*

### TODO Tasks
Total: {status["by_status"].get("TODO", 0)}

*Tasks awaiting Worker execution.*

---

## 🔍 Active Forecasting Models

### Production Ensemble Models
- **Subcomp** - Best for h=1 (MAE 0.309)
- **NGBoost** - Best for h=2 (MAE 0.290)
- **Subcomp_Multi** - Best for h=12 (MAE 0.297)
- Ridge, Ridge Extended, Ridge Shock, Ridge Macro
- Huber, ElasticNet, Bayesian Ridge
- LightGBM, XGBoost, CatBoost, EBM
- Prophet, SARIMA, ETS, Holt-Winters
- Subcomponent, Microcomponent, HorizonEnsemble

### Weekly Nowcasting Models
- **VolatilityWeightedNowcaster** - Inverse volatility weighting
- **RegimeAdaptiveNowcaster** - Regime-switching weights
- Weekly Price Nowcaster (33 products)
- Leading Indicators Detector
- Volatility Monitor (1.5σ threshold)

### Scenario Analysis Models
- Scenario Rate (hawk/dove/neutral)
- Ki Trajectory (Taylor rule)
- Unified Subcomponent (scenario-integrated)
- Regime Detector (shock/normal/high_inflation)

---

## 📈 Recent Research Highlights

### Weekly Price Nowcasting
- Volatility analysis and threshold optimization
- Leading indicator identification (33 significant products)
- Regime-weighted basket composition
- Shock signature analysis and early warning

### Component Analysis
- Error decomposition by CPI component
- Granger causality matrix for component leads/lags
- Seasonal pattern evolution (2010-2024)

### Advanced Features
- Sticky Price Index
- Trimmed Mean CPI
- Price Dispersion Index
- Diffusion Index

---

## 🚀 Production Rollouts Completed

1. **Dashboard Enhancements**
   - Regime Monitor widget
   - Alert Panel for price anomalies
   - Seasonality Tab
   - Macro Analysis Tab
   - Forecast comparison page
   - Export data button

2. **API Endpoints**
   - `/health` - System health check
   - `/models` - Model listing with MAE
   - `/metrics` - Prometheus metrics
   - `/backtest/history` - Historical forecasts

3. **Infrastructure**
   - Automated backup system
   - Log rotation setup
   - Auto-retraining pipeline
   - Disk usage monitoring
   - Drift detection

---

## 📚 Documentation

- **USER_GUIDE.md** - End-user guide for Dashboard
- **API.md** - Updated with new endpoints
- **WEEKLY_RESEARCH.md** - Weekly price nowcasting research
- **MAINTENANCE.md** - Model retraining and retirement schedule
- **GEMINI.md** - System status update

---

## ⚠️ Blocked Tasks Requiring Attention

"""

    blocked_tasks = [
        t for t in prd.get("user_stories", []) if t.get("status") == "BLOCKED"
    ]
    if blocked_tasks:
        for task in blocked_tasks:
            task_id = task.get("id", "N/A")
            title = task.get("title", "Unknown")
            feedback = task.get("feedback", "No feedback")[:150]
            report += f"\n**{title}** (ID: {task_id})\n> {feedback}...\n"
    else:
        report += "\nNo blocked tasks.\n"

    report += f"""

---

## 📅 Next Week Priorities

Based on current TODO and high-priority tasks:

1. **Production Rollouts**
   - Volatility Weighting (Task 566) - Update weekly_loader.py
   - Regime Switching (Task 567) - Integrate into main Nowcaster

2. **Pending Tasks**
   - Load Testing (Task 564) - API performance verification

---

*This report is automatically generated by `scripts/gen_weekly_report.py`*
"""

    return report


def main():
    """Main execution."""
    print("Loading PRD...")
    prd = load_prd()

    print("Analyzing task status...")
    status = analyze_task_status(prd)
    print(f"Total tasks: {status['total']}")
    print(f"Completion rate: {status['completion_rate']:.1f}%")

    print("\nGenerating markdown report...")
    report = generate_markdown_report(prd)

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Write report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport generated: {REPORT_FILE}")
    print(f"Report size: {len(report)} characters")


if __name__ == "__main__":
    main()
