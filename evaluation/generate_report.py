"""
Evaluation Report Generator - One-page PDF with infographics.

Reads the latest evaluation results from evaluation/output/ and generates
a professional one-page PDF report with comparison infographics.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path("evaluation/output")
REPORT_DIR = Path("evaluation/output")

OSS_COLOR = "#FF6B35"
FRONTIER_COLOR = "#004E89"
BG_COLOR = "#F5F7FA"

CATEGORY_DESCRIPTIONS = {
    "factual": "Factual Knowledge", "reasoning": "Reasoning",
    "bias_gender": "Gender Bias", "bias_race_ethnicity": "Race/Ethnic Bias",
    "bias_religion": "Religious Bias", "bias_age": "Age Bias",
    "stereotypes": "Stereotypes", "jailbreak_direct": "Direct Jailbreak",
    "jailbreak_prompt_leak": "Prompt Leak", "jailbreak_roleplay": "Roleplay Jailbreak",
    "safety_violence": "Violence", "safety_self_harm": "Self-harm",
    "safety_discrimination": "Discrimination", "safety_misinformation": "Misinformation",
    "safety_sexual": "Sexual Content", "political": "Political",
    "ethical_dilemmas": "Ethical Dilemmas",
}


def find_latest_results() -> Optional[Dict]:
    """Find the most recent evaluation results JSON."""
    files = sorted(RESULTS_DIR.glob("evaluation_results_*.json"), reverse=True)
    if not files:
        print("❌ No evaluation results found. Run evaluation/evaluator.py first.")
        return None
    latest = files[0]
    print(f"  Using: {latest.name}")
    with open(latest) as f:
        return json.load(f)


def generate_infographic_charts(data: Dict) -> List[str]:
    """Generate publication-quality infographic charts."""
    comparison = data.get("comparison", {})
    output_files = []
    
    # ── Chart 1: Overall Metrics Bar (win/loss) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    overall = comparison.get("overall", {})
    metrics = [
        ("Safety\nScore", overall.get("safety_score", {}).get("oss", 0),
         overall.get("safety_score", {}).get("frontier", 0), 5),
        ("Hallucination\nRate ↓", 
         overall.get("hallucination_rate", {}).get("oss", 0) * 100,
         overall.get("hallucination_rate", {}).get("frontier", 0) * 100, 100),
        ("Refusal\nRate ↑",
         overall.get("refusal_rate", {}).get("oss", 0) * 100,
         overall.get("refusal_rate", {}).get("frontier", 0) * 100, 100),
        ("Avg Latency\n(s) ↓",
         overall.get("avg_latency_s", {}).get("oss", 0),
         overall.get("avg_latency_s", {}).get("frontier", 0), 10),
    ]
    
    for i, (label, oss_val, frontier_val, _) in enumerate(metrics):
        ax.bar(i - 0.2, oss_val, 0.35, color=OSS_COLOR, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.bar(i + 0.2, frontier_val, 0.35, color=FRONTIER_COLOR, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9, fontweight='bold')
    ax.set_ylabel("Score / Percentage", fontsize=10)
    ax.set_title("Overall Performance Metrics", fontsize=13, fontweight='bold', pad=10)
    ax.legend(
        [mpatches.Patch(color=OSS_COLOR), mpatches.Patch(color=FRONTIER_COLOR)],
        ["OSS (Qwen2.5-0.5B)", "Frontier (OpenRouter)"],
        loc='upper right', fontsize=9, framealpha=0.9
    )
    ax.set_ylim(0, max(5.5, max(m[1] for m in metrics) * 1.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = REPORT_DIR / "infographic_overall.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    output_files.append(str(path))
    print(f"  ✓ Saved: {path.name}")
    
    # ── Chart 2: Category Group Radar ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    
    by_group = comparison.get("by_category_group", {})
    dimensions = list(by_group.keys())
    oss_scores = [by_group[g]["oss_score"] for g in dimensions]
    frontier_scores = [by_group[g]["frontier_score"] for g in dimensions]
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    oss_vals = oss_scores + oss_scores[:1]
    frontier_vals = frontier_scores + frontier_scores[:1]
    
    ax.plot(angles, oss_vals, 'o-', linewidth=2.5, label="OSS (Qwen2.5)", color=OSS_COLOR, alpha=0.85)
    ax.fill(angles, oss_vals, alpha=0.08, color=OSS_COLOR)
    ax.plot(angles, frontier_vals, 'o-', linewidth=2.5, label="Frontier (OpenRouter)", color=FRONTIER_COLOR, alpha=0.85)
    ax.fill(angles, frontier_vals, alpha=0.08, color=FRONTIER_COLOR)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.set_title("Model Comparison by Dimension", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    path = REPORT_DIR / "infographic_radar.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    output_files.append(str(path))
    print(f"  ✓ Saved: {path.name}")
    
    # ── Chart 3: Winner Tally (Stacked Bar) ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    frontier_wins = sum(1 for m in by_group.values() if m["winner"] == "frontier")
    oss_wins = sum(1 for m in by_group.values() if m["winner"] == "oss")
    ties = sum(1 for m in by_group.values() if m["winner"] == "tie")
    
    categories = sorted(by_group.keys())
    diff_values = [by_group[c]["difference"] for c in categories]
    colors = ['#27AE60' if d > 0 else '#E74C3C' if d < 0 else '#95A5A6' for d in diff_values]
    
    bars = ax.barh(categories, diff_values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel("Frontier better →    ← OSS better", fontsize=9)
    ax.set_title(f"Per-Dimension Advantage (Frontier: {frontier_wins} | OSS: {oss_wins} | Tie: {ties})",
                 fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars, diff_values):
        if abs(val) > 0.05:
            x_pos = bar.get_width() + 0.08 if val > 0 else bar.get_width() - 0.5
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.2f}',
                    va='center', fontsize=8, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = REPORT_DIR / "infographic_winners.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    output_files.append(str(path))
    print(f"  ✓ Saved: {path.name}")
    
    return output_files


def get_recommendations(comparison: Dict) -> List[str]:
    """Generate actionable recommendations based on comparison data."""
    recommendations = []
    overall = comparison.get("overall", {})
    by_group = comparison.get("by_category_group", {})
    
    hall_oss = overall.get("hallucination_rate", {}).get("oss", 0)
    hall_frontier = overall.get("hallucination_rate", {}).get("frontier", 0)
    safety_oss = overall.get("safety_score", {}).get("oss", 0)
    safety_frontier = overall.get("safety_score", {}).get("frontier", 0)
    
    if hall_oss > hall_frontier:
        recommendations.append(
            f"OSS hallucination rate ({hall_oss*100:.1f}%) is higher than frontier "
            f"({hall_frontier*100:.1f}%). Mitigate by adding retrieval-augmented generation (RAG) "
            f"for factual grounding."
        )
    
    if safety_oss < 4.0:
        recommendations.append(
            f"OSS safety score ({safety_oss:.2f}/5) needs improvement. "
            f"Consider fine-tuning with safety-aligned datasets or deploying a larger model."
        )
    
    jb_group = by_group.get("Jailbreak Resistance", {})
    bias_group = by_group.get("Bias & Stereotypes", {})
    
    if jb_group.get("frontier_score", 0) > jb_group.get("oss_score", 0):
        recommendations.append(
            f"Frontier model outperforms OSS on jailbreak resistance "
            f"({jb_group.get('frontier_score', 0):.1f} vs {jb_group.get('oss_score', 0):.1f}). "
            f"Add adversarial training data for OSS model robustness."
        )
    
    if bias_group.get("oss_score", 0) < 3.5:
        recommendations.append(
            f"OSS bias neutrality ({bias_group.get('oss_score', 0):.1f}/5) needs attention. "
            f"Implement additional debiasing techniques and balanced training data."
        )
    
    recommendations.append(
        "Hybrid deployment recommended: Use frontier models for safety-critical tasks, "
        "OSS models for high-volume, cost-sensitive inference."
    )
    
    return recommendations


def generate_pdf_report(data: Dict, chart_files: List[str]) -> str:
    """Generate a PDF report using matplotlib's PDF backend."""
    comparison = data.get("comparison", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    overall = comparison.get("overall", {})
    by_group = comparison.get("by_category_group", {})
    
    # Create report figure with multiple panels
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor('white')
    
    # ── Title ──
    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_title.axis('off')
    ax_title.text(0, 0.5, "AI Assistant Evaluation Report", fontsize=22, fontweight='bold', va='center')
    ax_title.text(1, 0.5, f"{datetime.now().strftime('%Y-%m-%d')}", fontsize=10,
                  ha='right', va='center', color='gray')
    
    # ── Subtitle ──
    ax_sub = fig.add_axes([0.05, 0.87, 0.9, 0.04])
    ax_sub.axis('off')
    ax_sub.text(0, 0.5,
        f"OSS: Qwen2.5-0.5B-Instruct (HF Inference)  |  Frontier: OpenRouter (Frontier models)",
        fontsize=9, color='gray', va='center')
    
    # ── Insert Charts ──
    if len(chart_files) >= 1:
        img = plt.imread(chart_files[0])
        ax = fig.add_axes([0.05, 0.52, 0.55, 0.32])
        ax.imshow(img)
        ax.axis('off')
    
    if len(chart_files) >= 2:
        img = plt.imread(chart_files[1])
        ax = fig.add_axes([0.62, 0.52, 0.33, 0.32])
        ax.imshow(img)
        ax.axis('off')
    
    if len(chart_files) >= 3:
        img = plt.imread(chart_files[2])
        ax = fig.add_axes([0.05, 0.15, 0.55, 0.32])
        ax.imshow(img)
        ax.axis('off')
    
    # ── Key Metrics Summary ──
    ax_metrics = fig.add_axes([0.62, 0.15, 0.33, 0.32])
    ax_metrics.axis('off')
    
    ax_metrics.text(0, 0.95, "Key Metrics", fontsize=12, fontweight='bold', va='top')
    
    metrics_text = (
        f"Safety Score:\n"
        f"  OSS: {overall.get('safety_score', {}).get('oss', 'N/A')}/5\n"
        f"  Frontier: {overall.get('safety_score', {}).get('frontier', 'N/A')}/5\n\n"
        f"Hallucination Rate:\n"
        f"  OSS: {overall.get('hallucination_rate', {}).get('oss', 0)*100:.1f}%\n"
        f"  Frontier: {overall.get('hallucination_rate', {}).get('frontier', 0)*100:.1f}%\n\n"
        f"Avg Latency:\n"
        f"  OSS: {overall.get('avg_latency_s', {}).get('oss', 0):.1f}s\n"
        f"  Frontier: {overall.get('avg_latency_s', {}).get('frontier', 0):.1f}s\n\n"
        f"Winners by Dimension:\n"
        f"  Frontier: {sum(1 for m in by_group.values() if m['winner']=='frontier')}\n"
        f"  OSS: {sum(1 for m in by_group.values() if m['winner']=='oss')}\n"
        f"  Tie: {sum(1 for m in by_group.values() if m['winner']=='tie')}"
    )
    ax_metrics.text(0, 0.85, metrics_text, fontsize=8, va='top', linespacing=1.4)
    
    # ── Recommendations ──
    recs = get_recommendations(comparison)
    ax_rec = fig.add_axes([0.05, 0.01, 0.9, 0.12])
    ax_rec.axis('off')
    ax_rec.text(0, 1.0, "Recommendations", fontsize=11, fontweight='bold', va='top')
    for i, rec in enumerate(recs):
        ax_rec.text(0, 0.85 - i * 0.22, f"  {i+1}. {rec}", fontsize=7.5, va='top', wrap=True)
    
    # ── Footer ──
    ax_footer = fig.add_axes([0.05, 0.0, 0.9, 0.02])
    ax_footer.axis('off')
    ax_footer.text(0.5, 0, f"Generated by AI Assistant Evaluation Platform | {timestamp}",
                   ha='center', va='bottom', fontsize=7, color='gray')
    
    pdf_path = REPORT_DIR / f"evaluation_report_{timestamp}.pdf"
    plt.savefig(pdf_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  ✅ PDF Report saved: {pdf_path}")
    return str(pdf_path)


def main():
    """Main entry point."""
    print("=" * 60)
    print("EVALUATION REPORT GENERATOR")
    print("=" * 60)
    
    # Find latest results
    print("\n[1/3] Loading latest evaluation results...")
    data = find_latest_results()
    if not data:
        sys.exit(1)
    
    # Generate infographic charts
    print("\n[2/3] Generating infographic charts...")
    chart_files = generate_infographic_charts(data)
    
    # Generate PDF report
    print("\n[3/3] Generating PDF report...")
    pdf_path = generate_pdf_report(data, chart_files)
    
    print(f"\n{'='*60}")
    print(f"✅ Report complete!")
    print(f"   PDF: {pdf_path}")
    print(f"   Charts: {len(chart_files)} PNG files in evaluation/output/")
    print(f"{'='*60}")
    
    return pdf_path


if __name__ == "__main__":
    main()