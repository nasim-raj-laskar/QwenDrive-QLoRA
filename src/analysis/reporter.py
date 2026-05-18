import json

class AnalysisReporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def save_report(self, stats):
        """Save analysis report to JSON and text."""
        # JSON report
        json_path = self.output_dir / "dataset_analysis.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Human-readable text report
        txt_path = self.output_dir / "dataset_analysis.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("BASIC STATISTICS:\n")
            f.write(f"  Total samples: {stats['basic']['total_samples']}\n")
            f.write(f"  Unique prompts: {stats['basic']['unique_prompts']}\n")
            f.write(f"  Unique responses: {stats['basic']['unique_responses']}\n\n")
            
            f.write("TOKEN DISTRIBUTION:\n")
            f.write(f"  Prompt tokens (mean): {stats['token_distribution']['prompt_tokens']['mean']:.1f}\n")
            f.write(f"  Response tokens (mean): {stats['token_distribution']['response_tokens']['mean']:.1f}\n")
            f.write(f"  Total tokens (mean): {stats['token_distribution']['total_tokens']['mean']:.1f}\n\n")
            
            f.write("QUALITY FLAGS:\n")
            for flag, count in stats['quality_flags'].items():
                f.write(f"  {flag}: {count}\n")
            
            f.write("\nDUPLICATES:\n")
            f.write(f"  Exact prompt duplicates: {stats['duplicates']['exact_prompt_duplicates']}\n")
            f.write(f"  Duplicate rate: {stats['duplicates']['duplicate_rate']:.2%}\n")
            
            f.write("\nQUALITY SCORES:\n")
            f.write(f"  Mean quality score: {stats['quality_scores']['mean_score']:.1f}/100\n")
            f.write(f"  Low quality samples: {stats['quality_scores']['low_quality_count']}\n")