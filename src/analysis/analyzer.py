from pathlib import Path
from src.utils.logger import setup_logger
from .statistics import StatisticsAnalyzer
from .quality import QualityAnalyzer
from .duplicates import DuplicateDetector
from .reporter import AnalysisReporter

logger = setup_logger(__name__)

class DatasetAnalyzer:
    def __init__(self, dataset, tokenizer, output_dir="output/data_analysis"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.stats_analyzer = StatisticsAnalyzer(dataset, tokenizer)
        self.quality_analyzer = QualityAnalyzer(dataset)
        self.duplicate_detector = DuplicateDetector(dataset)
        self.reporter = AnalysisReporter(self.output_dir)
    
    def analyze_all(self):
        """Run full analysis suite."""
        logger.info("Starting comprehensive dataset analysis...")
        
        stats = {
            "basic": self.stats_analyzer.basic_statistics(),
            "token_distribution": self.stats_analyzer.token_distribution(),
            "length_analysis": self.stats_analyzer.length_analysis(),
            "duplicates": self.duplicate_detector.detect_duplicates(),
            "quality_flags": self.quality_analyzer.quality_checks(),
            "quality_scores": self.quality_analyzer.quality_scoring()
        }
        
        self.reporter.save_report(stats)
        logger.info(f"Dataset analysis completed. Results saved to {self.output_dir}")
        
        return stats