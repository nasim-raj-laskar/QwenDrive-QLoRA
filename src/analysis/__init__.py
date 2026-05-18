from .analyzer import DatasetAnalyzer
from .versioning import DatasetVersioner
from .duplicates import DuplicateDetector

# Convenience function for backward compatibility
def detect_exact_duplicates(dataset):
    """Fast exact duplicate detection using hashing."""
    detector = DuplicateDetector(dataset)
    return detector.detect_exact_duplicates()

__all__ = ['DatasetAnalyzer', 'DatasetVersioner', 'detect_exact_duplicates']