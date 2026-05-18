import hashlib
from difflib import SequenceMatcher

class DuplicateDetector:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def detect_duplicates(self):
        """Detect exact and near-duplicate samples."""
        prompts = [ex["conversations"][0]["value"] for ex in self.dataset]
        responses = [ex["conversations"][1]["value"] for ex in self.dataset]
        
        # Exact duplicates
        exact_prompt_dupes = len(prompts) - len(set(prompts))
        exact_response_dupes = len(responses) - len(set(responses))
        
        # Near duplicates (similarity > 0.9) - sample for efficiency
        near_dupes = 0
        sample_size = min(500, len(prompts))  # Reduced for performance
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                similarity = SequenceMatcher(None, prompts[i], prompts[j]).ratio()
                if similarity > 0.9:
                    near_dupes += 1
        
        return {
            "exact_prompt_duplicates": exact_prompt_dupes,
            "exact_response_duplicates": exact_response_dupes,
            "near_duplicates_sample": near_dupes,
            "duplicate_rate": exact_prompt_dupes / len(prompts) if prompts else 0
        }
    
    def detect_exact_duplicates(self):
        """Fast exact duplicate detection using hashing."""
        seen_hashes = {}
        duplicates = []
        
        for idx, example in enumerate(self.dataset):
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            
            # Create hash of prompt+response
            content = f"{prompt}|||{response}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash in seen_hashes:
                duplicates.append({
                    "index": idx,
                    "duplicate_of": seen_hashes[content_hash],
                    "prompt": prompt[:100]  # First 100 chars
                })
            else:
                seen_hashes[content_hash] = idx
        
        return duplicates