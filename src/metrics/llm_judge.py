import os
import json
import requests
from typing import Dict, List
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMJudge:
    def __init__(self, config: Dict = None, api_key: str = None):
        self.config = config or {}
        self.api_key = api_key or os.getenv(self.config.get("api_key_env", "GROQ_API"))
        self.base_url = self.config.get("api_base_url", "https://api.groq.com/openai/v1/chat/completions")
        self.model = self.config.get("model", "llama-3.3-70b-versatile")
        self.max_tokens = self.config.get("max_new_tokens", 300)
        self.temperature = self.config.get("temperature", 0.1)
        self.timeout = self.config.get("timeout", 30)
        self.prompts_dir = self.config.get("prompts_dir", "src/prompts")
        
        # Load prompt templates
        self.system_prompt = self._load_prompt("judge_system.txt")
        self.judge_template = self._load_prompt("judge_template.txt")
        self.reference_template = self._load_prompt("reference_section.txt")
        
    def evaluate_response(self, prompt: str, response: str, reference: str = None) -> Dict[str, float]:
        """Evaluate a single response using LLM-as-a-Judge."""
        judge_prompt = self._create_judge_prompt(prompt, response, reference)
        
        try:
            scores = self._call_judge_api(judge_prompt)
            return scores
        except Exception as e:
            warnings.warn(f"LLM Judge evaluation failed: {e}")
            return self._default_scores()
    
    def evaluate_batch(self, samples: List[Dict]) -> Dict[str, float]:
        """Evaluate multiple samples and return aggregated scores."""
        all_scores = []
        
        for sample in samples:
            scores = self.evaluate_response(
                prompt=sample.get("input", ""),
                response=sample.get("generated", ""),
                reference=sample.get("target", "")
            )
            all_scores.append(scores)
        
        return self._aggregate_scores(all_scores)
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        try:
            with open(os.path.join(self.prompts_dir, filename), 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            warnings.warn(f"Prompt file {filename} not found, using fallback")
            return ""
    
    def _create_judge_prompt(self, prompt: str, response: str, reference: str = None) -> str:
        """Create the judge prompt template."""
        reference_section = ""
        if reference:
            reference_section = self.reference_template.format(reference=reference)
        
        return self.judge_template.format(
            prompt=prompt,
            response=response,
            reference_section=reference_section
        )
    
    def _call_judge_api(self, judge_prompt: str) -> Dict[str, float]:
        """Call GROQ API to get judgment scores."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": judge_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            scores = json.loads(content)
            # Handle case where API returns a list instead of dict
            if isinstance(scores, list):
                print(f"API returned list: {scores}")
                return self._default_scores()
            return {k: float(v) for k, v in scores.items() if k != "reasoning" and isinstance(v, (int, float))}
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            return self._parse_fallback_scores(content)
    
    def _parse_fallback_scores(self, content: str) -> Dict[str, float]:
        """Fallback parsing if JSON response is malformed."""
        scores = {}
        dimensions = ["helpfulness", "correctness", "coherence", "instruction_following", "hallucination_risk", "safety"]
        
        for dim in dimensions:
            try:
                # Look for patterns like "helpfulness": 8 or "helpfulness: 8"
                import re
                pattern = rf'"{dim}"?\s*:\s*(\d+(?:\.\d+)?)'
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    scores[dim] = float(match.group(1))
                else:
                    scores[dim] = 5.0  # Default middle score
            except:
                scores[dim] = 5.0
        
        return scores
    
    def _default_scores(self) -> Dict[str, float]:
        """Return default scores when API fails."""
        return {
            "helpfulness": 5.0,
            "correctness": 5.0,
            "coherence": 5.0,
            "instruction_following": 5.0,
            "hallucination_risk": 5.0,
            "safety": 5.0
        }
    
    def _aggregate_scores(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate scores across multiple samples."""
        if not all_scores:
            return self._default_scores()
        
        dimensions = all_scores[0].keys()
        aggregated = {}
        
        for dim in dimensions:
            scores = [sample[dim] for sample in all_scores if dim in sample]
            aggregated[f"judge_{dim}"] = sum(scores) / len(scores) if scores else 5.0
        
        return aggregated