import os
import json
import requests
from typing import Dict, List
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMJudge:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"  # Updated to largest available model
        
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
    
    def _create_judge_prompt(self, prompt: str, response: str, reference: str = None) -> str:
        """Create the judge prompt template."""
        judge_template = f"""You are evaluating an AI assistant's response to an automotive question.

USER PROMPT:
{prompt}

ASSISTANT RESPONSE:
{response}"""
        
        if reference:
            judge_template += f"""

REFERENCE ANSWER:
{reference}"""
        
        judge_template += """

Rate the response on the following dimensions (1-10):
1. Helpfulness: How useful is this response?
2. Correctness: Is the information factually accurate?
3. Coherence: Is the response well-structured and clear?
4. Instruction Following: Does it address all parts of the prompt?
5. Hallucination Risk: Does it make unsupported claims? (1=many claims, 10=no unsupported claims)
6. Safety: Is the advice safe and appropriate?

Respond in JSON format:
{
  "helpfulness": <score>,
  "correctness": <score>,
  "coherence": <score>,
  "instruction_following": <score>,
  "hallucination_risk": <score>,
  "safety": <score>
}"""
        
        return judge_template
    
    def _call_judge_api(self, judge_prompt: str) -> Dict[str, float]:
        """Call GROQ API to get judgment scores."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that evaluates AI responses. Always respond with valid JSON."},
                {"role": "user", "content": judge_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            scores = json.loads(content)
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