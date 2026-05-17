import warnings
from typing import Dict, List
from transformers import pipeline
import json

class PairwiseEvaluator:
    def __init__(self, base_model, finetuned_model, tokenizer, llm_judge=None):
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.tokenizer = tokenizer
        self.llm_judge = llm_judge
        
        # Create pipelines for both models
        self.base_pipe = pipeline(
            "text-generation", 
            model=base_model, 
            tokenizer=tokenizer,
            clean_up_tokenization_spaces=False
        )
        
        self.ft_pipe = pipeline(
            "text-generation", 
            model=finetuned_model, 
            tokenizer=tokenizer,
            clean_up_tokenization_spaces=False
        )
    
    def compare_responses(self, prompts: List[str], max_new_tokens: int = 100) -> Dict[str, float]:
        """Compare base model vs fine-tuned model responses."""
        results = []
        
        for prompt in prompts:
            # Generate responses from both models
            base_response = self._generate_response(self.base_pipe, prompt, max_new_tokens)
            ft_response = self._generate_response(self.ft_pipe, prompt, max_new_tokens)
            
            # Judge the winner
            winner = self._judge_winner(prompt, base_response, ft_response)
            
            results.append({
                "prompt": prompt,
                "base_response": base_response,
                "finetuned_response": ft_response,
                "winner": winner
            })
        
        return self._aggregate_results(results)
    
    def _generate_response(self, pipe, prompt: str, max_new_tokens: int) -> str:
        """Generate response using the given pipeline."""
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipe(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return result[0]["generated_text"].strip()
    
    def _judge_winner(self, prompt: str, base_response: str, ft_response: str) -> str:
        """Determine which response is better."""
        if self.llm_judge:
            return self._llm_judge_winner(prompt, base_response, ft_response)
        else:
            return self._simple_judge_winner(base_response, ft_response)
    
    def _llm_judge_winner(self, prompt: str, base_response: str, ft_response: str) -> str:
        """Use LLM judge to determine winner."""
        judge_prompt = f"""Compare these two responses to the same automotive question.

QUESTION:
{prompt}

RESPONSE A:
{base_response}

RESPONSE B:
{ft_response}

Which response is better? Consider:
- Accuracy and correctness
- Helpfulness and completeness
- Clarity and coherence
- Safety and appropriateness

Respond with JSON:
{{
  "winner": "A" | "B" | "tie",
  "confidence": <1-10>,
  "reasoning": "<explanation>"
}}"""
        
        try:
            # Call the judge API directly
            result = self.llm_judge._call_judge_api(judge_prompt)
            print(f"Judge API result: {result}")
            
            # Handle list response format
            if isinstance(result, list) and len(result) > 0:
                winner_data = result[0]  # First item has winner info
                winner_letter = winner_data.get("winner", "tie").upper()
                
                if winner_letter == "A":
                    return "base"
                elif winner_letter == "B":
                    return "finetuned"
                else:
                    return "tie"
            
            return self._simple_judge_winner(base_response, ft_response)
        except Exception as e:
            print(f"Judge error: {e}")
            return self._simple_judge_winner(base_response, ft_response)
    
    def _simple_judge_winner(self, base_response: str, ft_response: str) -> str:
        """Simple heuristic-based winner determination."""
        # Check for more specific automotive terms in fine-tuned response
        automotive_terms = ['coolant', 'thermostat', 'brake', 'engine', 'vehicle', 'check', 'inspect']
        
        base_score = sum(1 for term in automotive_terms if term.lower() in base_response.lower())
        ft_score = sum(1 for term in automotive_terms if term.lower() in ft_response.lower())
        
        print(f"Scoring - Base: {base_score}, FT: {ft_score}")
        
        if ft_score > base_score:
            return "finetuned"
        elif base_score > ft_score:
            return "base"
        else:
            # If tied on terms, check length (more detailed often better)
            if len(ft_response) > len(base_response) * 1.1:
                return "finetuned"
            elif len(base_response) > len(ft_response) * 1.1:
                return "base"
            else:
                return "tie"
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate pairwise comparison results."""
        total = len(results)
        if total == 0:
            return {}
        
        winners = [r["winner"] for r in results]
        
        ft_wins = winners.count("finetuned")
        base_wins = winners.count("base")
        ties = winners.count("tie")
        
        return {
            "pairwise_ft_win_rate": ft_wins / total,
            "pairwise_base_win_rate": base_wins / total,
            "pairwise_tie_rate": ties / total,
            "pairwise_total_comparisons": total
        }