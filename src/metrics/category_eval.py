import json
import warnings
from typing import Dict, List
from transformers import pipeline

class CategoryEvaluator:
    def __init__(self, model, tokenizer, config: Dict = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.categories_file = self.config.get("prompts_file", "prompts/eval_prompts.jsonl")
        self.max_new_tokens = self.config.get("max_new_tokens", 100)
        self.prompts = self._load_categorized_prompts()
        self.pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            clean_up_tokenization_spaces=False
        )
    
    def _load_categorized_prompts(self) -> Dict[str, List[Dict]]:
        """Load and organize prompts by category."""
        prompts_by_category = {}
        
        try:
            with open(self.categories_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    category = item.get("category", "unknown")
                    
                    if category not in prompts_by_category:
                        prompts_by_category[category] = []
                    
                    prompts_by_category[category].append(item)
        except FileNotFoundError:
            warnings.warn(f"Category file {self.categories_file} not found")
            return {}
        
        return prompts_by_category
    
    def evaluate_by_category(self) -> Dict[str, Dict]:
        """Evaluate model performance by category."""
        results = {}
        
        for category, prompts in self.prompts.items():
            print(f"Evaluating category: {category}")
            category_results = self._evaluate_category(category, prompts)
            results[category] = category_results
        
        return results
    
    def _evaluate_category(self, category: str, prompts: List[Dict]) -> Dict:
        """Evaluate a specific category of prompts."""
        generated_responses = []
        
        for prompt_item in prompts:
            # Format prompt for the model
            formatted_prompt = f"User: {prompt_item['input']}\nAssistant:"
            
            # Generate response
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipe(
                    formatted_prompt,
                    max_new_tokens=self.max_new_tokens,
                    max_length=None,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = result[0]["generated_text"].strip()
            
            generated_responses.append({
                "input": prompt_item["input"],
                "target": prompt_item["target"],
                "generated": generated_text,
                "category": category
            })
        
        # Calculate basic metrics for this category
        return {
            "samples": len(generated_responses),
            "responses": generated_responses
        }
    
    def get_category_samples_for_judge(self, category: str = None, max_samples: int = None) -> List[Dict]:
        """Get samples from specific category or all categories for LLM judge evaluation."""
        samples = []
        
        if category and category in self.prompts:
            # Get samples from specific category
            category_samples = self.prompts[category]
            if max_samples:
                category_samples = category_samples[:max_samples]
            
            for prompt_item in category_samples:
                formatted_prompt = f"User: {prompt_item['input']}\nAssistant:"
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.pipe(
                        formatted_prompt,
                        max_new_tokens=self.max_new_tokens,
                        max_length=None,
                        do_sample=False,
                        return_full_text=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                samples.append({
                    "input": prompt_item["input"],
                    "target": prompt_item["target"],
                    "generated": result[0]["generated_text"].strip(),
                    "category": category
                })
        else:
            # Get samples from all categories
            total_collected = 0
            for cat, prompts in self.prompts.items():
                if max_samples and total_collected >= max_samples:
                    break
                
                cat_limit = max_samples - total_collected if max_samples else len(prompts)
                cat_samples = prompts[:cat_limit]
                
                for prompt_item in cat_samples:
                    if max_samples and total_collected >= max_samples:
                        break
                    
                    formatted_prompt = f"User: {prompt_item['input']}\nAssistant:"
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = self.pipe(
                            formatted_prompt,
                            max_new_tokens=self.max_new_tokens,
                            max_length=None,
                            do_sample=False,
                            return_full_text=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    samples.append({
                        "input": prompt_item["input"],
                        "target": prompt_item["target"],
                        "generated": result[0]["generated_text"].strip(),
                        "category": cat
                    })
                    total_collected += 1
        
        return samples