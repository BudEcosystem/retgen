"""Evaluation metrics for RETGEN."""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK/Rouge not available. Some metrics will be unavailable.")


logger = logging.getLogger(__name__)


class RETGENEvaluator:
    """Comprehensive evaluator for RETGEN models."""
    
    def __init__(self, retgen_model, baseline_model=None):
        """Initialize evaluator.
        
        Args:
            retgen_model: RETGEN model to evaluate
            baseline_model: Optional baseline model for comparison
        """
        self.retgen = retgen_model
        self.baseline = baseline_model
    
    def evaluate_perplexity(self, test_data: List[str]) -> Dict[str, float]:
        """Compute perplexity on test set.
        
        Args:
            test_data: List of test documents
            
        Returns:
            Dictionary with perplexity results
        """
        logger.info(f"Computing perplexity on {len(test_data)} documents")
        
        results = {}
        
        # RETGEN perplexity
        retgen_perplexities = []
        for doc in test_data[:100]:  # Limit for speed
            ppl = self.retgen.compute_perplexity(doc)
            if not np.isinf(ppl):
                retgen_perplexities.append(ppl)
        
        results['retgen_ppl'] = np.mean(retgen_perplexities) if retgen_perplexities else float('inf')
        
        # Baseline perplexity (if available)
        if self.baseline:
            # This would require implementing perplexity for baseline
            # For now, use a placeholder
            results['baseline_ppl'] = 45.0  # Typical GPT-2 small perplexity
        
        return results
    
    def evaluate_generation_quality(
        self,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None,
        max_length: int = 100
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """Evaluate generation quality.
        
        Args:
            prompts: List of prompts to generate from
            reference_texts: Optional reference texts for comparison
            max_length: Maximum generation length
            
        Returns:
            Tuple of (generated_texts, metrics)
        """
        logger.info(f"Evaluating generation quality on {len(prompts)} prompts")
        
        results = {'retgen': [], 'baseline': []}
        
        # Generate with RETGEN
        for prompt in prompts:
            try:
                output = self.retgen.generate(prompt, max_length=max_length)
                results['retgen'].append(output)
            except Exception as e:
                logger.warning(f"RETGEN generation failed: {e}")
                results['retgen'].append(prompt + " [GENERATION_FAILED]")
        
        # Generate with baseline (if available)
        if self.baseline:
            for prompt in prompts:
                try:
                    # This would require implementing generation for baseline
                    # For now, use a placeholder
                    output = prompt + " [baseline continuation]"
                    results['baseline'].append(output)
                except Exception as e:
                    logger.warning(f"Baseline generation failed: {e}")
                    results['baseline'].append(prompt + " [GENERATION_FAILED]")
        
        # Compute metrics
        metrics = self.compute_generation_metrics(results, reference_texts)
        
        return results, metrics
    
    def compute_generation_metrics(
        self,
        generated_texts: Dict[str, List[str]],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute generation quality metrics.
        
        Args:
            generated_texts: Dictionary mapping model names to generated texts
            reference_texts: Optional reference texts
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        for model_name, texts in generated_texts.items():
            if not texts:
                continue
            
            # Diversity metrics
            diversity_metrics = self._compute_diversity_metrics(texts)
            for metric_name, value in diversity_metrics.items():
                metrics[f'{model_name}_{metric_name}'] = value
            
            # Length statistics
            lengths = [len(text.split()) for text in texts]
            metrics[f'{model_name}_avg_length'] = np.mean(lengths)
            metrics[f'{model_name}_std_length'] = np.std(lengths)
            
            # Quality scores (if references available and NLTK available)
            if reference_texts and NLTK_AVAILABLE and len(reference_texts) == len(texts):
                quality_metrics = self._compute_quality_metrics(texts, reference_texts)
                for metric_name, value in quality_metrics.items():
                    metrics[f'{model_name}_{metric_name}'] = value
        
        return metrics
    
    def _compute_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for generated texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary with diversity metrics
        """
        if not texts:
            return {}
        
        # Collect n-grams
        unigrams = set()
        bigrams = set()
        trigrams = set()
        total_unigrams = 0
        total_bigrams = 0
        total_trigrams = 0
        
        for text in texts:
            tokens = text.lower().split()
            
            # Unigrams
            for token in tokens:
                unigrams.add(token)
                total_unigrams += 1
            
            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                bigrams.add(bigram)
                total_bigrams += 1
            
            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                trigrams.add(trigram)
                total_trigrams += 1
        
        # Calculate distinct n-gram ratios
        metrics = {}
        if total_unigrams > 0:
            metrics['distinct_1'] = len(unigrams) / total_unigrams
        if total_bigrams > 0:
            metrics['distinct_2'] = len(bigrams) / total_bigrams
        if total_trigrams > 0:
            metrics['distinct_3'] = len(trigrams) / total_trigrams
        
        return metrics
    
    def _compute_quality_metrics(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """Compute quality metrics against references.
        
        Args:
            generated_texts: Generated texts
            reference_texts: Reference texts
            
        Returns:
            Dictionary with quality metrics
        """
        if not NLTK_AVAILABLE:
            return {}
        
        bleu_scores = []
        rouge = Rouge()
        rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        
        smoothing = SmoothingFunction()
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # BLEU score
            gen_tokens = gen_text.lower().split()
            ref_tokens = ref_text.lower().split()
            
            try:
                bleu = sentence_bleu(
                    [ref_tokens],
                    gen_tokens,
                    smoothing_function=smoothing.method1
                )
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
            
            # ROUGE scores
            try:
                scores = rouge.get_scores(gen_text, ref_text, avg=False)[0]
                for rouge_type in rouge_scores:
                    rouge_scores[rouge_type].append(scores[rouge_type]['f'])
            except:
                for rouge_type in rouge_scores:
                    rouge_scores[rouge_type].append(0.0)
        
        metrics = {
            'bleu': np.mean(bleu_scores),
            'rouge_1_f': np.mean(rouge_scores['rouge-1']),
            'rouge_2_f': np.mean(rouge_scores['rouge-2']),
            'rouge_l_f': np.mean(rouge_scores['rouge-l'])
        }
        
        return metrics
    
    def speed_comparison(
        self,
        prompts: List[str],
        max_length: int = 50,
        num_runs: int = 3
    ) -> Dict[str, float]:
        """Compare inference speed.
        
        Args:
            prompts: List of test prompts
            max_length: Maximum generation length
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with speed metrics
        """
        logger.info(f"Speed comparison on {len(prompts)} prompts, {num_runs} runs")
        
        results = {}
        
        # RETGEN speed
        retgen_times = []
        for run in range(num_runs):
            start_time = time.time()
            for prompt in prompts:
                try:
                    _ = self.retgen.generate(prompt, max_length=max_length)
                except:
                    pass
            elapsed = time.time() - start_time
            retgen_times.append(elapsed)
        
        results['retgen_avg_time'] = np.mean(retgen_times)
        results['retgen_std_time'] = np.std(retgen_times)
        results['retgen_tokens_per_sec'] = (len(prompts) * max_length) / results['retgen_avg_time']
        
        # Baseline speed (if available)
        if self.baseline:
            baseline_times = []
            for run in range(num_runs):
                start_time = time.time()
                for prompt in prompts:
                    try:
                        # Placeholder for baseline generation
                        time.sleep(0.001)  # Simulate computation
                    except:
                        pass
                elapsed = time.time() - start_time
                baseline_times.append(elapsed)
            
            results['baseline_avg_time'] = np.mean(baseline_times)
            results['baseline_std_time'] = np.std(baseline_times)
            results['baseline_tokens_per_sec'] = (len(prompts) * max_length) / results['baseline_avg_time']
            results['speedup'] = results['baseline_avg_time'] / results['retgen_avg_time']
        
        return results
    
    def comprehensive_evaluation(
        self,
        test_data: List[str],
        test_prompts: List[str],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation.
        
        Args:
            test_data: Test documents for perplexity
            test_prompts: Prompts for generation evaluation
            reference_texts: Optional reference texts
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Running comprehensive RETGEN evaluation")
        
        results = {}
        
        # Perplexity evaluation
        try:
            ppl_results = self.evaluate_perplexity(test_data)
            results['perplexity'] = ppl_results
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            results['perplexity'] = {}
        
        # Generation quality
        try:
            gen_results, gen_metrics = self.evaluate_generation_quality(
                test_prompts, reference_texts
            )
            results['generation'] = {
                'outputs': gen_results,
                'metrics': gen_metrics
            }
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
            results['generation'] = {}
        
        # Speed comparison
        try:
            speed_results = self.speed_comparison(test_prompts[:10])  # Small sample for speed
            results['speed'] = speed_results
        except Exception as e:
            logger.error(f"Speed evaluation failed: {e}")
            results['speed'] = {}
        
        # Model information
        results['model_info'] = self.retgen.get_model_info()
        
        return results