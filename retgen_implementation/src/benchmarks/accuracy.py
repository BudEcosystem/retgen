"""Accuracy benchmarking for RETGEN."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.retgen import RETGEN
from training.dataset_loader import DatasetLoader
from evaluation.metrics import RETGENEvaluator


logger = logging.getLogger(__name__)


class AccuracyBenchmark:
    """Accuracy benchmark suite for RETGEN."""
    
    def __init__(self, model: RETGEN, baseline_model=None):
        """Initialize accuracy benchmark.
        
        Args:
            model: RETGEN model to benchmark
            baseline_model: Optional baseline model for comparison
        """
        self.model = model
        self.baseline_model = baseline_model
        self.evaluator = RETGENEvaluator(model, baseline_model)
        self.results = {}
    
    def benchmark_perplexity(
        self,
        test_datasets: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Benchmark perplexity on multiple test datasets.
        
        Args:
            test_datasets: Dictionary mapping dataset names to document lists
            
        Returns:
            Dictionary with perplexity results
        """
        logger.info(f"Benchmarking perplexity on {len(test_datasets)} datasets")
        
        results = {}
        
        for dataset_name, test_docs in test_datasets.items():
            logger.info(f"Computing perplexity on {dataset_name}")
            
            try:
                ppl_results = self.evaluator.evaluate_perplexity(test_docs)
                results[dataset_name] = ppl_results
                
                logger.info(f"{dataset_name} RETGEN PPL: {ppl_results.get('retgen_ppl', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Perplexity benchmark failed for {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        self.results['perplexity'] = results
        return results
    
    def benchmark_generation_quality(
        self,
        prompt_sets: Dict[str, List[str]],
        reference_sets: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Benchmark text generation quality.
        
        Args:
            prompt_sets: Dictionary mapping set names to prompt lists
            reference_sets: Optional reference texts for each set
            
        Returns:
            Dictionary with generation quality results
        """
        logger.info(f"Benchmarking generation quality on {len(prompt_sets)} prompt sets")
        
        results = {}
        
        for set_name, prompts in prompt_sets.items():
            logger.info(f"Evaluating generation on {set_name}")
            
            references = reference_sets.get(set_name) if reference_sets else None
            
            try:
                gen_outputs, gen_metrics = self.evaluator.evaluate_generation_quality(
                    prompts, references
                )
                
                results[set_name] = {
                    'metrics': gen_metrics,
                    'sample_outputs': gen_outputs['retgen'][:5]  # Save first 5 examples
                }
                
                # Log key metrics
                if 'retgen_distinct_1' in gen_metrics:
                    logger.info(f"{set_name} Distinct-1: {gen_metrics['retgen_distinct_1']:.3f}")
                if 'retgen_bleu' in gen_metrics:
                    logger.info(f"{set_name} BLEU: {gen_metrics['retgen_bleu']:.3f}")
                
            except Exception as e:
                logger.error(f"Generation quality benchmark failed for {set_name}: {e}")
                results[set_name] = {'error': str(e)}
        
        self.results['generation_quality'] = results
        return results
    
    def benchmark_retrieval_quality(
        self,
        test_patterns: List[str],
        k_values: List[int] = [1, 5, 10, 20, 50]
    ) -> Dict[str, Any]:
        """Benchmark retrieval quality at different k values.
        
        Args:
            test_patterns: List of test pattern texts
            k_values: List of k values to test
            
        Returns:
            Dictionary with retrieval quality results
        """
        logger.info(f"Benchmarking retrieval quality on {len(test_patterns)} patterns")
        
        if not self.model.is_trained:
            logger.warning("Model not trained, skipping retrieval benchmark")
            return {}
        
        results = {
            'k_values': k_values,
            'avg_similarities': [],
            'coverage_rates': []
        }
        
        for k in k_values:
            similarities = []
            coverage_count = 0
            
            for pattern in test_patterns[:100]:  # Limit for speed
                try:
                    # Embed pattern
                    embedding = self.model.trainer.embedder.embed_text(pattern)
                    
                    # Search database
                    search_results = self.model.trainer.database.search(
                        embedding.reshape(1, -1), k=k
                    )
                    
                    if search_results[0]:  # Has results
                        # Get similarities
                        top_similarity = search_results[0][0]['distance']
                        similarities.append(top_similarity)
                        
                        # Count as covered if top similarity > threshold
                        if top_similarity > 0.7:
                            coverage_count += 1
                
                except Exception as e:
                    logger.warning(f"Retrieval failed for pattern: {e}")
                    similarities.append(0.0)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            coverage_rate = coverage_count / len(test_patterns[:100])
            
            results['avg_similarities'].append(avg_similarity)
            results['coverage_rates'].append(coverage_rate)
            
            logger.info(f"k={k}: Avg similarity={avg_similarity:.3f}, Coverage={coverage_rate:.3f}")
        
        self.results['retrieval_quality'] = results
        return results
    
    def benchmark_domain_adaptation(
        self,
        domain_datasets: Dict[str, Tuple[List[str], List[str]]]
    ) -> Dict[str, Any]:
        """Benchmark model performance across different domains.
        
        Args:
            domain_datasets: Dict mapping domain names to (train, test) document lists
            
        Returns:
            Dictionary with domain adaptation results
        """
        logger.info(f"Benchmarking domain adaptation on {len(domain_datasets)} domains")
        
        results = {}
        
        for domain_name, (train_docs, test_docs) in domain_datasets.items():
            logger.info(f"Testing on {domain_name} domain")
            
            try:
                # Create domain-specific model
                domain_model = RETGEN(self.model.config)
                domain_model.train(train_docs)
                
                # Evaluate on domain test set
                domain_evaluator = RETGENEvaluator(domain_model)
                
                # Perplexity
                ppl_results = domain_evaluator.evaluate_perplexity(test_docs[:50])
                
                # Generation quality
                test_prompts = [doc[:50] for doc in test_docs[:20]]  # First 50 chars as prompts
                gen_outputs, gen_metrics = domain_evaluator.evaluate_generation_quality(test_prompts)
                
                results[domain_name] = {
                    'perplexity': ppl_results,
                    'generation_metrics': gen_metrics,
                    'model_size_mb': domain_model.get_size_mb()
                }
                
                logger.info(f"{domain_name}: PPL={ppl_results.get('retgen_ppl', 'N/A'):.1f}")
                
                # Cleanup
                del domain_model
                
            except Exception as e:
                logger.error(f"Domain adaptation benchmark failed for {domain_name}: {e}")
                results[domain_name] = {'error': str(e)}
        
        self.results['domain_adaptation'] = results
        return results
    
    def benchmark_few_shot_learning(
        self,
        base_prompts: List[str],
        few_shot_examples: List[Tuple[str, str]],
        shot_counts: List[int] = [0, 1, 3, 5]
    ) -> Dict[str, Any]:
        """Benchmark few-shot learning capabilities.
        
        Args:
            base_prompts: Base prompts to test
            few_shot_examples: List of (prompt, completion) example pairs
            shot_counts: Number of examples to include (0 = zero-shot)
            
        Returns:
            Dictionary with few-shot learning results
        """
        logger.info("Benchmarking few-shot learning capabilities")
        
        if not self.model.is_trained:
            logger.warning("Model not trained, skipping few-shot benchmark")
            return {}
        
        results = {
            'shot_counts': shot_counts,
            'generation_quality': {},
            'consistency': {}
        }
        
        for shot_count in shot_counts:
            logger.info(f"Testing {shot_count}-shot learning")
            
            shot_results = []
            consistency_scores = []
            
            for base_prompt in base_prompts:
                # Construct few-shot prompt
                if shot_count == 0:
                    full_prompt = base_prompt
                else:
                    example_text = ""
                    for i in range(min(shot_count, len(few_shot_examples))):
                        ex_prompt, ex_completion = few_shot_examples[i]
                        example_text += f"{ex_prompt} {ex_completion}\n"
                    
                    full_prompt = example_text + base_prompt
                
                try:
                    # Generate multiple times to test consistency
                    generations = []
                    for _ in range(3):
                        gen_text = self.model.generate(
                            full_prompt,
                            max_length=50,
                            temperature=0.7
                        )
                        generations.append(gen_text)
                    
                    # Measure consistency (average pairwise similarity)
                    consistency = self._measure_consistency(generations)
                    consistency_scores.append(consistency)
                    
                    # Store first generation for quality analysis
                    shot_results.append(generations[0])
                
                except Exception as e:
                    logger.warning(f"Few-shot generation failed: {e}")
                    shot_results.append("")
                    consistency_scores.append(0.0)
            
            # Compute quality metrics
            if shot_results:
                quality_metrics = self._compute_text_quality(shot_results)
                results['generation_quality'][shot_count] = quality_metrics
            
            # Average consistency
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            results['consistency'][shot_count] = avg_consistency
            
            logger.info(f"{shot_count}-shot: Consistency={avg_consistency:.3f}")
        
        self.results['few_shot_learning'] = results
        return results
    
    def _measure_consistency(self, generations: List[str]) -> float:
        """Measure consistency between multiple generations.
        
        Args:
            generations: List of generated texts
            
        Returns:
            Consistency score (0-1)
        """
        if len(generations) < 2:
            return 1.0
        
        # Simple token overlap measure
        tokenized = [gen.lower().split() for gen in generations]
        
        similarities = []
        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                tokens1, tokens2 = set(tokenized[i]), set(tokenized[j])
                if len(tokens1) == 0 and len(tokens2) == 0:
                    sim = 1.0
                elif len(tokens1) == 0 or len(tokens2) == 0:
                    sim = 0.0
                else:
                    sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_text_quality(self, texts: List[str]) -> Dict[str, float]:
        """Compute basic quality metrics for texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary with quality metrics
        """
        if not texts:
            return {}
        
        # Length statistics
        lengths = [len(text.split()) for text in texts if text]
        avg_length = np.mean(lengths) if lengths else 0.0
        
        # Diversity
        all_tokens = []
        for text in texts:
            if text:
                all_tokens.extend(text.lower().split())
        
        if all_tokens:
            unique_tokens = len(set(all_tokens))
            total_tokens = len(all_tokens)
            diversity = unique_tokens / total_tokens
        else:
            diversity = 0.0
        
        return {
            'avg_length': avg_length,
            'diversity': diversity,
            'valid_generations': len([t for t in texts if t.strip()])
        }
    
    def run_full_benchmark(
        self,
        save_results: bool = True,
        results_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run complete accuracy benchmark suite.
        
        Args:
            save_results: Whether to save results to file
            results_path: Path to save results (optional)
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Running full accuracy benchmark suite")
        
        # Create test datasets
        test_datasets = {
            'sample': DatasetLoader.create_sample_dataset(50)
        }
        
        # Create prompt sets
        prompt_sets = {
            'general': [
                "The future of",
                "Natural language",
                "Machine learning",
                "Deep learning",
                "Text generation"
            ],
            'technical': [
                "The algorithm works by",
                "The neural network",
                "The optimization",
                "The implementation",
                "The evaluation"
            ]
        }
        
        # Perplexity benchmark
        try:
            self.benchmark_perplexity(test_datasets)
            logger.info("✓ Perplexity benchmark completed")
        except Exception as e:
            logger.error(f"Perplexity benchmark failed: {e}")
        
        # Generation quality benchmark
        try:
            self.benchmark_generation_quality(prompt_sets)
            logger.info("✓ Generation quality benchmark completed")
        except Exception as e:
            logger.error(f"Generation quality benchmark failed: {e}")
        
        # Retrieval quality benchmark
        try:
            test_patterns = [
                "the quick brown",
                "natural language processing",
                "machine learning algorithm",
                "deep neural network",
                "artificial intelligence"
            ]
            self.benchmark_retrieval_quality(test_patterns)
            logger.info("✓ Retrieval quality benchmark completed")
        except Exception as e:
            logger.error(f"Retrieval quality benchmark failed: {e}")
        
        # Few-shot learning benchmark
        try:
            few_shot_examples = [
                ("Complete this sentence: The cat", "sat on the mat."),
                ("Complete this sentence: The dog", "ran in the park."),
                ("Complete this sentence: The bird", "flew in the sky.")
            ]
            self.benchmark_few_shot_learning(
                prompt_sets['general'],
                few_shot_examples
            )
            logger.info("✓ Few-shot learning benchmark completed")
        except Exception as e:
            logger.error(f"Few-shot learning benchmark failed: {e}")
        
        # Save results
        if save_results:
            if results_path is None:
                results_path = Path("benchmark_results_accuracy.json")
            
            import json
            with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = self._convert_for_json(self.results)
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved accuracy benchmark results to {results_path}")
        
        return self.results
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj