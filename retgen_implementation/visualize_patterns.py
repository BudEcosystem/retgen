#!/usr/bin/env python3
"""Visualize RETGEN patterns and retrieval process."""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
import random
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class PatternVisualizer:
    """Visualize patterns in the RETGEN database."""
    
    def __init__(self, model_path: str, index_dir: str, sample_size: int = 5000):
        """Initialize visualizer with model and patterns."""
        print(f"Loading model from {model_path}...")
        
        # Load model metadata
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.total_patterns = self.model_data['total_patterns']
        self.num_shards = len(self.model_data.get('shard_indices', []))
        
        print(f"Model: {self.total_patterns:,} patterns, {self.num_shards} shards")
        
        # Initialize encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load sample patterns for visualization
        self.index_dir = Path(index_dir)
        self.load_sample_patterns(sample_size)
        
    def load_sample_patterns(self, sample_size: int):
        """Load a sample of patterns for visualization."""
        print(f"Loading {sample_size} sample patterns...")
        
        # Load from middle shard for typical patterns
        shard_files = sorted(self.index_dir.glob("shard_*.faiss"))
        if not shard_files:
            raise ValueError("No shard files found!")
        
        # Load shard 25 (middle of training)
        shard_idx = min(25, len(shard_files)-1)
        shard_file = shard_files[shard_idx]
        meta_path = str(shard_file).replace('.faiss', '_meta.pkl')
        
        print(f"Loading from {shard_file.name}")
        
        # Load index and metadata
        self.index = faiss.read_index(str(shard_file))
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            all_patterns = meta['patterns']
            all_continuations = meta['continuations']
        
        # Random sample
        indices = random.sample(range(len(all_patterns)), min(sample_size, len(all_patterns)))
        self.patterns = [all_patterns[i] for i in indices]
        self.continuations = [all_continuations[i] for i in indices]
        
        # Get embeddings for sampled patterns
        print("Computing embeddings...")
        with torch.no_grad():
            self.embeddings = self.encoder.encode(
                self.patterns,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        print(f"Loaded {len(self.patterns)} patterns with {self.embeddings.shape[1]}-dim embeddings")
    
    def reduce_dimensions(self, method='tsne', n_components=2):
        """Reduce embedding dimensions for visualization."""
        print(f"Reducing dimensions using {method.upper()}...")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(self.embeddings)
            variance_explained = reducer.explained_variance_ratio_
            print(f"PCA variance explained: {variance_explained.sum():.2%}")
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
            reduced = reducer.fit_transform(self.embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return reduced
    
    def analyze_patterns(self):
        """Analyze pattern statistics."""
        # Pattern lengths
        pattern_lengths = [len(p.split()) for p in self.patterns]
        
        # Continuation frequency
        cont_counter = Counter(self.continuations)
        
        # Pattern diversity
        unique_patterns = len(set(self.patterns))
        unique_continuations = len(set(self.continuations))
        
        stats = {
            'total_patterns': len(self.patterns),
            'unique_patterns': unique_patterns,
            'unique_continuations': unique_continuations,
            'pattern_diversity': unique_patterns / len(self.patterns),
            'continuation_diversity': unique_continuations / len(self.patterns),
            'avg_pattern_length': np.mean(pattern_lengths),
            'max_pattern_length': max(pattern_lengths),
            'min_pattern_length': min(pattern_lengths),
            'top_continuations': cont_counter.most_common(10)
        }
        
        return stats
    
    def visualize_embeddings(self, query: str = None):
        """Create interactive visualization of pattern embeddings."""
        print("Creating embedding visualization...")
        
        # Reduce dimensions
        coords_2d = self.reduce_dimensions(method='tsne', n_components=2)
        coords_3d = self.reduce_dimensions(method='pca', n_components=3)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('2D t-SNE Projection', '3D PCA Projection', 
                          'Pattern Length Distribution', 'Top Continuations'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Prepare colors based on pattern length
        pattern_lengths = [len(p.split()) for p in self.patterns]
        
        # Add query point if provided
        if query:
            query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            
            # Find nearest neighbors
            k = 20
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Highlight query and neighbors
            colors = ['lightgray'] * len(self.patterns)
            sizes = [5] * len(self.patterns)
            
            for idx in indices[0]:
                if idx < len(colors):
                    colors[idx] = 'red'
                    sizes[idx] = 10
            
            # Add query point (project to same space)
            # For visualization, we'll just highlight the nearest neighbor
            query_text = f"Query: '{query}'"
        else:
            colors = pattern_lengths
            sizes = [5] * len(self.patterns)
            query_text = "All Patterns"
        
        # 1. 2D t-SNE scatter
        hover_texts = [f"Pattern: {p}<br>Continuation: {c}<br>Length: {l}" 
                      for p, c, l in zip(self.patterns, self.continuations, pattern_lengths)]
        
        fig.add_trace(
            go.Scatter(
                x=coords_2d[:, 0],
                y=coords_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Pattern Length" if not query else "Similarity")
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                name=query_text
            ),
            row=1, col=1
        )
        
        # 2. 3D PCA scatter
        fig.add_trace(
            go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=pattern_lengths,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                name='Patterns'
            ),
            row=1, col=2
        )
        
        # 3. Pattern length histogram
        fig.add_trace(
            go.Histogram(
                x=pattern_lengths,
                nbinsx=20,
                marker_color='rgba(102, 126, 234, 0.7)',
                name='Pattern Lengths'
            ),
            row=2, col=1
        )
        
        # 4. Top continuations bar chart
        cont_counter = Counter(self.continuations)
        top_conts = cont_counter.most_common(15)
        
        fig.add_trace(
            go.Bar(
                x=[c[1] for c in top_conts],
                y=[c[0] for c in top_conts],
                orientation='h',
                marker_color='rgba(118, 75, 162, 0.7)',
                name='Frequency'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"RETGEN Pattern Visualization<br><sub>{len(self.patterns):,} patterns from shard</sub>",
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            height=900,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
        fig.update_xaxes(title_text="Pattern Length (words)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def visualize_retrieval_flow(self, query: str, k: int = 10):
        """Visualize the retrieval process for a query."""
        print(f"\nVisualizing retrieval for: '{query}'")
        
        # Encode query
        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Search in index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Get retrieved patterns
        retrieved_patterns = []
        retrieved_continuations = []
        similarities = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.patterns):
                retrieved_patterns.append(self.patterns[idx])
                retrieved_continuations.append(self.continuations[idx])
                similarities.append(1.0 / (1.0 + float(dist)))
        
        # Create Sankey diagram for retrieval flow
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[query] + retrieved_patterns + retrieved_continuations,
                color=["blue"] + ["lightblue"]*len(retrieved_patterns) + ["lightgreen"]*len(retrieved_continuations)
            ),
            link=dict(
                source=[0]*len(retrieved_patterns) + list(range(1, len(retrieved_patterns)+1)),
                target=list(range(1, len(retrieved_patterns)+1)) + list(range(len(retrieved_patterns)+1, len(retrieved_patterns)+len(retrieved_continuations)+1)),
                value=similarities + similarities,
                color=["rgba(102, 126, 234, 0.4)"]*len(retrieved_patterns) + ["rgba(76, 175, 80, 0.4)"]*len(retrieved_continuations)
            )
        )])
        
        fig.update_layout(
            title=f"Retrieval Flow for: '{query}'",
            font_size=10,
            height=600
        )
        
        return fig, retrieved_patterns, retrieved_continuations, similarities
    
    def create_visualization_report(self, queries: list = None):
        """Create comprehensive visualization report."""
        if queries is None:
            queries = [
                "The future of",
                "Scientists have discovered",
                "In recent years",
                "The president",
                "Climate change"
            ]
        
        # Analyze patterns
        stats = self.analyze_patterns()
        
        print("\n" + "="*60)
        print("PATTERN DATABASE STATISTICS")
        print("="*60)
        print(f"Total Patterns Sampled: {stats['total_patterns']:,}")
        print(f"Unique Patterns: {stats['unique_patterns']:,} ({stats['pattern_diversity']:.1%})")
        print(f"Unique Continuations: {stats['unique_continuations']:,} ({stats['continuation_diversity']:.1%})")
        print(f"Avg Pattern Length: {stats['avg_pattern_length']:.1f} words")
        print(f"Pattern Length Range: {stats['min_pattern_length']}-{stats['max_pattern_length']} words")
        
        print("\nTop 10 Continuations:")
        for cont, count in stats['top_continuations']:
            print(f"  '{cont}': {count} occurrences")
        
        # Create main visualization
        fig_main = self.visualize_embeddings()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RETGEN Pattern Visualization Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background: #f5f5f5;
                }}
                h1, h2 {{
                    color: #667eea;
                }}
                .stats {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .stat-item {{
                    background: #f9f9f9;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .query-section {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .retrieval-result {{
                    background: #f0f0f0;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    border-left: 3px solid #667eea;
                }}
            </style>
        </head>
        <body>
            <h1>RETGEN Pattern Visualization Report</h1>
            
            <div class="stats">
                <h2>Database Statistics</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value">{self.total_patterns:,}</div>
                        <div class="stat-label">Total Patterns</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{self.num_shards}</div>
                        <div class="stat-label">Index Shards</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{stats['pattern_diversity']:.1%}</div>
                        <div class="stat-label">Pattern Diversity</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{stats['avg_pattern_length']:.1f}</div>
                        <div class="stat-label">Avg Pattern Length</div>
                    </div>
                </div>
            </div>
            
            <div id="mainViz"></div>
        """
        
        # Add query-specific visualizations
        for query in queries:
            fig_query = self.visualize_embeddings(query=query)
            fig_flow, patterns, continuations, sims = self.visualize_retrieval_flow(query, k=5)
            
            html_content += f"""
            <div class="query-section">
                <h2>Query: "{query}"</h2>
                <div id="viz_{query.replace(' ', '_')}"></div>
                <div id="flow_{query.replace(' ', '_')}"></div>
                <h3>Top Retrieved Patterns:</h3>
            """
            
            for p, c, s in zip(patterns[:5], continuations[:5], sims[:5]):
                html_content += f"""
                <div class="retrieval-result">
                    <strong>Pattern:</strong> "{p}" → <strong>"{c}"</strong> 
                    <span style="float: right; color: #667eea;">Similarity: {s:.3f}</span>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += """
            <script>
        """
        
        # Add plotly scripts
        html_content += f"Plotly.newPlot('mainViz', {fig_main.to_json()});\n"
        
        for query in queries:
            fig_query = self.visualize_embeddings(query=query)
            fig_flow, _, _, _ = self.visualize_retrieval_flow(query, k=5)
            html_content += f"Plotly.newPlot('viz_{query.replace(' ', '_')}', {fig_query.to_json()});\n"
            html_content += f"Plotly.newPlot('flow_{query.replace(' ', '_')}', {fig_flow.to_json()});\n"
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = "pattern_visualization_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✅ Visualization report saved to: {report_path}")
        print("Open this file in your browser to explore the interactive visualizations!")
        
        return stats


def main():
    """Main visualization function."""
    print("="*60)
    print("RETGEN PATTERN VISUALIZATION")
    print("="*60)
    
    # Initialize visualizer
    visualizer = PatternVisualizer(
        model_path="models/retgen_memory_optimized_final.pkl",
        index_dir="models/index_shards",
        sample_size=5000
    )
    
    # Create visualization report
    stats = visualizer.create_visualization_report(
        queries=[
            "The future of artificial intelligence",
            "Climate change is",
            "Scientists have discovered",
            "In the next decade",
            "The most important"
        ]
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Check if plotly is installed
    try:
        import plotly
    except ImportError:
        print("Installing plotly...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "plotly", "scikit-learn"])
    
    main()