# RETGEN WikiText-103 Model

## Model Details

- **Model Type**: RETGEN (Retrieval-Enhanced Text Generation)
- **Training Dataset**: WikiText-103
- **Training Date**: 2025-07-22 02:48:31
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2
- **Embedding Dimension**: 768

## Training Configuration

```json
{
  "min_pattern_length": 1,
  "max_pattern_length": 10,
  "pattern_stride": 1,
  "min_pattern_frequency": 2,
  "resolutions": [
    1,
    2,
    3,
    5,
    8
  ],
  "resolution_weights": null,
  "embedding_dim": 768,
  "embedding_model": "sentence-transformers/all-mpnet-base-v2",
  "normalize_embeddings": true,
  "max_sequence_length": 512,
  "use_local_context": true,
  "use_global_context": true,
  "use_positional_encoding": true,
  "positional_encoding_dim": 128,
  "retrieval_k": 100,
  "temperature": 1.0,
  "similarity_metric": "cosine",
  "index_type": "Flat",
  "nprobe": 10,
  "use_gpu": false,
  "max_index_size": 10000000,
  "metadata_backend": "lmdb",
  "compression": true,
  "max_generation_length": 100,
  "beam_size": 1,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.2,
  "length_penalty": 1.0,
  "batch_size": 512,
  "validation_split": 0.1,
  "checkpoint_interval": 10000,
  "device": "cuda",
  "num_workers": 0,
  "prefetch_factor": 2,
  "log_level": "INFO",
  "log_interval": 100
}
```

## Training Metrics

- **Training Documents**: 100
- **Validation Documents**: 1,728
- **Total Patterns**: 16,855
- **Model Size**: 60.9 MB
- **Training Time**: 0.9 minutes

## Pattern Extraction Details

- **Resolutions**: [1, 2, 3, 5, 8]
- **Min Pattern Frequency**: 2
- **Max Pattern Length**: 10

## Retrieval Configuration

- **Retrieval K**: 100
- **Similarity Metric**: cosine
- **Index Type**: Flat

## Generation Samples

**Prompt**: The history of artificial intelligence
**Generated**: The history of artificial intelligenceheintheinfahebarkertheexceptnationalthechildkurtperhapstheasciourthechristmasinaastheabarkerinbarkerthevalthethethethe75shakespearethethisthevalduringthethe75christmasthetheforthechristmas

**Prompt**: Natural language processing is
**Generated**: Natural language processing iscithebarkerltgroundsaasthe75asexceptthetheastheona2exceptchristmasthereligiouskurtthelttheingroundsthisinsideaininheduringin"most75exceptbarkerastheperhapstheinunlikebarkerunlikeon

**Prompt**: Machine learning algorithms
**Generated**: Machine learning algorithmsduringguardiantheinunlikebarkerpicturesqueltshakespearebarkerthe75perhapstheaduringthisthisbarkertheintheontheasnationalthenationalthe2in23thekateasinintwointheinthebarkerbarkerperhapsthetheperhapsfabarker

**Prompt**: Deep neural networks have
**Generated**: Deep neural networks havebarkerltshakespearebarkeronthethethetheincithechristmasvalthein=barkertheintheltreligiousinguardianintheonshakespearebarkerthechristmas23grounds"thefaheininbarkerkurta75fairiesbarkerinperhapsthethe

**Prompt**: The future of technology
**Generated**: The future of technologygroundstheachristmasthisbarkerconceptbarkerguardianperhapstheatheinthetheininatheonhetheinasthekurtina23barkerbarkertheperhapstheinbarkerthethisperhapsbarkergroundskurttheinfairiesthethe2the


## Usage

```python
from run_retgen import RETGENSystem

# Load model
retgen = RETGENSystem.load('path/to/model')

# Generate text
generated = retgen.generate(
    "Your prompt here",
    max_length=100,
    temperature=0.8
)
print(generated)
```

## Citation

If you use this model, please cite the RETGEN paper:
```
@article{retgen2024,
  title={RETGEN: Retrieval-Enhanced Text Generation},
  author={...},
  year={2024}
}
```
