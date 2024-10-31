1. Make sure your working directory is ```.\embedders_comparison\```

2. Dump the embeddings:

```python
python dump_embeddings.py
```

*Use `--truncate_10k` flag to dump only 10'000 rows of data if you debuging the project*.

3. Train and evaluate an CatBoost model:

```python
python .\question_to_solver\train.py --target best
```

*You can specify a particular embedder from Hugging Face with `--embedder` flag*.