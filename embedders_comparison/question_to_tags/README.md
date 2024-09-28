1. Make sure your working directory is ```.\embedders_comparison\question_to_tags```

2. Dump the embeddings:

```python
py dump_embeddings.py
```

3. Train and evaluate an FCNN:

```python
py train.py
```

*You can specify a particular embedder from Hugging Face inside* ```train.py```.