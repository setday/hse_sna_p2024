1. Make sure your working directory is ```embedders_comparison```

2. Dump the embeddings:

```python
python3 dump_embeddings.py
```

3. Change the working directory:

```bash
cd question_to_tags
```

4. Train and evaluate an FCNN:

```python
python3 train.py
```

*You can specify a particular embedder from Hugging Face inside* ```train.py```.