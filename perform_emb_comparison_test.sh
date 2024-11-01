# /bin/bash
# 
echo '=>' Updating pip dependencies
#
pip install -r requirements.txt -q
#
#
echo '=>' Loading stackexchange softwareengineering data if needed
#
if [ ! -d "data" ]; then
    ./utils/load_data.sh
    echo '=>' Data loading complete
else
    echo '=!' Data already loaded
fi
#
#
echo '=>' Dumping embedding if needed
#
cd embedders_comparison
if [ ! -d "embeddings" ]; then
    python dump_embeddings.py --truncate_10k --embedder MiniLM3
    echo '=>' Dumping embedding complete
else
    echo '=!' Embedding already dumped
fi
#
#
echo '=>' Running train-eval to best target
#
cd question_to_solver
python train.py --truncate_100 --embedder MiniLM3 --target multy
#
echo '=!' Running train-eval to multy target complete
#
#
cd ../..
#
