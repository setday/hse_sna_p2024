# /bin/bash
# 
echo '=>' Updating pip dependencies
#
pip install -r requirements.txt -q
#
#
echo '=>' Loading stackexchange softwareengineering data if needed
#
./utils/load_data.sh
#
#
echo '=>' Dumping embedding if needed
#
cd embedders_comparison
if [ ! -d "embeddings" ]; then
    python dump_embeddings.py
    echo '=>' Dumping embedding complete
else
    echo '=!' Embedding already dumped
fi
#
#
echo '=>' Running train-eval to best target
#
cd question_to_solver
python train.py --target multy
#
echo '=!' Running train-eval to multy target complete
#
#
cd ../..
#
