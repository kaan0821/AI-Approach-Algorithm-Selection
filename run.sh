# check if Python virtual environment env exists
if [ ! -f "env/bin/activate" ]; then
    echo "Environment env does not exist. Please install it via ./install.sh"
    exit 1
fi

# activate Python virtual environment
source env/bin/activate

# create output folder
rm -rf models/
mkdir models

# train a model for part 4 - extension 2 and evaluate it
echo "#########   PART 4 (extension 2): random forest model ################"
python scripts/train.py --model-type random_forest --data data/train/ --save models/part4.pt 
python scripts/evaluate.py --model models/part4.pt --data data/test/

# train a model for part 1 (regression-based ANN) and evaluate it
echo "#########   PART 1: regression-based model ################"
python scripts/train.py --model-type regresion_nn --data data/train/ --save models/part1.pt 
python scripts/evaluate.py --model models/part1.pt --data data/test/
echo ""

# train a model for part 2 - basic (classification-based ANN) and evaluate it
echo "#########   PART 2 (basic): classification-based model ################"
python scripts/train.py --model-type classification_nn --data data/train/ --save models/part2_basic.pt
python scripts/evaluate.py --model models/part2_basic.pt --data data/test/
echo ""

echo "#########   PART 2 (advanced): cost-sensitive classification-based model ################"
python scripts/train.py --model-type classification_nn_cost --data data/train/ --save models/part2_advanced.pt 
python scripts/evaluate.py --model models/part2_advanced.pt --data data/test/
echo ""

# train a model for part 3 - extension 1 and evaluate it
echo "#########   PART 3 (extension 1): pairwise cost-sensitive classification model ################"
python scripts/train.py --model-type pairwise_classifier --data data/train/ --save models/part3_pairwise.pt 
python scripts/evaluate.py --model models/part3_pairwise.pt --data data/test/

# train a model for part 4 - extension 2 and evaluate it
echo "#########   PART 4 (extension 2): random forest model ################"
python scripts/train.py --model-type random_forest --data data/train/ --save models/part4.pt 
python scripts/evaluate.py --model models/part4.pt --data data/test/

# deactivate Python virtual environment
deactivate
