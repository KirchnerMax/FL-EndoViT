
echo "-----------------------------------"
echo "Running for GynSurg Models f1 score"
echo "-----------------------------------"
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv --fl test_FL_ENDOVIT_predictions_fold_0.csv
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_1.csv --fl test_FL_ENDOVIT_predictions_fold_1.csv
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_2.csv --fl test_FL_ENDOVIT_predictions_fold_2.csv
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_3.csv --fl test_FL_ENDOVIT_predictions_fold_3.csv

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv,test_CEN_ENDOVIT_predictions_fold_1.csv,test_CEN_ENDOVIT_predictions_fold_2.csv,test_CEN_ENDOVIT_predictions_fold_3.csv --fl test_FL_ENDOVIT_predictions_fold_0.csv,test_FL_ENDOVIT_predictions_fold_1.csv,test_FL_ENDOVIT_predictions_fold_2.csv,test_FL_ENDOVIT_predictions_fold_3.csv

echo "-----------------------------------"

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv --fl ../output/test_RESNET50_predictions_fold_0.csv
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_1.csv --fl ../output/test_RESNET50_predictions_fold_1.csv
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_2.csv --fl ../output/test_RESNET50_predictions_fold_2.csv
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_3.csv

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv,test_CEN_ENDOVIT_predictions_fold_1.csv,test_CEN_ENDOVIT_predictions_fold_2.csv,test_CEN_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_0.csv,../output/test_RESNET50_predictions_fold_1.csv,../output/test_RESNET50_predictions_fold_2.csv,../output/test_RESNET50_predictions_fold_3.csv

echo "-----------------------------------"

python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_0.csv --fl ../output/test_RESNET50_predictions_fold_0.csv
python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_1.csv --fl ../output/test_RESNET50_predictions_fold_1.csv
python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_2.csv --fl ../output/test_RESNET50_predictions_fold_2.csv
python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_3.csv

python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_0.csv,test_FL_ENDOVIT_predictions_fold_1.csv,test_FL_ENDOVIT_predictions_fold_2.csv,test_FL_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_0.csv,../output/test_RESNET50_predictions_fold_1.csv,../output/test_RESNET50_predictions_fold_2.csv,../output/test_RESNET50_predictions_fold_3.csv

echo "-----------------------------------"
echo "Running for GynSurg Models acc"
echo "-----------------------------------"

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv --fl test_FL_ENDOVIT_predictions_fold_0.csv --metric acc
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_1.csv --fl test_FL_ENDOVIT_predictions_fold_1.csv --metric acc
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_2.csv --fl test_FL_ENDOVIT_predictions_fold_2.csv --metric acc
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_3.csv --fl test_FL_ENDOVIT_predictions_fold_3.csv --metric acc

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv,test_CEN_ENDOVIT_predictions_fold_1.csv,test_CEN_ENDOVIT_predictions_fold_2.csv,test_CEN_ENDOVIT_predictions_fold_3.csv --fl test_FL_ENDOVIT_predictions_fold_0.csv,test_FL_ENDOVIT_predictions_fold_1.csv,test_FL_ENDOVIT_predictions_fold_2.csv,test_FL_ENDOVIT_predictions_fold_3.csv --metric acc

echo "-----------------------------------"

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv --fl ../output/test_RESNET50_predictions_fold_0.csv --metric acc
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_1.csv --fl ../output/test_RESNET50_predictions_fold_1.csv --metric acc
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_2.csv --fl ../output/test_RESNET50_predictions_fold_2.csv --metric acc
python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_3.csv --metric acc

python bootstrap_2.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv,test_CEN_ENDOVIT_predictions_fold_1.csv,test_CEN_ENDOVIT_predictions_fold_2.csv,test_CEN_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_0.csv,../output/test_RESNET50_predictions_fold_1.csv,../output/test_RESNET50_predictions_fold_2.csv,../output/test_RESNET50_predictions_fold_3.csv --metric acc

echo "-----------------------------------"

python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_0.csv --fl ../output/test_RESNET50_predictions_fold_0.csv --metric acc
python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_1.csv --fl ../output/test_RESNET50_predictions_fold_1.csv --metric acc
python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_2.csv --fl ../output/test_RESNET50_predictions_fold_2.csv --metric acc
python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_3.csv --metric acc

python bootstrap_2.py --cen test_FL_ENDOVIT_predictions_fold_0.csv,test_FL_ENDOVIT_predictions_fold_1.csv,test_FL_ENDOVIT_predictions_fold_2.csv,test_FL_ENDOVIT_predictions_fold_3.csv --fl ../output/test_RESNET50_predictions_fold_0.csv,../output/test_RESNET50_predictions_fold_1.csv,../output/test_RESNET50_predictions_fold_2.csv,../output/test_RESNET50_predictions_fold_3.csv --metric acc


