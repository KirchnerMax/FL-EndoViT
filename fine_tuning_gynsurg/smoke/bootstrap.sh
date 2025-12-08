echo "Starting bootstrap analysis..."
echo "--------------------------------"

python bootstrap.py --cen test_CEN_ENDOVIT_predictions_fold_0.csv --fl test_FL_ENDOVIT_predictions_fold_0.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen test_CEN_ENDOVIT_predictions_fold_1.csv --fl test_FL_ENDOVIT_predictions_fold_1.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen test_CEN_ENDOVIT_predictions_fold_2.csv --fl test_FL_ENDOVIT_predictions_fold_2.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen test_CEN_ENDOVIT_predictions_fold_3.csv --fl test_FL_ENDOVIT_predictions_fold_3.csv --reps 10000 --hierarchical
echo "--------------------------------"

echo "Starting comparison with RESNET50..."
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_0.csv --fl test_FL_ENDOVIT_predictions_fold_0.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_1.csv --fl test_FL_ENDOVIT_predictions_fold_1.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_2.csv --fl test_FL_ENDOVIT_predictions_fold_2.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_3.csv --fl test_FL_ENDOVIT_predictions_fold_3.csv --reps 10000 --hierarchical
echo "--------------------------------"

echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_0.csv --fl test_CEN_ENDOVIT_predictions_fold_0.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_1.csv --fl test_CEN_ENDOVIT_predictions_fold_1.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_2.csv --fl test_CEN_ENDOVIT_predictions_fold_2.csv --reps 10000 --hierarchical
echo "--------------------------------"
python bootstrap.py --cen ../output/test_RESNET50_predictions_fold_3.csv --fl test_CEN_ENDOVIT_predictions_fold_3.csv --reps 10000 --hierarchical
echo "--------------------------------"
echo "Bootstrap analysis completed."

