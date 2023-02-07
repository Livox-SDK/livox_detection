#! /bin/bash

echo "Start to test the model..."
python3 test_ros.py --pt ../pt/livox_model_2.pt >>../results/result_1.txt
