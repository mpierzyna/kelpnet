#!/bin/bash
for i in {1..24}
do
    python torch_deeplabv3.py $i
done