#!/bin/bash
for i in {1..10}
do
   SLURM_JOB_ID=$i python torch_dlv3_ens.py
done