#!/bin/bash
export IFTEST=$1
export True="True"

if [[ $IFTEST == $True ]] ; then
    sudo /media/anaconda3/envs/placenta/bin/python /home/placenta/placenta_all/placenta_diagnosis/MIL/train/main.py --debug True 
else
    sudo /media/anaconda3/envs/placenta/bin/python /home/placenta/placenta_all/placenta_diagnosis/MIL/train/main.py 
fi