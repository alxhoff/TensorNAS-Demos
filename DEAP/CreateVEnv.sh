#!/bin/bash

directory="env"
python=python3.8
requirements="requirements.txt"

while getopts d:p:r: flag
do
    case "${flag}" in
        d) directory="${OPTARG}/$directory";;
        p) python=${OPTARG};;
        r) requirements={$OPTARG};;
    esac
done

if test -f "$requirements"; then
  echo "requirements.txt found"
else
  echo "requirements.txt not found, please ensure file exists or give location using'-r'"
  exit 1
fi

echo "Creating venv in '$directory'"

mkdir $directory
cd $directory
virtualenv -p $python .
cd ..
source $directory/bin/activate
$python -m pip install --no-cache-dir --upgrade --force-reinstall -r ../$requirements

echo "Cloning TensorNAS source"

if [ -d "TensorNAS" ]; then
  echo "TensorNAS found, checking that it is recent"
  cd TensorNAS
