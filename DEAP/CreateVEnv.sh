#!/bin/bash

directory="venv"
python=python3.8
requirements="requirements.txt"

while getopts d:p:r: flag
do
    case "${flag}" in
        d) directory=${OPTARG};;
        p) python=${OPTARG};;
        r) requirements=${OPTARG};;
    esac
done

echo "Requirements: $requirements"

if [[ -f "$requirements" ]]; then
  echo "requirements.txt found"
else
  echo "requirements.txt not found, please ensure file exists or give location using'-r'"
  exit 1
fi

if [[ -d "$directory" ]]; then
  echo "Virtual environment directory '$directory' already exists"
else
  echo "Creating venv in '$directory'"
#  mkdir $directory
#  cd $directory
#  virtualenv -p $python .
  $python -m venv $directory
  cd ..
fi

echo "~/$directory/bin/$python -m pip install --upgrade -r $requirements"
~/$directory/bin/$python -m pip install --upgrade -r $requirements

echo "Cloning TensorNAS source"

if [ -d "TensorNAS-Project" ]; then
  echo "TensorNAS found, checking that it is recent"
  cd TensorNAS-Project
  git submodule foreach git pull origin master
else
  git clone --remote --recurse-submodules https://github.com/alxhoff/TensorNAS-Project.git
fi

echo "Create symlinks for demos"

ln -s ~/TensorNAS-Project/TensorNAS ~/TensorNAS-Project/Demos/DEAP/Distributed/TensorNAS
ln -s ~/TensorNAS-Project/TensorNAS ~/TensorNAS-Project/Demos/DEAP/Standalone/TensorNAS
ln -s ~/TensorNAS-Project/TensorNAS ~/TensorNAS-Project/Demos/Standalone/TensorNAS

