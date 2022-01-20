#!/bin/bash

host_file="hosts"
requirements="../requirements.txt"

if ! command -v pdsh &> /dev/null
then
    echo "pdsh could not be found, please install and rerun script"
    exit 1
fi

while getopts u:h:p: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        h) host_file=${OPTARG};;
        p) python=${OPTARG};;
        r) requirements=${OPTARG};;
    esac
done

echo "Copying requirements.txt to all hosts"

while IFS= read -r line; do
    ip=${line%% *}
    scp $requirements $username@$ip:~/
    scp ../CreateVEnv.sh $username@$ip:~/
done < "hosts"

echo "Setting up virtual environment and installing requirements"

pdsh -w ^hosts -l $username -R ssh "bash ~/CreateVEnv.sh -p $python -r $requirements"