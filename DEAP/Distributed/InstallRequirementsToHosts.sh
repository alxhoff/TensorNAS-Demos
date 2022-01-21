#!/bin/bash

python=python3.8
directory="env"
host_file="hosts"
requirements="../requirements.txt"

if ! command -v pdsh &> /dev/null
then
    echo "pdsh could not be found, please install and rerun script"
    exit 1
fi

while getopts u:h:p:r:d: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        h) host_file=${OPTARG};;
        p) python=${OPTARG};;
        r) requirements=${OPTARG};;
        d) directory=${OPTARG};;
    esac
done

echo "Copying scripts, setting up virtual environment and installing requirements to all hosts"

while IFS= read -r line; do
    ip=${line%% *}
    echo "HOST: $ip"
    scp $requirements $username@$ip:~/
    scp ../CreateVEnv.sh $username@$ip:~/
done < "hosts"

pdsh -w ^hosts -l $username -R ssh "bash ~/CreateVEnv.sh -p $python -r $requirements"

echo "Exporting PATH variables to a host as file system is shared"

first_host=`head -n 1 hosts`

python_command='grep -q "$HOME/env/bin" $HOME/.bashrc || echo "PATH=$HOME/env/bin:$PATH" >> $HOME/.bashrc'
ssh $username@$first_host $python_command
python_path="\$HOME/$directory/lib/$python/site-packages/"
python_path_command="grep -q \"$python_path\" \$HOME/.bashrc || echo \"PYTHONPATH=$python_path:\$PYTHONPATH\" >> \$HOME/.bashrc"
ssh $username@$first_host $python_path_command

pdsh -w ^hosts -l $username -R ssh "source ~/.bashrc"
