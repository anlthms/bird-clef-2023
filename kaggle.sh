#!/bin/bash
#
# This script is intended for use with Kaggle code competitions.
# It creates a dataset on Kaggle with all the files required to
# make a submission.

dataset_name=birdclefcode
mkdir -p ../$dataset_name

# check for Kaggle API credentials
[[ ! -f ~/.kaggle/kaggle.json ]]  && { echo 'error: Kaggle API token needs to be configured using the "Import Data" tab'; exit 1; }

# create metadata file
[[ ! -f ../$dataset_name/dataset-metadata.json ]] &&
{
    echo "kaggle init..."
    cd ../$dataset_name

    # create metadata file
    user_name=$(cat ~/.kaggle/kaggle.json | cut -f4 -d'"')
    echo \
"{
  \"title\": \"$dataset_name\",
  \"id\": \"$user_name/$dataset_name\",
  \"licenses\": [
    {
      \"name\": \"CC0-1.0\"
    }
  ]
}" > dataset-metadata.json
}

cd ../$dataset_name
touch __init__.py

status=$(kaggle datasets status $dataset_name)
[[ $status != "ready" ]]  && { echo "creating kaggle dataset..."; kaggle datasets create; }

# copy source code
for file in config.py dataset.py inference.py models.py util.py
do
    cp -v ../code/$file .
done

# copy trained model
cp -v ../output/model1.pth .
cp -v ../output/model2.pth .

python3 ../code/strip.py model1.pth
python3 ../code/strip.py model2.pth
echo
echo $(tput -T xterm setaf 4)Press enter to upload the code to a private dataset on Kaggle$(tput -T xterm sgr0)
read

# upload to kaggle
kaggle datasets version -m "$(date)"
