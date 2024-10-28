# /bin/bash
#
echo Loading stackexchange softwareengineering data
#
echo Create a directory to store the data
cd ..
mkdir -p data
cd data
#
echo Download the data
wget https://archive.org/download/stackexchange/softwareengineering.stackexchange.com.7z
#
echo Extract the data
7z x softwareengineering.stackexchange.com.7z
#
cd ..
#