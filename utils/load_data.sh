 # /bin/bash
 
 ### Loading stackexchange softwareengineering data

# Create a directory to store the data
mkdir -p data
cd data

# Download the data
wget https://archive.org/download/stackexchange/softwareengineering.stackexchange.com.7z

# Extract the data
7z x softwareengineering.stackexchange.com.7z

cd ..
