# /bin/bash
#
echo Create a directory to store the data
if [ ! -d "data" ]; then
    mkdir data
else
    echo "Data directory already exists"
fi
cd data
#
echo Download the data
if [ ! -f "softwareengineering.stackexchange.com.7z" ]; then
    wget https://archive.org/download/stackexchange/softwareengineering.stackexchange.com.7z
else
    echo "Data already downloaded"
fi
#
echo Extract the data
if [ ! -f "Posts.xml" ]; then
    7z x softwareengineering.stackexchange.com.7z
else
    echo "Data already extracted"
fi
#
cd ..
#
echo Data loading complete
