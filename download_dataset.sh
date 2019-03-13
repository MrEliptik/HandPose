#!/bin/bash
echo 'Downloading dataset from Mediafire..'
wget http://download1079.mediafire.com/5lfrid08tnxg/wt7dc5e9jgnym04/Poses.tar.gz

echo 'Extracting files into ./Poses/ ..'
tar -xzvf Poses.tar.gz

echo 'Removing archive..'
rm Poses.tar.gz

echo 'Success!'