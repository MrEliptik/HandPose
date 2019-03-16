#!/bin/bash
echo '[INFO] >>> Downloading dataset from Mediafire..'
wget http://download1079.mediafire.com/5lfrid08tnxg/wt7dc5e9jgnym04/Poses.tar.gz

echo '[INFO] >>> Extracting files into ./Poses/ ..'
tar -xzvf Poses.tar.gz

echo '[INFO] >>> Removing archive..'
rm Poses.tar.gz

echo '[INFO] >>> Success!'