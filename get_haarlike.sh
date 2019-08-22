#!/bin/bash

HAARLIKE_FILE="haarcascade_frontalface_alt.xml"

if [ ! -e $HAARLIKE_FILE ]; then
	wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
fi

