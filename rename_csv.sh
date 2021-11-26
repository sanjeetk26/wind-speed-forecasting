#!/bin/bash

for file in *.csv; do
	mv $file "${file: -8}"
done
