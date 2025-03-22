#!/bin/bash

for i in 0{0..9} {10..30}
do
	for j in 0{0..9} {10..99}
	do
		if [ $i = "30" ]
		then
			if [ $j = "05" ]
			then
				break
			fi
		fi
		wget --user USERNAME --password PASSWORD URL/${i}/${i}${j}.warc.gz
	done
done



