#!/bin/bash

function filter() {
        if !( grep -q "HTML Manual Not Found..." "$1" ) ; then
		echo "filtering $1"
                sed -n '/<body>/,/body>/p' "$1" > "manuals/$1"
	else
		echo "removing $1"
		rm "$1"
        fi

}

for i in {0..5000}; do
	string="https://atariage.com/manual_html_page.php?SoftwareID=$i"
	( wget -q -O "$i.text" $string && filter "$i.text" ) &
done
wait

cat manuals/*.text >> final.txt
rm manuals/*.text
rm *.text
