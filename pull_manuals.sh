#!/bin/bash

function filter() {
        if !( grep -q "HTML Manual Not Found..." "$1" ) ; then
		    echo "filtering $1"
            sed -n '/<body>/,/body>/p' "$1" > "manuals/$1"
            rm "$1"
        else
            rm "$1"
        fi

}

function max() {
   while [[ `jobs | wc -l` -ge $1 ]]
   do
      sleep 1
   done
}
mkdir manuals
for i in {0..5000}; do
	string="https://atariage.com/manual_html_page.php?SoftwareID=$i"
	max 100
	( wget -q -O "$i.text" $string && filter "$i.text" ) &
done
wait
echo "finished pulling, catting"
#cat manuals/*.text >> final.txt
#rm -rf manuals
