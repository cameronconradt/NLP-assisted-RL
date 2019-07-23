import csv
import os
import re
import string
import subprocess
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):

    def error(self, message):
        pass

    def handle_starttag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        f = open('output.txt', 'a+')
        f.write(data)


if not os.path.exists('./final.txt'):
    subprocess.call("./pull_manuals.sh")
    parser = MyHTMLParser()
    data = open('./final.txt', 'r', errors='ignore').read()
    parser.feed(data)

data = open('output.txt', 'r', errors='ignore').read()
data = data.replace('\n', ' ')
data = data.lower()
article_regex = '(\\s+)(a|an|and|the|or|of|on|in|to)(\\s+)'
old_data = data
data = re.sub(article_regex, ' ', data)
while old_data is not data:
    old_data = data
    data = re.sub(article_regex, ' ', data)
data = data.translate(str.maketrans('', '', string.digits))
data = re.split('[.?!]\\s', data)

dictionary = {}
for i in data:
    i = i.translate(str.maketrans('', '', string.punctuation))
    words = i.split()
    for j in words:
        for k in words:
            if j is not k:
                if j in dictionary:
                    if k in dictionary[j]:
                        count = dictionary[j][k]
                        dictionary[j].update({k: count + 1})
                    else:
                        dictionary[j].update({k: 1})
                else:
                    dictionary.update({j: {}})

maxes = {}
for key in dictionary:
    total = 0
    if key not in maxes:
        maxes.update({key: ['', 0]})
    for item in dictionary[key]:
        total += dictionary[key][item]
    for item in dictionary[key]:
        value = (dictionary[key][item] / total) * 100
        if value > maxes[key][1]:
            maxes.update({key: [item, value]})
        dictionary[key].update({item: value})

w = csv.writer(open("output.csv", "w"))
for key, val in maxes.items():
    w.writerow([key, val])
