import pickle
import csv

map = pickle.load(open('checkpoint.obj', 'rb'))
w = csv.writer(open("output.csv", "wt"))
for key, val in map.items():
    if key != 'count':
        if len(val.keys()) > 1:
            w.writerow([key, val])
