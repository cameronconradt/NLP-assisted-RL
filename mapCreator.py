import pickle
import csv

map = pickle.load(open('output.obj', 'rb'))
w = csv.writer(open("output.csv", "wt"))
max = csv.writer(open('output_condensed.csv', 'wt'))
condensed_map = {}
for key, val in map.items():
    if key != 'count':
        if len(val.keys()) > 1:
            w.writerow([key, val])
            maxkey = ''
            maxval = 0
            for littlekey, littleval in val.items():
                if littleval > maxval:
                    maxkey = littlekey
                    maxval = littleval
            if maxval > 1:
                max.writerow([key, maxkey])


