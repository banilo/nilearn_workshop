# first possibility
rec = np.recfromcsv(target_file, delimiter=' ')
labels = rec['labels']
runs = rec['chunks']

# second possibility
import pandas
csv = pandas.read_csv(target_file, sep=' ')
labels = csv['labels'].values
runs = rec['chunks']

# third possibility
arr = np.recfromtxt(target_file)

print(labels)

