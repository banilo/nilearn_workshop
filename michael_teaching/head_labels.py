#!head /Users/admin/nilearn_data/haxby2001/subj1/labels.txt

f = open(target_file)
for i in range(5):
    print(f.readline())
    
img = nibabel.load(func_file)
print img.shape

