import random

filelist = '/ifp/users/haichao/projects/camid/data/dresden/all_train_val_imgs'
filelist_label_train = '/ifp/users/haichao/projects/camid/data/dresden/filelist_train'
filelist_label_valid = '/ifp/users/haichao/projects/camid/data/dresden/filelist_valid'
save_label_dict = '/ifp/users/haichao/projects/camid/data/dresden/label_dict'
val_percent = 0.15

with open(filelist) as f:
    lines = f.read().splitlines()

label_dict = {}
label_counter = 0
lines_with_label = []
for line in lines:
    label_strs = line.split('/')[-1]
    label_strs = label_strs.split('_')
    label_str = label_strs[0] + '_' + label_strs[1]
    if label_str == 'Nikon_D70s':
        label_str = 'Nikon_D70'
    if label_str not in label_dict.keys():
        label_dict[label_str] = label_counter
        label_counter += 1
    line_with_label = [line, label_dict[label_str]]
    lines_with_label.append(line_with_label)

random.shuffle(lines_with_label)
num_imgs = len(lines_with_label)
val_idxs = random.sample(range(num_imgs), int(num_imgs * val_percent))
lines_valid = [lines_with_label[i] for i in range(num_imgs) if i in val_idxs]
lines_train = [lines_with_label[i] for i in range(num_imgs) if i not in val_idxs]

with open(filelist_label_valid, 'w') as f:
    for line in lines_valid:
        f.write(line[0] + ' ' + str(line[1]) + '\n')

with open(filelist_label_train, 'w') as f:
    for line in lines_train:
        f.write(line[0] + ' ' + str(line[1]) + '\n')

with open(save_label_dict, 'w') as f:
    for key in label_dict.keys():
        f.write(key + ': ' + str(label_dict[key]) + '\n')
