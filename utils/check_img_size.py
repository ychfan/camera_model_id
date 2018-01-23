import re

header_file = '/ifp/users/haichao/projects/camid/data/header_list'

with open(header_file) as f:
    lines = f.read().splitlines()

for line in lines:
    try:
        dim1 = int(re.search('xresolution=(\d)+', line).group().split('=')[-1])
        dim2 = int(re.search('yresolution=(\d)+', line).group().split('=')[-1])
    except AttributeError:
        try:
            dims = print(re.search('(\d)+x(\d)+', line).group())
        except AttributeError:
            print(line)
    if dim1 < 128 or dim2 < 128:
        print(line)
