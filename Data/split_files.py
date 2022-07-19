import os
import shutil
names = ["Dong","Vinh","Phuc","Tommy","An","Duy","Khoa","Thanh","Đạt"]
files = [file for file in os.listdir('./image')]
batchsize = 300
index = 0
remaining = len(files)
while remaining > 0:
    batch = min(remaining, batchsize)
    if not names: break

    name = names.pop(0)
    os.mkdir(f'./{name}')
    for file in files[index:index+batch]:
        shutil.copy(os.path.join('./image/', file), f'./{name}')

    index += batch
    remaining -= batch

