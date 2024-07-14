import shutil
import os
from PIL import Image

def remove_file(old_path, new_path):
    filelist = os.listdir(old_path)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)

with open('/home/ftt/UAVM/model/uavm_opt1/result.csv','r') as f:
    line =f.readline().split(',')
    for i in range(6):
        line = f.readline().replace('\n', '').split(',')
        for j in range(10):
            filename = line[j+1].split('/')[-1]
            des_path = "/home/ftt/UAVM/datasets/result/"+str(j)
            folder = os.path.exists(des_path)
            if not folder:
                os.makedirs(des_path)
            src = str(line[j+1])
            des = des_path + '/' + str(i) + '.webp'
            des_jpg = des_path + '/' + str(i) + '.jpg'
            shutil.copyfile(src, des)
            im = Image.open(des).convert('RGB')
            im.save(des_jpg, 'jpeg')
            os.remove(des)

# if __name__ == '__main__':
#     remove_file(r"/data/temp1", r"/data/temp2")