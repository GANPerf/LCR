#  Reading data
import os
import pathlib
import shutil

class CUB():
    def __init__(self, root, is_train=True):
        self.root = root
        self.is_train = is_train

        def convert_bird(data_root):
            images_txt = os.path.join(data_root, 'images.txt')
            train_val_txt = os.path.join(data_root, 'train_test_split.txt')
            labels_txt = os.path.join(data_root, 'image_class_labels.txt')
            image_folder=os.path.join(data_root, 'images')
            id_name_dict = {}
            id_class_dict = {}
            id_train_val = {}
            with open(images_txt, 'r', encoding='utf-8') as f:
                line = f.readline()
                while line:
                    id, name = line.strip().split()
                    id_name_dict[id] = name
                    line = f.readline()

            with open(train_val_txt, 'r', encoding='utf-8') as f:
                line = f.readline()
                while line:
                    id, trainval = line.strip().split()
                    id_train_val[id] = trainval
                    line = f.readline()

            with open(labels_txt, 'r', encoding='utf-8') as f:
                line = f.readline()
                while line:
                    id, class_id = line.strip().split()
                    id_class_dict[id] = int(class_id)
                    line = f.readline()

            train_txt = os.path.join(data_root, 'bird_train.txt')
            test_txt = os.path.join(data_root, 'bird_test.txt')

            train_folder=os.path.join(data_root, 'train')
            test_folder = os.path.join(data_root, 'test')
            if os.path.exists(train_txt):
                os.remove(train_txt)
            if os.path.exists(test_txt):
                os.remove(test_txt)
            if os.path.exists(train_folder):
                os.remove(train_folder)
            if os.path.exists(test_folder):
                os.remove(test_folder)

            f1 = open(train_txt, 'a', encoding='utf-8')
            f2 = open(test_txt, 'a', encoding='utf-8')

            for id, trainval in id_train_val.items():
                if trainval == '1':
                    src_path=os.path.join(image_folder, id_name_dict[id])
                    dst_path=os.path.join(train_folder,str(id_class_dict[id] - 1))
                    pathlib.Path(dst_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_path,os.path.join(dst_path,os.path.basename(src_path)))
                    f1.write('%s %d\n' % (id_name_dict[id], id_class_dict[id] - 1))
                else:

                    src_path=os.path.join(image_folder, id_name_dict[id])
                    dst_path=os.path.join(test_folder,str(id_class_dict[id] - 1))
                    pathlib.Path(dst_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_path,os.path.join(dst_path,os.path.basename(src_path)))
                    f2.write('%s %d\n' % (id_name_dict[id], id_class_dict[id] - 1))
            f1.close()
            f2.close()

        convert_bird(self.root)