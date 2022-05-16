import os


if __name__ == '__main__':
    root = '../PMT_dataset/images_paired'
    save_train_path = os.path.join('../PMT_dataset', 'inversion_train.txt')
    save_test_path = os.path.join('../PMT_dataset', 'inversion_test.txt')
    save_makeup_path = os.path.join('../PMT_dataset', 'inversion_makeup.txt')
    save_nonmakeup_path = os.path.join('../PMT_dataset', 'inversion_nonmakeup.txt')
    f_makeup = open(save_makeup_path, mode='a')
    f_nonmakeup = open(save_nonmakeup_path, mode='a')
    f_train = open(save_train_path, mode='a')
    f_test = open(save_test_path, mode='a')

    for d in os.listdir(root):
        cur_path = os.path.join(root, d)
        is_makeup = (d[0] == 'm')
        for c in os.listdir(cur_path):
            label = c[6:]
            c_path = os.path.join(cur_path, c)
            c_list = os.listdir(c_path)
            num = len(c_list)
            for i, img_name in enumerate(c_list):
                img_path = os.path.join(c_path, img_name)
                if is_makeup:
                    f_makeup.write(img_path + ' ' + label + '\n')
                else:
                    f_nonmakeup.write(img_path + ' ' + label + '\n')
                if i < 0.9 * num:
                    f_train.write(img_path + ' ' + label + '\n')
                else:
                    f_test.write(img_path + ' ' + label + '\n')

    f_makeup.close()
    f_nonmakeup.close()
    f_train.close()
    f_test.close()
