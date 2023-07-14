import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()
    
    if args.mode == 'train':
        pid = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310']
        # pid = ['L067', 'L096', 'L109']
    elif args.mode == 'val':
        pid = ['L506']
    elif args.mode == 'test':
        pid = ['L333']

    idx_min = 0

    with open(f'./datasets/mayo/mayo_{args.mode}.txt', 'w') as f:
        for id in pid:
            folder_LDCT = f'datasets/mayo/LDCT/{id}'
            folder_NDCT = f'datasets/mayo/NDCT/{id}'

            names_LDCT = os.listdir(folder_LDCT)
            names_LDCT.sort()
            num_LDCT = len(names_LDCT)
            names_NDCT = os.listdir(folder_NDCT)
            names_NDCT.sort()
            num_NDCT = len(names_NDCT)

            assert num_NDCT == num_LDCT
            idx_max = idx_min + num_NDCT - 1
            for i in range(num_LDCT):
                name_LDCT = names_LDCT[i]
                file_LDCT = '{}/{}'.format(folder_LDCT, name_LDCT)
                name_NDCT = names_NDCT[i]
                file_NDCT = '{}/{}'.format(folder_NDCT, name_NDCT)
                assert os.path.exists(file_LDCT)
                assert os.path.exists(file_NDCT)
                if i == 0:
                    label = 0
                elif i == num_LDCT - 1:
                    label = -1
                else:
                    label = 1
                f.write('{} {} {} {} {}\n'.format(file_LDCT, file_NDCT, label, idx_min, idx_max))
            idx_min = idx_max + 1


if __name__ == "__main__":
    main()