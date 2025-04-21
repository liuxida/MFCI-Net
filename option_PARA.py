import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--path_to_PARA_images', type=str, default='/mnt/nfs-storage/ljw/EAMB-Net-main/EAMB-Net-main/PARA/PARA_data/imgs/', help='directory to images')

    parser.add_argument('--path_to_PARA_save_csv', type=str,default="/mnt/nfs-storage/ljw/EAMB-Net-main/EAMB-Net-main/PARA/PARA_data/",help='directory to csv_folder')
    # #
    # parser.add_argument('--path_to_PARA_images', type=str,
    #                     default='/root/shared-nvme/EAMB-Net-main/para/imgs/',
    #                     help='directory to images')
    #
    # parser.add_argument('--path_to_PARA_save_csv', type=str,
    #                     default="/root/shared-nvme/EAMB-Net-main/para/",
    #                     help='directory to csv_folder')

    parser.add_argument('--weight_decay',  type=int, default=5e-4, help='Weight decay')

    # parser.add_argument('--init_lr_res365_last', type=int, default=0.0003, help='learning_rate')
    # parser.add_argument('--init_lr_mobileNet', type=int, default=0.0003, help='learning_rate')
    # parser.add_argument('--init_lr_head', type=int, default=0.0003, help='learning_rate')
    # parser.add_argument('--init_lr_head_rgb', type=int, default=0.0003, help='learning_rate')
    # parser.add_argument('--init_lr_hypernet', type=int, default=0.0003, help='learning_rate')
    # parser.add_argument('--init_lr_tygertnet', type=int, default=0.0003, help='learning_rate')
    parser.add_argument('--init_lr', type=int, default=0.00003, help='learning_rate')


    parser.add_argument('--num_epoch', type=int, default=100, help='epoch num for train')

    parser.add_argument('--batch_size', type=int,default=4,help='how many pictures to process one time')

    parser.add_argument('--train_num_workers', type=int, default=0, help ='num_workers')

    parser.add_argument('--test_num_workers', type=int, default=0, help='num_workers')

    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')


    args = parser.parse_args()
    return args
