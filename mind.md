tensorboard --logdir=./output --port 6006 --bind_all
python train.py -s /home/ysc/3d-gaussian-splatting/data/zhanting/train
python trainmask.py -s  /home/ysc/3dgs-npy/data/zhanting/train --sh_degree 2 --iterations 50000 --densify_until_iter 30000 -r 1