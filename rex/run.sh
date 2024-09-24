echo "Running 1w1a..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 1w1a --weight_decay 0 --lr 2e-3 --epochs 200
cp checkpoint/ckpt_state.pth checkpoint/alexnet_1w1a.pth
