echo "Running 1w1a..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 1w1a --weight_decay 0 --lr 2e-3 --epochs 200
cp checkpoint/ckpt_state.pth checkpoint/alexnet_1w1a.pth
echo "Running 2w2a..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 2w2a --weight_decay 0 --lr 2e-3 --epochs 200 
cp checkpoint/ckpt_state.pth checkpoint/alexnet_2w2a.pth
echo "Running 3w3a..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 3w3a --weight_decay 0 --lr 2e-3 --epochs 200
cp checkpoint/ckpt_state.pth checkpoint/alexnet_3w3a.pth
echo "Running 8w8a..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 8w8a --weight_decay 2e-7 --lr 5e-3 --epochs 200
cp checkpoint/ckpt_state.pth checkpoint/alexnet_8w8a.pth
echo "Running larger 1w1a network..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 1w1a_larger --weight_decay 0 --lr 2e-3 --epochs 200
cp checkpoint/ckpt_state.pth checkpoint/alexnet_1w1a_larger.pth
echo "Running larger 8w8a network..."
python3.10 run.py --network AlexNet --dataset CIFAR10 --networkversion 8w8a_larger --weight_decay 2e-7 --lr 5e-3 --epochs 200
cp checkpoint/ckpt_state.pth checkpoint/alexnet_8w8a_larger.pth
