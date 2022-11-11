echo "testing all models"
pwd
python main.py --model efficient_net_b7 --epochs 15 --verbose 2 --weights False
python main.py --model efficient_net_v2s --epochs 15 --verbose 2 --weights False
python main.py --model enet --epochs 15 --verbose 2 --weights False
python main.py --model inception_v3 --epochs 15 --verbose 2 --weights False
python main.py --model knn --epochs 15 --verbose 2 --weights False
python main.py --model resnet50 --epochs 15 --verbose 2 --weights False
python main.py --model rf --epochs 15 --verbose 2 --weights False
python main.py --model seq --epochs 15 --verbose 2 --weights False
python main.py --model simple_seq --epochs 15 --verbose 2 --weights False
python main.py --model svm --epochs 15 --verbose 2 --weights False
python main.py --model unet --epochs 15 --verbose 2 --weights False
python main.py --model vgg16 --epochs 15 --verbose 2 --weights False
echo "all models run and testing complete"