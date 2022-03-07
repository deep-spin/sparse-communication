WANDB_NOTES="gaussiansp" python main.py --cfg cfg/iclr-gaussiansp.json --batch_size 100 --device cuda:0 --seed 10 &> logs/iclr-gaussiansp-10 &
sleep 10
WANDB_NOTES="gaussiansp" python main.py --cfg cfg/iclr-gaussiansp.json --batch_size 100 --device cuda:0 --seed 11 &> logs/iclr-gaussiansp-11 &
sleep 10
WANDB_NOTES="gaussiansp" python main.py --cfg cfg/iclr-gaussiansp.json --batch_size 100 --device cuda:0 --seed 12 &> logs/iclr-gaussiansp-12 &
sleep 10
WANDB_NOTES="gaussiansp" python main.py --cfg cfg/iclr-gaussiansp.json --batch_size 100 --device cuda:0 --seed 13 &> logs/iclr-gaussiansp-13 &
sleep 10
WANDB_NOTES="gaussiansp" python main.py --cfg cfg/iclr-gaussiansp.json --batch_size 100 --device cuda:0 --seed 14 &> logs/iclr-gaussiansp-14 &
sleep 10

wait
