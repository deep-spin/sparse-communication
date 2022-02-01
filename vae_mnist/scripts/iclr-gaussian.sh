WANDB_NOTES="gaussian" python main.py --cfg cfg/iclr-gaussian.json --batch_size 100 --device cuda:1 --seed 10 &> logs/iclr-gaussian-10 &
sleep 10
WANDB_NOTES="gaussian" python main.py --cfg cfg/iclr-gaussian.json --batch_size 100 --device cuda:1 --seed 11 &> logs/iclr-gaussian-11 &
sleep 10
WANDB_NOTES="gaussian" python main.py --cfg cfg/iclr-gaussian.json --batch_size 100 --device cuda:1 --seed 12 &> logs/iclr-gaussian-12 &
sleep 10
WANDB_NOTES="gaussian" python main.py --cfg cfg/iclr-gaussian.json --batch_size 100 --device cuda:1 --seed 13 &> logs/iclr-gaussian-13 &
sleep 10
WANDB_NOTES="gaussian" python main.py --cfg cfg/iclr-gaussian.json --batch_size 100 --device cuda:1 --seed 14 &> logs/iclr-gaussian-14 &
sleep 10

wait
