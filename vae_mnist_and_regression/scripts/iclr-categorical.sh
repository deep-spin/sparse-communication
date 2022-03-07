WANDB_NOTES="categorical" python main.py --cfg cfg/iclr-categorical.json --batch_size 100 --project iclr-submission --device cuda:1 --seed 10 &> logs/iclr-categorical-10 &
sleep 10
WANDB_NOTES="categorical" python main.py --cfg cfg/iclr-categorical.json --batch_size 100 --project iclr-submission --device cuda:1 --seed 11 &> logs/iclr-categorical-11 &
sleep 10
WANDB_NOTES="categorical" python main.py --cfg cfg/iclr-categorical.json --batch_size 100 --project iclr-submission --device cuda:1 --seed 12 &> logs/iclr-categorical-12 &
sleep 10
WANDB_NOTES="categorical" python main.py --cfg cfg/iclr-categorical.json --batch_size 100 --project iclr-submission --device cuda:1 --seed 13 &> logs/iclr-categorical-13 &
sleep 10
WANDB_NOTES="categorical" python main.py --cfg cfg/iclr-categorical.json --batch_size 100 --project iclr-submission --device cuda:1 --seed 14 &> logs/iclr-categorical-14 &
sleep 10

wait
