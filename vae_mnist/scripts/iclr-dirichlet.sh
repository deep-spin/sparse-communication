WANDB_NOTES="dirichlet" python main.py --cfg cfg/iclr-dirichlet.json --batch_size 100 --device cuda:1 --seed 10 &> logs/iclr-dirichlet-10 &
sleep 10
WANDB_NOTES="dirichlet" python main.py --cfg cfg/iclr-dirichlet.json --batch_size 100 --device cuda:1 --seed 11 &> logs/iclr-dirichlet-11 &
sleep 10
WANDB_NOTES="dirichlet" python main.py --cfg cfg/iclr-dirichlet.json --batch_size 100 --device cuda:1 --seed 12 &> logs/iclr-dirichlet-12 &
sleep 10
WANDB_NOTES="dirichlet" python main.py --cfg cfg/iclr-dirichlet.json --batch_size 100 --device cuda:1 --seed 13 &> logs/iclr-dirichlet-13 &
sleep 10
WANDB_NOTES="dirichlet" python main.py --cfg cfg/iclr-dirichlet.json --batch_size 100 --device cuda:1 --seed 14 &> logs/iclr-dirichlet-14 &
sleep 10

wait
