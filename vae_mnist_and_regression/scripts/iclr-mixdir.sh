WANDB_NOTES="mixdir" python main.py --cfg cfg/iclr-mixed-dirichlet.json --batch_size 100 --device cuda:0 --exact_KL_Y false --seed 10 &> logs/iclr-mixed-dir-10 &
sleep 10
WANDB_NOTES="mixdir" python main.py --cfg cfg/iclr-mixed-dirichlet.json --batch_size 100 --device cuda:0 --exact_KL_Y false --seed 11 &> logs/iclr-mixed-dir-11 &
sleep 10
WANDB_NOTES="mixdir" python main.py --cfg cfg/iclr-mixed-dirichlet.json --batch_size 100 --device cuda:0 --exact_KL_Y false --seed 12 &> logs/iclr-mixed-dir-12 &
sleep 10
WANDB_NOTES="mixdir" python main.py --cfg cfg/iclr-mixed-dirichlet.json --batch_size 100 --device cuda:0 --exact_KL_Y false --seed 13 &> logs/iclr-mixed-dir-13 &
sleep 10
WANDB_NOTES="mixdir" python main.py --cfg cfg/iclr-mixed-dirichlet.json --batch_size 100 --device cuda:0 --exact_KL_Y false --seed 14 &> logs/iclr-mixed-dir-14 &
sleep 10

wait
