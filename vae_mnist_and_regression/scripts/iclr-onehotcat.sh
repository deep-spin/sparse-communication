WANDB_NOTES="onehotcat" python main.py --cfg cfg/iclr-onehotcat.json --batch_size 100 --gen_lr 1e-5 --inf_lr 1e-4 --inf_p_drop 0.1 --project iclr-submission --device cuda:1 --seed 10 &> logs/iclr-onehotcat-10 &
sleep 10
WANDB_NOTES="onehotcat" python main.py --cfg cfg/iclr-onehotcat.json --batch_size 100 --gen_lr 1e-5 --inf_lr 1e-4 --inf_p_drop 0.1 --project iclr-submission --device cuda:1 --seed 11 &> logs/iclr-onehotcat-11 &
sleep 10
WANDB_NOTES="onehotcat" python main.py --cfg cfg/iclr-onehotcat.json --batch_size 100 --gen_lr 1e-5 --inf_lr 1e-4 --inf_p_drop 0.1 --project iclr-submission --device cuda:1 --seed 12 &> logs/iclr-onehotcat-12 &
sleep 10
WANDB_NOTES="onehotcat" python main.py --cfg cfg/iclr-onehotcat.json --batch_size 100 --gen_lr 1e-5 --inf_lr 1e-4 --inf_p_drop 0.1 --project iclr-submission --device cuda:1 --seed 13 &> logs/iclr-onehotcat-13 &
sleep 10
WANDB_NOTES="onehotcat" python main.py --cfg cfg/iclr-onehotcat.json --batch_size 100 --gen_lr 1e-5 --inf_lr 1e-4 --inf_p_drop 0.1 --project iclr-submission --device cuda:1 --seed 14 &> logs/iclr-onehotcat-14 &
sleep 10

wait
