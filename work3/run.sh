CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --master_port 12312 advanced_version.py --n_threads 8 --n_devices 1