Long History Aloha
```
torchrun --standalone --nproc_per_node=7 train.py --config-dir="experiment_configs/aloha" --config-name="transformer_aloha_emb" training.seed=42 logging.mode=disabled dataloader.batch_size=2048
```