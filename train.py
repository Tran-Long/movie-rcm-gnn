from dataclasses import dataclass

import torch
from dataset import MovieLens, train_val_test_split, MovieLensDataloader, TransformRatings, PredefinedMovieLensDataset
from lightgcn import LightGCNModule
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


@dataclass
class Config:
    num_users: int
    num_items: int
    n_layers: int = 3
    embedding_dim: int = 64
    add_self_loops: bool = False
    lambda_val: float = 0.001
    k: int = 20
    batch_size: int = 1024
    learning_rate: float = 0.001
    max_steps: int = 10000
    val_check_interval: int = 10

# rating_transform = TransformRatings()
# data = MovieLens("./data", transform=rating_transform, user_subset=0.001).get()
# train_data, val_data, test_data = train_val_test_split(data)
# print(f"Number of edges: {train_data.edge_index.size(1)} (train), {val_data.edge_index.size(1)} (val), {test_data.edge_index.size(1)} (test)")
# n_users, n_items = data.n_users, data.n_movies

dataset = PredefinedMovieLensDataset("./data")
n_users, n_items = dataset.n_users, dataset.n_movies
train_data, val_data = dataset.train_data, dataset.val_data
print(f"Number of users: {n_users}, Number of items: {n_items}")
config = Config(num_users=n_users, num_items=n_items)
train_dataloader = MovieLensDataloader(train_data, [], batch_size=train_data.edge_index.shape[1] // config.val_check_interval, shuffle=True, exclude_sampling=False)
val_dataloader = MovieLensDataloader(val_data, [train_data], batch_size=val_data.edge_index.shape[1], shuffle=False, exclude_sampling=True)

lightgcn = LightGCNModule(
    num_users=config.num_users,
    num_items=config.num_items,
    n_layers=config.n_layers,
    embedding_dim=config.embedding_dim,
    add_self_loops=config.add_self_loops,
    lambda_val=config.lambda_val,
    k=config.k
)

logger = TensorBoardLogger("./logs", name="lightgcn_movielens")
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = L.Trainer(accelerator=device, max_steps=config.max_steps, logger=logger, val_check_interval=config.val_check_interval, log_every_n_steps=5)
trainer.fit(
    lightgcn,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)