from dataclasses import dataclass
from dataset import MovieLens, train_val_test_split, MovieLensDataloader, TransformRatings
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
    batch_size: int = 64
    learning_rate: float = 0.001
    max_steps: int = 200
    val_check_interval: int = 1

rating_transform = TransformRatings()
data = MovieLens("./data", transform=rating_transform, user_subset=None).get()
train_data, val_data, test_data = train_val_test_split(data)
n_users, n_items = data.n_users, data.n_movies
print(f"Number of users: {n_users}, Number of items: {n_items}")
config = Config(num_users=n_users, num_items=n_items)
train_dataloader = MovieLensDataloader(train_data, [], batch_size=config.batch_size, shuffle=True, exclude_sampling=False)
val_dataloader = MovieLensDataloader(val_data, [train_data], batch_size=val_data.edge_index.size(1), shuffle=False, exclude_sampling=True)

lightgcn = LightGCNModule(
    num_users=config.num_users,
    num_items=config.num_items,
    n_layers=config.n_layers,
    embedding_dim=config.embedding_dim,
    add_self_loops=config.add_self_loops,
    lambda_val=config.lambda_val,
    k=config.k
)

logger = TensorBoardLogger("./logs", name="lightgcn_movielens", version="v1")
trainer = L.Trainer(accelerator="cpu", max_steps=config.max_steps, logger=logger, val_check_interval=config.val_check_interval)
trainer.fit(
    lightgcn,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)