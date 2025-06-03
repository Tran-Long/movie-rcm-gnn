import os.path as osp
import zipfile
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, download_url
from torch_geometric.utils import structured_negative_sampling
from utils import convert_itr_coo_to_adj_coo

class TransformRatings:
    def __init__(self, threshold=3.5):
        """
        Transform function that assign non-negative entries >= thres 1, and non-
        negative entries <= thres 0. Keep other entries the same.
        """
        self.threshold = threshold

    def __call__(self, data: Data):
        rating_mat = data['raw_edge_index']
        rating_mat[(rating_mat < self.threshold) & (rating_mat > -1)] = 0
        rating_mat[(rating_mat >= self.threshold)] = 1
        itr_coo = rating_mat.nonzero(as_tuple=False).t().contiguous().long()
        data['edge_index'] = itr_coo
        return data

class MovieLens(Dataset):
    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    def __init__(self, root, transform=None, pre_transform=None, transform_args=None, pre_transform_args=None, user_subset=None):
        """
        root = where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (process data).
        """
        self.user_subset = user_subset
        super(MovieLens, self).__init__(root, transform, pre_transform)
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args
    
    @property
    def raw_file_names(self):
        return "ml-1m.zip"

    @property
    def processed_file_names(self):
        return [f"data_movielens_{self.user_subset}.pt" if self.user_subset is not None else "data_movielens.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_url(self.DATA_URL, self.raw_dir)

    def _load(self):
        with zipfile.ZipFile(self.raw_paths[0], 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_csv(self.raw_dir+'/ml-1m/users.dat', 
                              sep='::', header=None, names=unames,
                              engine='python', encoding='latin-1')
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(self.raw_dir+'/ml-1m/ratings.dat', sep='::', 
                                header=None, names=rnames, engine='python',
                                encoding='latin-1')
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(self.raw_dir+'/ml-1m/movies.dat', sep='::', 
                               header=None, names=mnames, engine='python',
                               encoding='latin-1')

        return users, ratings, movies

    def process(self):
        # load information from file
        users_df, ratings_df, movies_df = self._load()

        users_df = users_df['user_id']
        movies_df = movies_df['movie_id']

        if self.user_subset is not None:
            # SAMPLE a subset of users only
            if isinstance(self.user_subset, float) and 0 < self.user_subset <= 1:
                n_users = int(self.user_subset * len(users_df))
            elif isinstance(self.user_subset, int):
                n_users = self.user_subset
            else:
                raise ValueError("user_subset should be a float in (0, 1] or an integer.")
            users_df = users_df.sample(n=n_users, random_state=42).reset_index(drop=True)   

        user_ids = range(len(users_df))
        movie_ids = range(len(movies_df))

        user_to_id = dict(zip(users_df, user_ids))
        movie_to_id = dict(zip(movies_df, movie_ids))

        # get adjacency info
        self.num_users = users_df.shape[0]
        self.num_movies = movies_df.shape[0]

        # initialize the adjacency matrix
        rat = torch.zeros(self.num_users, self.num_movies)
        for index, row in ratings_df.iterrows():
            user, movie, rating = row[:3]
            if user not in user_to_id:
                continue
            # create ratings matrix where (i, j) entry represents the ratings
            # of movie j given by user i.
            rat[user_to_id[user], movie_to_id[movie]] = rating
            
        
        # create Data object
        coo_edge_index = None # Will be change with the transform
        data = Data(
            edge_index = coo_edge_index,
            raw_edge_index = rat.clone(),
            data = ratings_df,
            users = users_df,
            movies = movies_df,
            n_users = self.num_users,
            n_movies = self.num_movies,
        )

        data = self.transform(data)
        torch.save(data, osp.join(self.processed_dir, f"data_movielens_{self.user_subset}.pt" if self.user_subset is not None else "data_movielens.pt"))
      
    def len(self):
        """
        return the number of examples in your graph
        """
        return len(self.processed_file_names)

    def get(self):
        """
        The logic to load a single graph
        """
        data = torch.load(osp.join(self.processed_dir, f"data_movielens_{self.user_subset}.pt" if self.user_subset is not None else "data_movielens.pt"), weights_only=False)
        return data

def train_val_test_split(data: Data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    """
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    all_indices = np.arange(num_edges)
    train_indices, test_indices = train_test_split(all_indices, test_size=val_ratio+test_ratio, random_state=37)
    val_indices, test_indices = train_test_split(test_indices, test_size=test_ratio/(val_ratio+test_ratio), random_state=37)
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    train_data = Data(edge_index=train_edge_index, n_users=data.n_users, n_movies=data.n_movies)
    val_data = Data(edge_index=val_edge_index, n_users=data.n_users, n_movies=data.n_movies)
    test_data = Data(edge_index=test_edge_index, n_users=data.n_users, n_movies=data.n_movies)
    return train_data, val_data, test_data


def sample_mini_batch(data, batch_size=32):
    """
    Sample a mini-batch of data.
    """
    edge_index = data
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = torch.randperm(edges.size(1))[:batch_size]
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch
    return user_indices, pos_item_indices, neg_item_indices

class MovieLensDataloader:
    def __init__(self, data: Data, exclude_data: list[Data], batch_size=32, shuffle=True, exclude_sampling=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        self.exclude_data = exclude_data
        self.exclude_itr_edge_indices = [d.edge_index for d in exclude_data]
        self.exclude_sampling = exclude_sampling
        self.itr_edge_index = self.data.edge_index
        self.adj_edge_index = convert_itr_coo_to_adj_coo(self.itr_edge_index, self.data.n_users, self.data.n_movies)

    def __iter__(self):
        """
        Return an iterator that yields batches of data.
        Each batch contains the adjacency edge index, user indices, positive item indices, and negative item indices.
        - The adjacency edge index is the adj form to calculate the embeddings
        - The indices are sampled from the interaction edge index, because the model returns embeddings via users & items
        """
        indices = torch.randperm(self.itr_edge_index.size(1)) if self.shuffle else torch.arange(self.itr_edge_index.size(1))
        for start in range(0, len(indices), self.batch_size):
            edges_w_neg = structured_negative_sampling(self.itr_edge_index, num_nodes=self.data.n_movies, contains_neg_self_loops=False)
            edges_w_neg = torch.stack(edges_w_neg, dim=0)
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            batch = edges_w_neg[:, batch_indices]
            user_indices, pos_item_indices, neg_item_indices = batch
            if self.exclude_sampling and len(self.exclude_itr_edge_indices):
                yield self.adj_edge_index, self.exclude_itr_edge_indices, user_indices, pos_item_indices, neg_item_indices
            else:
                yield self.adj_edge_index, user_indices, pos_item_indices, neg_item_indices