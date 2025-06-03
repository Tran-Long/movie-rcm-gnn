import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import convert_adj_coo_to_itr_coo, get_user_positive_items, calculate_metrics, calculate_metrics_contigous
from torch_geometric.utils import structured_negative_sampling
import lightning as L


class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, n_layers=3, embedding_dim=64, add_self_loops=False):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.add_self_loops = add_self_loops
    
        
        # Initialize user and item embeddings
        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights for the embedding layers
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)

    def forward(self, edge_index):
        """_summary_

        Args:
            edge_index (Tensor): Adjacency list of the graph in COO format, shape [2, num_edges].
        Returns:
        """
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        e_0 = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        es = [e_0]
        e_k = e_0
        for _ in range(self.n_layers):
            e_k = self.propagate(edge_index=edge_index_norm[0], x=e_k, norm=edge_index_norm[1])
            es.append(e_k)

        # Aggregate embeddings from all layers
        e = torch.stack(es, dim=1).mean(dim=1)
        user_embdings, item_embeddings = e[:self.num_users], e[self.num_users:]
        return user_embdings, item_embeddings, self.user_embeddings.weight, self.item_embeddings.weight
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCNModule(L.LightningModule):
    def __init__(self, num_users, num_items, n_layers=3, embedding_dim=64, add_self_loops=False, lr=0.001, lambda_val=0.001, k=10):
        super(LightGCNModule, self).__init__()
        self.model = LightGCN(num_users=num_users, num_items=num_items, n_layers=n_layers, embedding_dim=embedding_dim, add_self_loops=add_self_loops)
        self.lambda_val = lambda_val
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        train_edge_index, user_indices, pos_item_indices, neg_item_indices = batch
        user_embs, item_embs, user_embs_init, item_embs_init = self.model(train_edge_index)
        
        user_embs, user_embs_init = user_embs[user_indices], user_embs_init[user_indices]
        pos_items_emb_final, pos_item_embs_init = item_embs[pos_item_indices], item_embs_init[pos_item_indices]
        neg_item_embs, neg_item_embs_init = item_embs[neg_item_indices], item_embs_init[neg_item_indices]
        loss = bpr_loss(
            user_embs, pos_items_emb_final, neg_item_embs,
            user_embs_init, pos_item_embs_init, neg_item_embs_init,
            self.lambda_val
        )
        self.log('train_loss', loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Calculate validation loss
        val_edge_index, exclude_itr_edge_indices, user_indices, pos_item_indices, neg_item_indices = batch
        user_embs, item_embs, user_embs_init, item_embs_init = self.model(val_edge_index)
        user_embs, user_embs_init = user_embs[user_indices], user_embs_init[user_indices]
        pos_items_emb_final, pos_item_embs_init = item_embs[pos_item_indices], item_embs_init[pos_item_indices]
        neg_item_embs, neg_item_embs_init = item_embs[neg_item_indices], item_embs_init[neg_item_indices]
        loss = bpr_loss(
            user_embs, pos_items_emb_final, neg_item_embs,
            user_embs_init, pos_item_embs_init, neg_item_embs_init,
            self.lambda_val
        )
        # Calculate metrics
        user_embs, item_embs = self.model.user_embeddings.weight, self.model.item_embeddings.weight
        score_matrix = user_embs @ item_embs.t()
        precision, recall = calculate_metrics_contigous(
            score_matrix,
            convert_adj_coo_to_itr_coo(val_edge_index, self.hparams.num_users, self.hparams.num_items),
            exclude_itr_edge_indices,
            self.hparams.k
        )
        # precision, recall = calculate_metrics(
        #     score_matrix,
        #     convert_adj_coo_to_itr_coo(val_edge_index, self.hparams.num_users, self.hparams.num_items),
        #     exclude_itr_edge_indices,
        #     self.hparams.k
        # )
        self.log('val_loss', loss, on_step=True, logger=True)
        self.log('val_precision', precision, on_step=True, logger=True)
        self.log('val_recall', recall, on_step=True, logger=True)
        return {'val_loss': loss, 'val_precision': precision, 'val_recall': recall}

    # def on_train_epoch_end(self):
    #     print(f"Epoch {self.current_epoch} completed.")
    #     return super().on_train_epoch_end()

# def calculate_metrics(model, adj_edge_index, exclude_adj_edge_indices, k):
#     user_embeddings = model.user_embeddings.weight
#     item_embeddings = model.item_embeddings.weight
#     itr_edge_index = convert_adj_coo_to_itr_coo(adj_edge_index, model.num_users, model.num_items)
#     exclude_itr_edge_indices = [convert_adj_coo_to_itr_coo(exclude_edge_index, model.num_users, model.num_items) for exclude_edge_index in exclude_adj_edge_indices]

#     scores = torch.matmul(user_embeddings, item_embeddings.t())

#     for exclude_edge_index in exclude_itr_edge_indices:
#         user_pos_items = get_user_positive_items(exclude_edge_index)
#         exclude_users, exclude_movies = [], []
#         for user, pos_items in user_pos_items.items():
#             exclude_users.extend([user] * len(pos_items))
#             exclude_movies.extend(pos_items)
#         scores[exclude_users, exclude_movies] = float('-inf')
#     # Get top-k items for each user
#     _, top_k_items = torch.topk(scores, k, dim=1)
#     validate_user_positive_items = get_user_positive_items(itr_edge_index)


#     users = itr_edge_index[0].unique()
#     is_ins = []
#     n_total = 0
#     for user in users:
#         user_true_relevant_items = validate_user_positive_items.get(user.item(), [])
#         n_total += len(user_true_relevant_items)
#         user_predicted_items = top_k_items[user]
#         is_in = list(map(lambda x: x in user_true_relevant_items, user_predicted_items))
#         is_ins.append(is_in)
#     is_ins = torch.tensor(is_ins, dtype=torch.float32)
#     precision = is_ins.sum(dim=1).mean() / k
#     recall = is_ins.sum(dim=1).mean() / n_total if n_total > 0 else 0
#     return precision.item(), recall.item()


# def evaluation_loop(model, edge_index, exclude_edge_indices, lambda_val, k):
#     users_emb_final, items_emb_final, users_emb_init, items_emb_init = model(edge_index)
#     itr_edge_index = convert_adj_mat_ei_to_interaction_mat_ei(edge_index, model.num_users, model.num_items)
#     edges = structured_negative_sampling(itr_edge_index, contains_neg_self_loops=False)
#     user_indices, pos_item_indices, neg_item_indices = edges
#     users_emb_final, users_emb_init = users_emb_final[user_indices], users_emb_init[user_indices]
#     pos_items_emb_final, pos_items_emb_init = items_emb_final[pos_item_indices], items_emb_init[pos_item_indices]
#     neg_items_emb_final, neg_items_emb_init = items_emb_final[neg_item_indices], items_emb_init[neg_item_indices]

#     loss = bpr_loss(
#         users_emb_final, pos_items_emb_final, neg_items_emb_final,
#         users_emb_init, pos_items_emb_init, neg_items_emb_init,
#         lambda_val
#     )
#     precision, recall = calculate_metrics(model, edge_index, exclude_edge_indices, k)
#     return loss, precision, recall

def bpr_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final,
             users_emb_init, pos_items_emb_init, neg_items_emb_init, lambda_val):
    pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=1)
    # loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))
    # Regularization term
    reg_loss = users_emb_init.norm(2).pow(2) + \
               pos_items_emb_init.norm(2).pow(2) + \
               neg_items_emb_init.norm(2).pow(2)
    
    loss += lambda_val * reg_loss
    return loss