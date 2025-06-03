import torch

def convert_itr_coo_to_adj_coo(itr_coo, n_users, n_items):
    R = torch.zeros((n_users, n_items))
    for i in range(itr_coo.shape[1]):
        user, item = itr_coo[0, i], itr_coo[1, i]
        R[user, item] = 1  # Assuming binary interactions; change if needed
    R_T = R.t()
    adj_mat = torch.zeros((n_users + n_items, n_users + n_items))
    adj_mat[:n_users, n_users:] = R.clone()
    adj_mat[n_users:, :n_users] = R_T.clone()
    adj_coo = adj_mat.to_sparse_coo().indices()
    return adj_coo

def convert_adj_coo_to_itr_coo(adj_coo, n_users, n_items):
    sparse_adj_mat = torch.sparse_coo_tensor(
        adj_coo, 
        torch.ones(adj_coo.shape[1]).to(adj_coo.device), 
        (n_users + n_items, n_users + n_items)
    )
    adj_mat = sparse_adj_mat.to_dense()
    itr_coo = adj_mat[:n_users, n_users:].to_sparse_coo().indices()
    return itr_coo

def get_user_positive_items(itr_coos: list[torch.Tensor]) -> dict:
    user_positive_items = {}
    for itr_coo in itr_coos:
        for i in range(itr_coo.shape[1]):
            user, item = itr_coo[0, i].item(), itr_coo[1, i].item()
            if user not in user_positive_items:
                user_positive_items[user] = []
            user_positive_items[user].append(item)
    return user_positive_items

def calculate_metrics(user_item_score, val_itr_coo, exclude_itr_coos, k):
    exclude_itr_coo = torch.cat(exclude_itr_coos, dim=1)
    user_item_score[exclude_itr_coo[0], exclude_itr_coo[1]] = float('-inf')
    _, top_k_items = torch.topk(user_item_score, k, dim=1)

    val_user_positive_items = get_user_positive_items([val_itr_coo])
    users = val_itr_coo[0].unique()
    true_positive, total_pred, total_label = 0, 0, 0 
    
    for user in users:
        user_true_relevant_items = val_user_positive_items.get(user.item(), [])
        total_label += len(user_true_relevant_items)
        user_predicted_items = top_k_items[user]
        is_in = list(map(lambda x: x in user_true_relevant_items, user_predicted_items))
        true_positive += sum(is_in)
        total_pred += len(is_in)
        total_label += len(user_true_relevant_items)
    precision = true_positive / total_pred if total_pred > 0 else 0
    recall = true_positive / total_label if total_label > 0 else 0
    return precision * 100, recall * 100