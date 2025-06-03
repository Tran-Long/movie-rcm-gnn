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
    if len(exclude_itr_coos):
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
        is_in = [item in user_true_relevant_items for item in user_predicted_items]
        true_positive += sum(is_in)
        total_pred += len(is_in)
    precision = true_positive / total_pred if total_pred > 0 else 0
    recall = true_positive / total_label if total_label > 0 else 0
    return precision * 100, recall * 100


def calculate_metrics_vectorized(user_item_score, val_itr_coo, exclude_itr_coos, k):
    device = user_item_score.device
    N_users, N_items = user_item_score.shape

    # Step 1: Exclude known interactions
    if exclude_itr_coos:
        exclude = torch.cat(exclude_itr_coos, dim=1)
        user_item_score[exclude[0], exclude[1]] = float('-inf')

    # Step 2: Get top-K items for all users
    _, topk_items = torch.topk(user_item_score, k=k, dim=1)  # shape: (N_users, K)

    # Step 3: Build dict of user -> [relevant items]
    user_ids = val_itr_coo[0]
    item_ids = val_itr_coo[1]
    val_user_positive_items = {}
    for u, i in zip(user_ids.tolist(), item_ids.tolist()):
        val_user_positive_items.setdefault(u, []).append(i)

    # Step 4: Compute precision & recall over users with ground truth
    true_positives, total_pred, total_label = 0, 0, 0

    for user, gt_items in val_user_positive_items.items():
        pred_items = topk_items[user].tolist()
        hits = sum([i in gt_items for i in pred_items])
        true_positives += hits
        total_pred += k
        total_label += len(gt_items)

    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_label if total_label > 0 else 0
    return precision * 100, recall * 100

def calculate_metrics_contigous(user_item_score, val_itr_coo, exclude_itr_coos, k):
    """
    Fully vectorized GPU version of Precision@K and Recall@K.
    Assumes user IDs are contiguous in [0, N-1].
    """
    device = user_item_score.device
    N_users, N_items = user_item_score.shape

    # Step 1: Mask out known interactions (e.g., training data)
    if exclude_itr_coos:
        exclude = torch.cat(exclude_itr_coos, dim=1)
        user_item_score[exclude[0], exclude[1]] = float('-inf')

    # Step 2: Get top-K items for each user
    topk_items = torch.topk(user_item_score, k=k, dim=1).indices  # shape: [N_users, K]

    # Step 3: Create dense relevance matrix from val_itr_coo
    relevance = torch.sparse_coo_tensor(
        val_itr_coo,
        torch.ones(val_itr_coo.shape[1], dtype=torch.float32, device=device),
        size=(N_users, N_items),
        device=device
    ).to_dense()  # shape: [N_users, N_items]

    # Step 4: Gather top-K relevance for each user
    user_idx = torch.arange(N_users, device=device).unsqueeze(1).expand(-1, k)  # shape: [N_users, K]
    topk_relevance = relevance[user_idx, topk_items]  # shape: [N_users, K]

    # Step 5: Count true positives, labels, and predictions
    hits_per_user = topk_relevance.sum(dim=1)  # shape: [N_users]
    label_count_per_user = relevance.sum(dim=1).clamp(min=1)  # avoid div-by-zero

    precision = (hits_per_user / k).mean().item()
    recall = (hits_per_user / label_count_per_user).mean().item()

    return precision * 100, recall * 100


if __name__ == "__main__":
    N, M, K = 100, 1000, 10
    for _ in range(10):
        score_matrix = torch.rand(N, M)
        val_itr_coo = torch.tensor(
            [torch.arange(0, N).numpy().tolist() * 10,
            torch.randint(0, M, size=(N*10,)).numpy().tolist()],
            dtype=torch.long
        )
        precision1, recall1 = calculate_metrics(score_matrix, val_itr_coo, [], K)
        precision2, recall2 = calculate_metrics_contigous(score_matrix, val_itr_coo, [], K)
        assert abs(precision1 - precision2) < 1e-6, f"Precision mismatch: {precision1} != {precision2}"
        assert abs(recall1 - recall2) < 1e-6, f"Recall mismatch: {recall1} != {recall2}"
        print("True")