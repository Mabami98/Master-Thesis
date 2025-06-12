import numpy as np
import pickle


def save_adjacency_matrix(adj_matrix, filename='adj_matrix_large.pkl'):
    N = adj_matrix.shape[0]
    
    # 1 Node labels as strings
    node_labels = [str(i) for i in range(N)]
    
    # 2 Mapping from string to index
    node_map = {str(i): i for i in range(N)}
    
    # 3 adjacency matrix
    data_to_save = [node_labels, node_map, adj_matrix]
    
    # Save to file
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)




def generate_rnn_data(data, seq_len=8, pred_len=1):
    """
    Convert (H, W, T, C) tensor into RNN-ready data:
    x: (N, seq_len, H*W, C)
    y: (N, pred_len, H*W, C)
    
    Args:
        data: np.ndarray of shape (H, W, T, C)
        seq_len: int, length of input sequence
        pred_len: int, length of prediction sequence (default 1)
    
    Returns:
        x: np.ndarray of shape (N, seq_len, H*W, C)
        y: np.ndarray of shape (N, pred_len, H*W, C)
        x_offsets: np.ndarray of shape (seq_len, 1)
        y_offsets: np.ndarray of shape (pred_len, 1)
    """
    H, W, T, C = data.shape
    N = T - seq_len - pred_len + 1
    x, y = [], []

    for t in range(N):
        x_seq = data[:, :, t:t+seq_len, :]         # (H, W, seq_len, C)
        y_seq = data[:, :, t+seq_len:t+seq_len+pred_len, :]  # (H, W, pred_len, C)

        # Reshape spatial dims: (H, W, T, C) → (T, H*W, C)
        x.append(x_seq.transpose(2, 0, 1, 3).reshape(seq_len, H * W, C))
        y.append(y_seq.transpose(2, 0, 1, 3).reshape(pred_len, H * W, C))

    x = np.stack(x)  # (N, seq_len, H*W, C)
    y = np.stack(y)  # (N, pred_len, H*W, C)

    x_offsets = np.expand_dims(np.arange(-seq_len, 0, 1), axis=1)  # (seq_len, 1)
    y_offsets = np.expand_dims(np.arange(0, pred_len), axis=1)     # (pred_len, 1)

    return x, y, x_offsets, y_offsets



def generate_adjacency_matrix(H, W, eight_neighbors=False):
    N = H * W
    A = np.zeros((N, N), dtype=int)

    for i in range(H):
        for j in range(W):
            node = i * W + j
            neighbors = []

            # 4 connected neighbors
            if i > 0: neighbors.append(((i - 1) * W + j))      # Up
            if i < H - 1: neighbors.append(((i + 1) * W + j))  # Down
            if j > 0: neighbors.append((i * W + (j - 1)))      # Left
            if j < W - 1: neighbors.append((i * W + (j + 1)))  # Right

            if eight_neighbors:
                # Diagonals neigbors
                if i > 0 and j > 0: neighbors.append(((i - 1) * W + (j - 1)))  # Up-left
                if i > 0 and j < W - 1: neighbors.append(((i - 1) * W + (j + 1)))  # Up-right
                if i < H - 1 and j > 0: neighbors.append(((i + 1) * W + (j - 1)))  # Down-left
                if i < H - 1 and j < W - 1: neighbors.append(((i + 1) * W + (j + 1)))  # Down-right

            for n in neighbors:
                A[node, n] = 1
                A[n, node] = 1 # symmetric matrix

    return A
