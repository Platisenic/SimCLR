import torch
import torch.nn.functional as F

batch_size = 4
n_views = 2
device = 'cpu'
temperature = 0.1

def info_nce_loss(features):
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    print(labels)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    print(labels)
    print(similarity_matrix)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    print(similarity_matrix)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

if __name__  == '__main__':
    x = torch.randn(batch_size*2, 5)
    print('x :', x.shape)
    logits, labels = info_nce_loss(x)
    print('logits :', logits.shape)
    print(logits)
    print('labels :', labels.shape)
    print(labels)
