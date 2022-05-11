import os
import shutil
import torch.nn.functional as F
import torch
# import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# def save_config_file(model_checkpoints_folder, args):
#     if not os.path.exists(model_checkpoints_folder):
#         os.makedirs(model_checkpoints_folder)
#         with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
#             yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def info_nce_loss(batch_size, features,device,real_labels):
    labels = real_labels
   # labels = torch.cat([torch.arange(batch_size) for i in range(1)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
   # labels = labels.to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    test=similarity_matrix.shape[0]
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)

    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    bool_check = similarity_matrix[labels.bool()]
    positives = similarity_matrix[0].view(labels.shape[1], -1)
    #positives = similarity_matrix[labels.bool()]
    #positives = []
    #positives.append(similarity_matrix[labels.bool()].view(-1))

    # select only the negatives the negatives
    boolne_check = similarity_matrix[~labels.bool()]
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / 0.07

    return logits, labels

