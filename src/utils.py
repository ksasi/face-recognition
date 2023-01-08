from pathlib import Path
import cv2
from numpy.core.fromnumeric import nonzero
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
from itertools import combinations
from kornia.contrib import FaceDetector, FaceDetectorResult
import kornia as K


class detectalign(object):
    def __init__(self, size):
      self.size = size

    def __call__(self, img):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        face_detection = FaceDetector().to(device, torch.float32)
        img = K.utils.image_to_tensor(np.asarray(img)).to(device, torch.float32)
        with torch.no_grad():
            dets = face_detection(img.unsqueeze(0))
        det = [FaceDetectorResult(o) for o in dets]
        x1, y1 = det[0].xmin.int(), det[0].ymin.int()
        x2, y2 = det[0].xmax.int(), det[0].ymax.int()
        roi = img[..., y1:y2, x1:x2]
        if roi.squeeze(0).shape[1] == 0 or roi.squeeze(0).shape[2] == 0:
            img = K.geometry.transform.resize(img, (self.size, self.size))
            return img/255
        else:
            roi = K.geometry.transform.resize(roi, (self.size, self.size))
            return roi/255

    def __repr__(self):
        return "face detect and align augmentation"

class MetricLog():
    """Class to store current and running metrics"""
    def __init__(self) -> None:
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.vlist = []

    def update(self, val):
        self.vlist.append(val)
        self.val = val
        self.sum = self.sum + val
        self.count = self.count + 1
        self.avg = self.sum/self.count

def save_checkpoint(state, filename):
    """Saves the checkpoint for future use"""
    torch.save(state, filename)

def accuracyk(output, actual, topk):
    """Returns topk accuracy metrics"""
    maxk = max(topk)
    batch_size = output.size(0)
    _, predk = torch.topk(output, maxk, 1, True, True)
    predk = predk.t()
    correct = torch.eq(predk, actual.view(1,-1).expand_as(predk))

    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul(100)/batch_size)
    return result

def generate_embeddings(model, root_path, transform):
    path = Path(root_path)
    transform = transform
    img_embeddings = []
    img_labels = []
    for subdir in sorted(path.iterdir()):
        label = subdir.name
        image_path = [x for x in subdir.iterdir()][0]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = transform(image)
        image = torch.unsqueeze(image, 0).cuda()
        model.cuda().eval()
        embedding = model(image)
        img_embeddings.append(embedding[1].detach().cpu().numpy())
        img_labels.append(label)
    return img_labels, img_embeddings


def generate_embeddings_v2(model, root_path, transform):
    path = Path(root_path)
    transform = transform
    img_embeddings = []
    img_labels = []
    for subdir in sorted(path.iterdir()):
        label = subdir.name
        image_path = [x for x in subdir.iterdir()][0]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0).cuda()
        model.cuda().eval()
        embedding = model(image)
        img_embeddings.append(embedding[1].detach().cpu().numpy())
        img_labels.append(label)
    return img_labels, img_embeddings

def cosine_similarity_matrix(probe_embeddings, gallery_embeddings):
    row_size = probe_embeddings.shape[0]
    col_size = gallery_embeddings.shape[0]
    sim_matrix = np.empty([row_size, col_size])
    for row in range(row_size):
      for col in range(col_size):
        sim_matrix[row, col] = cosine_similarity(probe_embeddings[row], gallery_embeddings[col])
    return sim_matrix


def get_cmc_scores(sim_matrix):
    row_size, col_size = sim_matrix.shape
    rank_list = []
    cmc_scores = []
    for row in range(row_size):
      val = sim_matrix[row, :][row]
      rank = np.where(sorted(sim_matrix[row, :], reverse = True) == val)
      rank_list.append(rank[0][0]+1)
    current_score = 0
    for rs in range(row_size):
      current_score = current_score + (np.array(rank_list) == rs+1).sum()
      cmc_scores.append(current_score/row_size)
    return cmc_scores


def plot_cmc_curve(cmc_scores, path, figsize = (8,8)):
    x = np.arange(1,11)
    y = np.array(cmc_scores[0:10])
    plt.figure(figsize=figsize)
    plt.title("CMC Curve")
    plt.xlabel("Rank (m)")
    plt.ylabel("Rank-m Identification Accuracy")
    plt.xlim(1,10)
    plt.xticks(np.arange(1, 11, 1))
    plt.ylim(0,30)
    plt.yticks(np.arange(0, 30, 1))
    plt.plot(x, y*100, color ="green", linestyle='--', marker='o', label='line with marker')
    plt.grid(True)
    plt.legend(loc=2)
    plt.savefig(path, format = 'pdf')
    plt.show()
    plt.close()


def plot_roc_curve():
    pass

def pcosdist(input1, input2, eps = 1e-08):
    input1_n = input1.norm(dim = 1)[:, None]
    input2_n = input2.norm(dim = 1)[:, None]
    input1_norm = input1/torch.max(input1_n, eps*torch.ones_like(input1_n))
    input2_norm = input1/torch.max(input2_n, eps*torch.ones_like(input2_n))
    cosdist_mt = torch.mm(input1_norm, torch.t(input2_norm))
    return cosdist_mt


class pairselector():
    def __init__(self):
        super(pairselector, self).__init__()
        pass

    def get_pairs(self, embeddings, labels):
        cosdist_mt = pcosdist(embeddings, embeddings)
        labels = labels.cpu().numpy()
        pairs = np.array(list(combinations(range(len(labels)), 2)))
        pairs = torch.LongTensor(pairs)
        pos_pairs = pairs[(labels[pairs[:, 0]] == labels[pairs[:, 1]]).nonzero()]
        neg_pairs = pairs[(labels[pairs[:, 0]] != labels[pairs[:, 1]]).nonzero()]

        neg_cosdist = cosdist_mt[neg_pairs[:, 0], neg_pairs[:, 1]]
        neg_cosdist = -neg_cosdist.cpu().detach().numpy()

        top_negs = np.argpartition(neg_cosdist, len(pos_pairs))[:len(pos_pairs)]
        top_neg_pairs = neg_pairs[torch.LongTensor(top_negs)]
        return pos_pairs, top_neg_pairs



