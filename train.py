import itertools
import os
import pickle
import random
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import hadamard
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable

from network import CrossModalNet
from utils.tools import (calc_map_k, compute_img_result, compute_tag_result,
                         get_data,pr_curve)



def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config():
    config = {
        "optimizer": {
            "type": optim.Adam,
            "optim_params": {"lr": lr, "weight_decay": weight_decay},
        },

        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": batch_size,
        "dataset": dataset,
        "epoch": epoch,
        "device": torch.device("cuda:"+args.gpus),
        "bit_len": bit_len,
        "noise_type": "symmetric",
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "tag_len": tag_len,
        "train_size": train_size,
        "alpha" :alpha,
        "Lambda" :Lambda,
        "k" :k,
        "save_model":save_model,
        "lambda_contrast": lambda_contrast,
        "lambda_pairwise": lambda_pairwise,
        "lambda_class": lambda_class,
        "temperature": temperature,
        "is_PR":is_PR,

    }
    return config


# =============================================================================
# Loss function classes
# =============================================================================

class MultiLabelCrossModalLoss(nn.Module):
    """
    Unified multi-label cross-modal loss class, integrating contrastive loss, quantization loss, classification loss and inter-class loss.
    """
    def __init__(
        self,
        lambda_contrast=0.7,
        lambda_class=0.5,
        lambda_pairwise=0.5,
        tau=1.0,
        min_cluster_size=1,
        temperature=1.0
    ):
        super(MultiLabelCrossModalLoss, self).__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_class = lambda_class
        self.lambda_pairwise = lambda_pairwise
        self.tau = tau
        self.min_cluster_size = min_cluster_size
        self.temperature = temperature

    def compute_label_similarity(self, label):
        """
        Compute the label similarity matrix.
        """
        intersection = torch.matmul(label, label.T)
        union = (
            torch.sum(label, dim=1, keepdim=True)
            + torch.sum(label, dim=1, keepdim=True).T
            - intersection
        )
        return intersection / (union + 1e-8)
    

    def contrast_loss(self, u, v, label, eps=1e-8, tau=0.5):
        """
        Contrastive loss: pull samples with similar labels closer and push samples with dissimilar labels apart.
        """
        S = torch.matmul(u, v.T)  # feature similarity [N, N]
        T = self.compute_label_similarity(label)  # label similarity [N, N]

        pos_mask = (T > 0).float()
        neg_mask = (T <= 0).float()

        # Positive sample loss: encourage samples with similar labels to be similar
        pos_sim = S * pos_mask
        pos_loss = -pos_sim.sum() / (pos_mask.sum() + eps) if pos_mask.sum() > 0 else 0.0

        # Negative sample loss: penalize samples with dissimilar labels but similar features
        neg_sim = S * neg_mask
        neg_loss = (neg_sim ** 2).sum() / (neg_mask.sum() + eps) if neg_mask.sum() > 0 else 0.0

        # Optional: encourage consistency of each sample's own features
        self_sim_loss = -torch.diag(S).mean()

        loss = self_sim_loss + pos_loss + neg_loss
        return loss

    def quant_loss(self, u, v, label=None, label_confidence=None):
        """
        Quantization loss function.
        """
        loss_u = torch.norm(u - u.sign(), p=1) / u.numel()
        loss_v = torch.norm(v - v.sign(), p=1) / v.numel()
        return loss_u + loss_v

    def class_loss(self, u, v, label, cluster_centers=None):
        """
        Inter-class loss function.
        u, v: Tensor, shape [N, D], dual-modal/view sample features.
        label: Tensor, shape [N, C], multi-hot encoding, label[i, c] = 1 means sample i belongs to cluster c.
        cluster_centers: Tensor, shape [C, D], optional cluster center initialization.

        Returns:
        total_loss: scalar Tensor, weighted loss including intra-cluster aggregation + inter-cluster repulsion.
        """
        # ======================
        # 1. Feature fusion and normalization
        # ======================
        features = F.normalize(torch.cat([u, v], dim=1), p=2, dim=1)  # [N, 2D]
        N, D = features.shape
        C = label.shape[1]  # number of clusters

        # ======================
        # 2. Prototype computation: use soft weights (based on similarity weights)
        # ======================

        # Step 1: Initialize cluster prototype centers
        if cluster_centers is None:
            # If cluster centers are not provided, initialize with random projection of features
            with torch.no_grad():
                # Use label information to initialize cluster centers for better quality
                # Compute the mean feature of each cluster as the initial center
                cluster_counts = label.sum(dim=0)  # [C]
                cluster_features_sum = torch.matmul(label.t(), features)  # [C, D]
                # For clusters with no samples, use random initialization
                cluster_centers = torch.where(
                    cluster_counts.unsqueeze(1) > 0,
                    cluster_features_sum / cluster_counts.unsqueeze(1),
                    torch.randn(C, D, device=features.device)
                )
                cluster_centers = F.normalize(cluster_centers, p=2, dim=1)

        # Step 2: Compute the similarity between each sample and each cluster center (cosine similarity)
        similarities = torch.matmul(features, cluster_centers.t())  # [N, C]

        # Step 3: Use sigmoid to get each sample's weight for each cluster (soft assignment)
        weights = torch.sigmoid(similarities / self.temperature)  # [N, C]

        # Step 4: Weighted sum -> weighted prototype of each cluster
        weighted_sum = torch.matmul(weights.t(), features)  # [C, D]
        sum_weights = torch.clamp(weights.sum(dim=0), min=1e-6)  # [C], total weight of each cluster
        prototypes = weighted_sum / sum_weights.unsqueeze(1)  # [C, D], weighted cluster prototypes

        # ======================
        # 3. Filter out clusters with too few samples for numerical stability
        # ======================
        valid_clusters = sum_weights >= self.min_cluster_size
        if valid_clusters.sum() == 0:
            # If no valid clusters, return zero loss
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        valid_prototypes = prototypes[valid_clusters]             # [C_valid, D]
        # valid_weights = weights[:, valid_clusters]                # [N, C_valid]
        valid_labels = label[:, valid_clusters]                   # [N, C_valid]

        # ======================
        # 4. Intra-cluster aggregation loss: pull samples toward the weighted prototypes of their assigned clusters
        # ======================
        # Compute the cosine similarity between each sample and its cluster prototypes
        # Expand dimensions for computation
        features_expanded = features.unsqueeze(1)  # [N, 1, D]
        valid_prototypes_expanded = valid_prototypes.unsqueeze(0)  # [1, C_valid, D]
        
        # Compute cosine similarity in [-1, 1]; larger values mean more similar
        cos_similarities = F.cosine_similarity(
            features_expanded, 
            valid_prototypes_expanded, 
            dim=2
        )  # [N, C_valid]
        
        # For each sample, only consider the clusters it belongs to.
        # valid_labels marks the clusters each sample belongs to.
        # We want the similarity between a sample and its cluster prototypes to be as large as possible,
        # so the loss is (1 - similarity); the larger the similarity, the smaller the loss.
        intra_cluster_loss = (1 - cos_similarities) * valid_labels  # [N, C_valid]
        
        # Only compute loss for valid samples (with cluster assignments)
        valid_sample_mask = valid_labels.sum(dim=1) > 0  # [N]
        if valid_sample_mask.sum() > 0:
            # For each sample, compute the weighted average loss over its assigned clusters
            sample_losses = intra_cluster_loss.sum(dim=1) / (valid_labels.sum(dim=1) + 1e-6)  # [N]
            cluster_loss = (sample_losses * valid_sample_mask).sum() / (valid_sample_mask.sum() + 1e-6)
        else:
            cluster_loss = torch.tensor(0.0, device=features.device, requires_grad=True)

        # ======================
        # 5. Inter-cluster repulsion loss: keep different cluster prototypes apart
        # ======================
        num_valid_clusters = valid_prototypes.shape[0]
        if num_valid_clusters > 1:
            # Compute cosine similarity between valid cluster prototypes
            prototype_similarities = torch.matmul(
                valid_prototypes, 
                valid_prototypes.t()
            )  # [C_valid, C_valid]
            
            # Only consider the upper triangular part (excluding the diagonal) to avoid duplicate computation
            mask = torch.triu(torch.ones_like(prototype_similarities, dtype=torch.bool), diagonal=1)
            upper_triangular_similarities = prototype_similarities[mask]  # [M]
            
            # We want inter-cluster similarity to be as small as possible (negative); use margin loss
            # margin is set to 0.1, i.e., inter-cluster similarity should not exceed 0.1
            margin = 0.1
            inter_cluster_loss = torch.clamp(upper_triangular_similarities - margin, min=0).mean()
        else:
            inter_cluster_loss = torch.tensor(0.0, device=features.device, requires_grad=True)

        # ======================
        # 6. Total loss = intra-cluster loss + inter-cluster loss
        # ======================
        total_loss = 0.3 * cluster_loss + 0.7 * inter_cluster_loss

        return total_loss
    
    def pairwise_ranking_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction: str = 'mean'):
        """
        Multi-label Pairwise Ranking BCE Loss.
        Train (positive, negative) pairs with BCE, target = 1.
        """
        pos_mask = labels.bool()
        neg_mask = ~pos_mask

        logits = logits.unsqueeze(2)
        score_diff = logits - logits.transpose(1, 2)   # (B, C, C)

        mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        target = torch.ones_like(score_diff)

        # BCEWithLogitsLoss already includes sigmoid
        loss_mat = F.binary_cross_entropy_with_logits(
            score_diff, target, reduction='none')

        loss_mat = loss_mat * mask.float()
        num_pairs = mask.sum(dim=(1, 2)).clamp_min(1)
        loss = loss_mat.sum(dim=(1, 2)) / num_pairs

        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        return loss


    def stability_loss(self, sample_stab):
        """
        Sample stability loss.
        """
        return F.mse_loss(sample_stab, torch.ones_like(sample_stab))

    def forward(self, u, v, label, u_logits, v_logits, image=None, tag=None, 
                sample_stab=None, clean_sample=None, cluster_centers=None):
        """
        Forward pass, computing the weighted sum of all losses.
        
        Args:
        u, v: hash codes of image and text
        label: ground-truth labels
        u_logits, v_logits: classification logits of image and text
        p_u, p_v: classification probabilities of image and text
        img_feature, txt_feature: image and text features generated by the generator
        image, tag: original image and text features
        sample_stab: sample stability values
        clean_sample: clean sample indicators
        cluster_centers: cluster centers
        """
        # Contrastive loss
        loss_contrast = self.contrast_loss(u, v, label)
        
        # Inter-class loss
        loss_class = self.class_loss(u, v, label, cluster_centers)
        
        # Classification loss (Pairwise Ranking Loss)
        loss_cls_u = self.pairwise_ranking_loss(u_logits, label)
        loss_cls_v = self.pairwise_ranking_loss(v_logits, label)
        loss_cls = (loss_cls_u + loss_cls_v) / 2
        
        # Total loss
        total_loss = (
            self.lambda_contrast * loss_contrast + 
            self.lambda_class * loss_class + 
            self.lambda_pairwise * loss_cls
        )
        
            

        
        return total_loss




# =============================================================================
# Training and testing functions
# =============================================================================

def train(config, bit, seed,aa):

    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = (
        get_data(config)
    )
    config["num_train"] = num_train

    tag_len=config["tag_len"]

    net= CrossModalNet(img_feature_dim=4096, txt_feature_dim=tag_len, bit=bit,num_classes=n_class,hiden_layer=3).to(device)

    

    optimizer =   config["optimizer"]["type"](
        net.parameters(), **(config["optimizer"]["optim_params"])
    )
    
    criterion = MultiLabelCrossModalLoss(
        lambda_contrast=config["lambda_contrast"],
        lambda_class=config["lambda_class"],
        lambda_pairwise=config["lambda_pairwise"],
        temperature=config["temperature"]
    )

    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    bestt2i = 0
    besti2t = 0

    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./other", exist_ok=True)
    os.makedirs("./PR", exist_ok=True)




    with open(
        "./logs/data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
            config["dataset"],
            seed,
            config["noise_rate"],
            bit,
        ),
        "w",
    ) as f:
        for epoch in range(config["epoch"]):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print(
                "%s[%2d/%2d][%s] bit:%d, dataset:%s, training...."
                % (
                    config["info"],
                    epoch + 1,
                    config["epoch"],
                    current_time,
                    bit,
                    config["dataset"],
                ),
                end="",
            )

            net.train()

            for i, (image, tag, tlabel, label, ind) in enumerate(train_loader):
                image = image.float().to(device)
                tag = tag.float().to(device)
                label = label.float().to(device)
                optimizer.zero_grad()

                u,v,u_prob_logits,v_prob_logits = net(image,tag)



                # Compute the total loss using the unified loss function
                loss = criterion(
                    u=u,
                    v=v,
                    label=label,
                    u_logits=u_prob_logits,
                    v_logits=v_prob_logits,
                    image=image,
                    tag=tag,
                )

                loss.backward()
                optimizer.step()
            net.eval()

            if (epoch + 1) % 1 == 0:
                print("calculating test binary code......")
                img_tst_binary, img_tst_label = compute_img_result(
                    test_loader, net, device=device
                )
                print("calculating dataset binary code.......")
                img_trn_binary, img_trn_label = compute_img_result(
                    dataset_loader, net, device=device
                )
                txt_tst_binary, txt_tst_label = compute_tag_result(
                    test_loader, net, device=device
                )
                txt_trn_binary, txt_trn_label = compute_tag_result(
                    dataset_loader, net, device=device
                )
                print("calculating map.......")
                t2i_mAP = calc_map_k(
                    img_trn_binary.numpy(),
                    txt_tst_binary.numpy(),
                    img_trn_label.numpy(),
                    txt_tst_label.numpy(),
                    device=device,
                )
                                

                i2t_mAP = calc_map_k(
                    txt_trn_binary.numpy(),
                    img_tst_binary.numpy(),
                    txt_trn_label.numpy(),
                    img_tst_label.numpy(),
                    device=device,
                )
                if config["is_PR"]:
                    t2i_r, t2i_p = pr_curve(
                        img_trn_binary.numpy(),
                        txt_tst_binary.numpy(),
                        img_trn_label.numpy(),
                        txt_tst_label.numpy(),
                        device=device,
                    )
                    i2t_r, i2t_p = pr_curve(
                        txt_trn_binary.numpy(),
                        img_tst_binary.numpy(),
                        txt_trn_label.numpy(),
                        img_tst_label.numpy(),
                        device=device,
                    )
                if t2i_mAP + i2t_mAP > bestt2i + besti2t:
                    bestt2i = t2i_mAP
                    besti2t = i2t_mAP
                    # Save the model
                    if config["save_model"]:
                        model_dict = {
                            'net_state_dict': net.state_dict(),
                        }
                        model_path = "./checkpoint/model_{}_seed_{}_noiseRate_{}_bit_{}.pth".format(
                            config["dataset"],
                            seed,
                            config["noise_rate"],
                            bit
                        )
                        torch.save(model_dict, model_path)
                    if config["is_PR"]:

                        bestt2i_r = t2i_r
                        bestt2i_p = t2i_p
                        besti2t_r = i2t_r
                        besti2t_p = i2t_p
                        data_to_save = {
                            "bestt2i_r": bestt2i_r,
                            "bestt2i_p": bestt2i_p,
                            "besti2t_r": besti2t_r,
                            "besti2t_p": besti2t_p,
                        }
                        with open(
                            "./PR/data_{}_seed_{}_noiseRate_{}_bit_{}_best_PR.pkl".format(
                                config["dataset"],
                                seed,
                                config["noise_rate"],
                                bit,
                            ),
                            "wb",
                        ) as f1:
                            pickle.dump(data_to_save, f1)
                t2i_mAP_list.append(t2i_mAP.item())
                i2t_mAP_list.append(i2t_mAP.item())
                epoch_list.append(epoch)
                print(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f \n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
                f.writelines(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.1f,t2i_mAP:%.3f, i2t_mAP:%.3f\n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
            
            print("loss:{}".format(loss / len(train_loader)))


        f.writelines(
            f"best result : bit:{bit}, dataset:{config['dataset']}, noise_rate:{config['noise_rate']:.1f}, t2i_mAP:{bestt2i:.3f}, i2t_mAP:{besti2t:.3f}, average:{(besti2t + bestt2i) / 2.0 * 100.0:.1f}\n"
        )


def test(config, bit, model_path="./checkpoint/best_model.pth"):
    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)

    net= CrossModalNet(img_feature_dim=4096, txt_feature_dim=tag_len, bit=bit,num_classes=n_class,hiden_layer=3).to(device)

    # Load the saved models
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net_state_dict"])
    net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(
        dataset_loader, net, device=device
    )
    txt_tst_binary, txt_tst_label = compute_tag_result(
        test_loader, net, device=device
    )
    txt_trn_binary, txt_trn_label = compute_tag_result(
        dataset_loader, net, device=device
    )
    print("calculating map.......")
    t2i_mAP = calc_map_k(
        img_trn_binary.numpy(),
        txt_tst_binary.numpy(),
        img_trn_label.numpy(),
        txt_tst_label.numpy(),
        device=device,
    )
    i2t_mAP = calc_map_k(
        txt_trn_binary.numpy(),
        img_tst_binary.numpy(),
        txt_trn_label.numpy(),
        img_tst_label.numpy(),
        device=device,
    )
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))


# =============================================================================
# Main program entry
# =============================================================================

if __name__ == "__main__":
    # Set random seed
    parser = ArgumentParser(description="manual to this script")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--hash_dim", type=int, default=32)
    parser.add_argument("--noise_rate", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--num_gradual", type=int, default=100)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--Lambda", type=float, default=0.9)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_contrast", type=float, default=0.3)
    parser.add_argument("--lambda_class", type=float, default=0.5)
    parser.add_argument("--lambda_pairwise", type=float, default=1)
    parser.add_argument("--is_PR", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    bit_len = args.hash_dim
    noise_rate = args.noise_rate
    dataset = args.dataset
    num_gradual = args.num_gradual
    k=args.k
    alpha = args.alpha
    Lambda = args.Lambda
    save_model=args.save_model
    lr=args.lr
    weight_decay=args.weight_decay
    epoch=args.epoch
    batch_size=args.batch_size
    lambda_contrast=args.lambda_contrast
    lambda_class=args.lambda_class
    lambda_pairwise=args.lambda_pairwise
    is_PR=args.is_PR
    temperature=args.temperature

    if dataset == "flickr":
        train_size = 10000
    elif dataset == "ms-coco":
        train_size = 10000
    elif dataset == "nuswide21":
        train_size = 10500
    elif dataset == "iapr":
        train_size = 10000
    n_class = 0
    tag_len = 0

    torch.multiprocessing.set_sharing_strategy("file_system")
    
    data_name_list = ["flickr", "nuswide21", "ms-coco","iapr"]
    data_name_list = [ "nuswide21", "ms-coco","iapr"]
    bit_list = [32,64,128]
    noise_rate_list = [0.0,0.2,0.5,0.8]
    for rand_num in [123]:
        for data_name in data_name_list:
            for rate in noise_rate_list:
                for bit in bit_list:
                    for a in [0]:
                        setup_seed(rand_num)
                        bit_len = bit
                        noise_rate = rate
                        dataset = data_name
                        if dataset == "nuswide21":
                            n_class = 21
                            tag_len = 1000

                            lambda_contrast = 0.3
                            lambda_class = 0.5
                            lambda_pairwise = 1

                            min_cluster_size = 1 
                            temperature =0.2

                        elif dataset == "flickr":
                            n_class = 24
                            tag_len = 1386

                            lambda_contrast = 0.3
                            lambda_class = 0.5
                            lambda_pairwise = 1

                            min_cluster_size = 1

                        elif dataset == "ms-coco":
                            n_class = 80
                            tag_len = 300

                              # lambda_contrast = 0.05
                            lambda_contrast = 0.3
                            # lambda_class = 0.8
                            lambda_class = 0.9
                            lambda_pairwise = 3
                            lr=0.00005
                            min_cluster_size = 1
                            temperature = 10

                        elif dataset == "iapr":
                            n_class = 255
                            tag_len = 2912

                            lambda_contrast = 0.3
                            lambda_class = 0.5
                            lambda_pairwise = 1

                            epoch=80

                            min_cluster_size = 1
                            temperature=0.7
                                                        
                        config = get_config()
                        print(config)
                        train(config, bit, rand_num,a)
                        # test(config, bit)