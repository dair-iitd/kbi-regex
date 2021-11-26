import torch
import torch.nn.functional as F


class query2box_loss(torch.nn.Module):
    def __init__(self, args):
        super(query2box_loss, self).__init__()
        self.margin = args.margin
        self.neg_adv_sampling = args.neg_adv_sampling
        self.adv_temperature = args.adv_temperature
        self.sub_sample = args.sub_sample
        if args.truncate_loss:
            self.truncate_loss = True
        else:
            self.truncate_loss = args.model == 'Query2Box' and args.dataset == 'wiki_v2' and len(
                args.query_types) > 0
        if self.truncate_loss:
            print("trancating loss enabled\n")

    def forward(self, positive, negative, sub_sampling_weight=None):
        if self.truncate_loss:
            # if torch.min(positive).item() < -100.0:
            #     print('truncating loss\n')
            positive = torch.max(positive, torch.ones_like(positive)*-100.0)

        positive_score = torch.log(torch.sigmoid(self.margin + positive))
        negative_score = torch.log(
            torch.sigmoid(-1 * (self.margin + negative)))

        if self.neg_adv_sampling:
            negative_weights = F.softmax(
                (self.margin + negative) * self.adv_temperature,
                dim=-1).detach()
            negative_score = (negative_weights * negative_score).sum(dim=-1)
        else:
            negative_score = negative_score.mean(dim=-1)

        if self.sub_sample:
            sub_sampling_weight = sub_sampling_weight.float() / \
                sub_sampling_weight.sum()
            losses = -sub_sampling_weight * (positive_score + negative_score)
            return losses.sum()
        else:
            losses = -1 * (positive_score + negative_score)
            return losses.mean()

class betae_loss(torch.nn.Module):
    def __init__(self, args):
        super(betae_loss, self).__init__()
        self.margin = args.margin
        self.neg_adv_sampling = args.neg_adv_sampling
        self.adv_temperature = args.adv_temperature
        self.sub_sample = args.sub_sample
        if args.truncate_loss:
            self.truncate_loss = True
        else:
            self.truncate_loss = args.model == 'Query2Box' and args.dataset == 'wiki_v2' and len(
                args.query_types) > 0
        if self.truncate_loss:
            print("trancating loss enabled\n")

    def forward(self, positive, negative, sub_sampling_weight=None):
        if self.truncate_loss:
            # if torch.min(positive).item() < -100.0:
            #     print('truncating loss\n')
            positive = torch.max(positive, torch.ones_like(positive)*-100.0)

        positive_score = torch.log(torch.sigmoid(self.margin + positive))
        negative_score = torch.log(
            torch.sigmoid(-1 * (self.margin + negative)))

        if self.neg_adv_sampling:
            negative_weights = F.softmax(
                (self.margin + negative) * self.adv_temperature,
                dim=-1).detach()
            negative_score = (negative_weights * negative_score).sum(dim=-1)
        else:
            negative_score = negative_score.mean(dim=-1)

        if self.sub_sample:
            sub_sampling_weight = sub_sampling_weight.float() / \
                sub_sampling_weight.sum()
            losses = -sub_sampling_weight * (positive_score + negative_score)
            return losses.sum() / 2
        else:
            losses = -1 * (positive_score + negative_score)
            return losses.mean() / 2
