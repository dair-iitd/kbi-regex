import torch
from torch.nn import Embedding, Linear
import torch.nn.functional as F

import losses

from .pl_model import Model
from regexkb import constants


class Query2Box_hybrid(Model):

    def __init__(self, args):

        super().__init__()
        self.save_hyperparameters(args)

        self.E = Embedding(args.num_entity, args.embedding_dim)
        self.R = Embedding(args.num_relation, args.embedding_dim)
        self.R_offset = Embedding(args.num_relation, args.embedding_dim)

        if args.query_types is not [0]:
            self.kleene_plus_R = Embedding(
                args.num_relation, args.embedding_dim)
            self.kleene_plus_R_offset = Embedding(
                args.num_relation, args.embedding_dim)

            self.kleene_plus_R_hybrid = Linear(args.embedding_dim,
                                               args.embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.kleene_plus_R_hybrid.weight)

            self.kleene_plus_R_offset_hybrid = Linear(args.embedding_dim,
                                                      args.embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(
                self.kleene_plus_R_offset_hybrid.weight)

            self.disjunction_R_premat = Linear(args.embedding_dim,
                                               args.embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.disjunction_R_premat.weight)

            self.disjunction_R_postmat = Linear(args.embedding_dim,
                                                args.embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(
                self.disjunction_R_postmat.weight)

            self.disjunction_R_offset_premat = Linear(args.embedding_dim,
                                                      args.embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(
                self.disjunction_R_offset_premat.weight)

            self.disjunction_R_offset_postmat = Linear(args.embedding_dim,
                                                       args.embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(
                self.disjunction_R_offset_postmat.weight)

            self.agg_func = lambda x: torch.min(x, dim=0)[0]

        init_scale = 1e-3
        torch.nn.init.normal_(self.E.weight, 0, init_scale)
        torch.nn.init.normal_(self.R.weight, 0, init_scale)
        torch.nn.init.uniform_(self.R_offset.weight, 0, init_scale * 2)

        if args.query_types is not [0]:
            torch.nn.init.normal_(self.kleene_plus_R.weight, 0.0, init_scale)
            torch.nn.init.normal_(
                self.kleene_plus_R_offset.weight, 0.0, init_scale)

        self.loss = getattr(losses, args.loss)(args)
        self.minimum_value = -float("Inf")

    def load_from_checkpoint(self, path):
        data = torch.load(path)["state_dict"]
        # temporary
        if 'kleene_plus_R.weight' in data:
            del data['kleene_plus_R.weight']
        if 'kleene_plus_R_offset.weight' in data:
            del data['kleene_plus_R_offset.weight']
        self.load_state_dict(data, strict=False)
        # If starting regex training from pre-trained KBC weights
        self.kleene_plus_R.weight.data = data['R.weight']
        self.kleene_plus_R_offset.weight.data = data['R_offset.weight']

    def forward(self, s, r, o, rel_path_ids, query_type, mode):

        def kleene_op(r):
            return self.kleene_plus_R(r)

        def kleene_op_hybrid(r):
            if query_type == 20:
                return self.kleene_plus_R_hybrid(r)
            else:
                return self.kleene_plus_R_hybrid(self.R(r))

        def kleene_op_offset(r):
            return self.kleene_plus_R_offset(r)

        def kleene_op_offset_hybrid(r):
            if query_type == 20:
                return self.kleene_plus_R_offset_hybrid(r)
            else:
                return self.kleene_plus_R_offset_hybrid(self.R_offset(r))

        def gqe_disjunction_center(rels):
            temp1 = F.relu(self.disjunction_R_premat(rels))
            combined = self.agg_func(temp1)
            return self.disjunction_R_postmat(combined)

        def gqe_disjunction_offset(rels):
            temp1 = F.relu(self.disjunction_R_offset_premat(rels))
            combined = self.agg_func(temp1)
            return self.disjunction_R_offset_postmat(combined)

        head = (self.E(s)).unsqueeze(0)
        tail = (self.E(o) if o is not None else self.E.weight).unsqueeze(0)

        rel2 = None
        query_offset2 = None

        # (e1, r, e2)
        if query_type == 0:
            assert r.shape[1] == 1
            r = r.squeeze(1)
            rel = self.R(r).unsqueeze(0)
            query_offset = torch.abs(self.R_offset(r)).unsqueeze(0)

        # (e1, r+, e2)
        elif query_type == 1:
            assert r.shape[1] == 1
            r = r.squeeze(1)
            rel = kleene_op(r).unsqueeze(0)
            if mode is 'train':
                rel2 = kleene_op_hybrid(r).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(r)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(
                    kleene_op_offset_hybrid(r)).unsqueeze(0)

        # (e1, r1+, r2+, e2)
        elif query_type == 2:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = (kleene_op(r1) + kleene_op(r2)).unsqueeze(0)
            if mode is 'train':
                rel2 = (kleene_op_hybrid(r1) +
                        kleene_op_hybrid(r2)).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(
                r1)) + torch.abs(kleene_op_offset(r2)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(kleene_op_offset_hybrid(
                    r1)) + torch.abs(kleene_op_offset_hybrid(r2)).unsqueeze(0)

        # (e1, r1+, r2+, r3+, e2)
        elif query_type == 3:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (kleene_op(r1) + kleene_op(r2) + kleene_op(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (kleene_op_hybrid(r1) + kleene_op_hybrid(r2) +
                        kleene_op_hybrid(r3)).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(
                r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(kleene_op_offset_hybrid(
                    r1)) + torch.abs(kleene_op_offset_hybrid(r2)) + torch.abs(kleene_op_offset_hybrid(r3)).unsqueeze(0)

        # (e1, r1, r2, e2)
        elif query_type == 21:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = (self.R(r1) + self.R(r2)).unsqueeze(0)
            query_offset = torch.abs(self.R_offset(
                r1)) + torch.abs(self.R_offset(r2)).unsqueeze(0)

        # (e1, r1, r2+, e2)
        elif query_type == 4:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = (self.R(r1) + kleene_op(r2)).unsqueeze(0)
            if mode is 'train':
                rel2 = (self.R(r1) + kleene_op_hybrid(r2)).unsqueeze(0)

            query_offset = torch.abs(self.R_offset(
                r1)) + torch.abs(kleene_op_offset(r2)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(self.R_offset(
                    r1)) + torch.abs(kleene_op_offset_hybrid(r2)).unsqueeze(0)

        # (e1, r1+, r2, e2)
        elif query_type == 5:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = (kleene_op(r1) + self.R(r2)).unsqueeze(0)
            if mode is 'train':
                rel2 = (kleene_op_hybrid(r1) + self.R(r2)).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(
                r1)) + torch.abs(self.R_offset(r2)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(kleene_op_offset_hybrid(
                    r1)) + torch.abs(self.R_offset(r2)).unsqueeze(0)

        # (e1, r1+, r2+, r3, e2)
        elif query_type == 6:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (kleene_op(r1) + kleene_op(r2) + self.R(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (kleene_op_hybrid(r1) + kleene_op_hybrid(r2) +
                        self.R(r3)).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(
                r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(kleene_op_offset_hybrid(
                    r1)) + torch.abs(kleene_op_offset_hybrid(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1+, r2, r3+, e2)
        elif query_type == 7:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (kleene_op(r1) + self.R(r2) + kleene_op(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (kleene_op_hybrid(r1) + self.R(r2) +
                        kleene_op_hybrid(r3)).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(
                r1)) + torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(kleene_op_offset_hybrid(
                    r1)) + torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset_hybrid(r3)).unsqueeze(0)

        # (e1, r1, r2+, r3+, e2)
        elif query_type == 8:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (self.R(r1) + kleene_op(r2) + kleene_op(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (self.R(r1) + kleene_op_hybrid(r2) +
                        kleene_op_hybrid(r3)).unsqueeze(0)

            query_offset = torch.abs(self.R_offset(
                r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(self.R_offset(
                    r1)) + torch.abs(kleene_op_offset_hybrid(r2)) + torch.abs(kleene_op_offset_hybrid(r3)).unsqueeze(0)

        # (e1, r1, r2, r3+, e2)
        elif query_type == 9:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (self.R(r1) + self.R(r2) + kleene_op(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (self.R(r1) + self.R(r2) +
                        kleene_op_hybrid(r3)).unsqueeze(0)

            query_offset = torch.abs(self.R_offset(
                r1)) + torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(self.R_offset(
                    r1)) + torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset_hybrid(r3)).unsqueeze(0)

        # (e1, r1, r2+, r3, e2)
        elif query_type == 10:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (self.R(r1) + kleene_op(r2) + self.R(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (self.R(r1) + kleene_op_hybrid(r2) +
                        self.R(r3)).unsqueeze(0)

            query_offset = torch.abs(self.R_offset(
                r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(self.R_offset(
                    r1)) + torch.abs(kleene_op_offset_hybrid(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1+, r2, r3, e2)
        elif query_type == 11:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = (kleene_op(r1) + self.R(r2) + self.R(r3)).unsqueeze(0)
            if mode is 'train':
                rel2 = (kleene_op_hybrid(r1) + self.R(r2) +
                        self.R(r3)).unsqueeze(0)

            query_offset = torch.abs(kleene_op_offset(
                r1)) + torch.abs(self.R_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)
            if mode is 'train':
                query_offset2 = torch.abs(kleene_op_offset_hybrid(
                    r1)) + torch.abs(self.R_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1/r2, e2)
        elif query_type == 12:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = torch.stack([self.R(r1),
                               self.R(r2)], dim=0)
            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([self.R(r1),
                                                           self.R(r2)], dim=0)).unsqueeze(0)

            query_offset = torch.stack([torch.abs(self.R_offset(r1)),
                                        torch.abs(self.R_offset(r2))], dim=0)
            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                              self.R_offset(r2)], dim=0))).unsqueeze(0)

        # (e1, r1/r2, r3, e2)
        elif query_type == 13:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = torch.stack([(self.R(r1) + self.R(r3)),
                               (self.R(r2) + self.R(r3))], dim=0)

            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([self.R(r1),
                                                           self.R(r2)], dim=0))
                rel2 = rel2 + self.R(r3).unsqueeze(0)

            query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(self.R_offset(r3)),
                                        torch.abs(self.R_offset(r2)) + torch.abs(self.R_offset(r3))], dim=0)

            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                              self.R_offset(r2)], dim=0)))
                query_offset2 = query_offset2 + \
                    torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1, r2/r3, e2)
        elif query_type == 14:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = torch.stack([self.R(r1) + self.R(r2),
                               self.R(r1) + self.R(r3)], dim=0)

            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([self.R(r2),
                                                           self.R(r3)], dim=0))
                rel2 = self.R(r1).unsqueeze(0) + rel2

            query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(self.R_offset(r2)),
                                        torch.abs(self.R_offset(r1)) + torch.abs(self.R_offset(r3))], dim=0)

            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r2),
                                                                              self.R_offset(r3)], dim=0)))
                query_offset2 = torch.abs(self.R_offset(r1)).unsqueeze(0) + \
                    query_offset2

        # (e1, r1+/r2+, e2)
        elif query_type == 15:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = torch.stack([kleene_op(r1),
                               kleene_op(r2)], dim=0)

            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([kleene_op_hybrid(r1),
                                                           kleene_op_hybrid(r2)], dim=0)).unsqueeze(0)

            query_offset = torch.stack([torch.abs(kleene_op_offset(r1)),
                                        torch.abs(kleene_op_offset(r2))], dim=0)
            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([kleene_op_offset_hybrid(r1),
                                                                              kleene_op_offset_hybrid(r2)], dim=0))).unsqueeze(0)

        # (e1, r1/r2, r3+, e2)
        elif query_type == 16:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = torch.stack([self.R(r1) + kleene_op(r3),
                               self.R(r2) + kleene_op(r3)], dim=0)
            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([self.R(r1),
                                                           self.R(r2)], dim=0))
                rel2 = rel2 + kleene_op_hybrid(r3).unsqueeze(0)

            query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(kleene_op_offset(r3)),
                                        torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset(r3))], dim=0)
            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                              self.R_offset(r2)], dim=0)))
                query_offset2 = query_offset2 + \
                    torch.abs(kleene_op_offset_hybrid(r3)).unsqueeze(0)

        # (e1, r1+/r2+, r3, e2)
        elif query_type == 17:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = torch.stack([kleene_op(r1) + self.R(r3),
                               kleene_op(r2) + self.R(r3)], dim=0)
            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([kleene_op_hybrid(r1),
                                                           kleene_op_hybrid(r2)], dim=0))
                rel2 = rel2 + self.R(r3).unsqueeze(0)

            query_offset = torch.stack([torch.abs(kleene_op_offset(r1)) + torch.abs(self.R_offset(r3)),
                                        torch.abs(kleene_op_offset(r2)) + torch.abs(self.R_offset(r3))], dim=0)
            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([kleene_op_offset_hybrid(r1),
                                                                              kleene_op_offset_hybrid(r2)], dim=0)))
                query_offset2 = query_offset2 + \
                    torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1+, r2/r3, e2)
        elif query_type == 18:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = torch.stack([kleene_op(r1) + self.R(r2),
                               kleene_op(r1) + self.R(r3)], dim=0)

            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([self.R(r2),
                                                           self.R(r3)], dim=0))
                rel2 = kleene_op_hybrid(r1).unsqueeze(0) + rel2

            query_offset = torch.stack([torch.abs(kleene_op_offset(r1)) + torch.abs(self.R_offset(r2)),
                                        torch.abs(kleene_op_offset(r1)) + torch.abs(self.R_offset(r3))], dim=0)
            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r2),
                                                                              self.R_offset(r3)], dim=0)))
                query_offset2 = torch.abs(kleene_op_offset_hybrid(r1)).unsqueeze(0) + \
                    query_offset2

        # (e1, r1, r2+/r3+, e2)
        elif query_type == 19:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = torch.stack([self.R(r1) + kleene_op(r2),
                               self.R(r1) + kleene_op(r3)], dim=0)
            if mode is 'train':
                rel2 = gqe_disjunction_center(torch.stack([kleene_op_hybrid(r2),
                                                           kleene_op_hybrid(r3)], dim=0))
                rel2 = self.R(r1).unsqueeze(0) + rel2

            query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(kleene_op_offset(r2)),
                                        torch.abs(self.R_offset(r1)) + torch.abs(kleene_op_offset(r3))], dim=0)
            if mode is 'train':
                query_offset2 = torch.abs(gqe_disjunction_offset(torch.stack([kleene_op_offset_hybrid(r2),
                                                                              kleene_op_offset_hybrid(r3)], dim=0)))
                query_offset2 = torch.abs(self.R_offset(r1)).unsqueeze(0) + \
                    query_offset2

        # (e1, (r1/r2)+, e2)
        elif query_type == 20:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                      self.R(r2)], dim=0))
            rel = kleene_op_hybrid(rel).unsqueeze(0)

            query_offset = gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                               self.R_offset(r2)], dim=0))
            query_offset = torch.abs(
                kleene_op_offset_hybrid(query_offset)).unsqueeze(0)

        else:
            assert False

        query_center = head + rel
        query_max = query_center + query_offset
        query_min = query_center - query_offset

        if not (head.shape == tail.shape):
            query_max = query_max.unsqueeze(2)
            query_min = query_min.unsqueeze(2)
            query_center = query_center.unsqueeze(2)

        zeros = torch.zeros_like(query_min)
        dist_out = torch.max(tail - query_max, zeros) + \
            torch.max(query_min - tail, zeros)
        dist_out = torch.norm(dist_out, p=1, dim=-1)

        dist_in = query_center - \
            torch.min(query_max, torch.max(query_min, tail))
        dist_in = torch.norm(dist_in, p=1, dim=-1)

        score = -1 * torch.min(dist_out +
                               self.hparams.alpha * dist_in, dim=0)[0]

        if rel2 is None and query_offset2 is None:
            return score

        else:
            assert rel2 is not None
            assert query_offset2 is not None

            query_center2 = head + rel2
            query_max2 = query_center2 + query_offset2
            query_min2 = query_center2 - query_offset2

            if not (head.shape == tail.shape):
                query_max2 = query_max2.unsqueeze(2)
                query_min2 = query_min2.unsqueeze(2)
                query_center2 = query_center2.unsqueeze(2)

            zeros = torch.zeros_like(query_min2)
            dist_out2 = torch.max(tail - query_max2, zeros) + \
                torch.max(query_min2 - tail, zeros)
            dist_out2 = torch.norm(dist_out2, p=1, dim=-1)

            dist_in2 = query_center2 - \
                torch.min(query_max2, torch.max(query_min2, tail))
            dist_in2 = torch.norm(dist_in2, p=1, dim=-1)

            score2 = -1 * torch.min(dist_out2 +
                                    self.hparams.alpha * dist_in2, dim=0)[0]
            return score + score2
