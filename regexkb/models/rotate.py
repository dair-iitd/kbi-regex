import torch
from torch.nn import Embedding, Linear
import torch.nn.functional as F

import losses

from .pl_model import Model
from regexkb import constants


class RotatE(Model):

    def __init__(self, args):

        super().__init__()
        self.save_hyperparameters(args)
        self.kleene_plus_op = args.kleene_plus_op
        self.disjunction_op = args.disjunction_op

        self.E = Embedding(args.num_entity, args.embedding_dim)
        self.R = Embedding(args.num_relation, args.embedding_dim // 2)

        # Regex query training
        if args.query_types is not [0]:
            if self.kleene_plus_op in [constants.GEOMETRIC, constants.FREE_PARAM]:
                self.kleene_plus_R = Embedding(
                    args.num_relation, args.embedding_dim // 2)
            elif self.kleene_plus_op == constants.GQE:
                self.kleene_plus_R = Linear(args.embedding_dim // 2,
                                            args.embedding_dim // 2, bias=False)
                torch.nn.init.xavier_uniform_(self.kleene_plus_R.weight)
            else:
                assert False
            
            if self.disjunction_op == constants.GQE:
                self.disjunction_R_premat = Linear(args.embedding_dim // 2,
                                                   args.embedding_dim // 2, bias=False)
                torch.nn.init.xavier_uniform_(self.disjunction_R_premat.weight)

                self.disjunction_R_postmat = Linear(args.embedding_dim // 2,
                                                    args.embedding_dim // 2, bias=False)
                torch.nn.init.xavier_uniform_(
                    self.disjunction_R_postmat.weight)

                self.agg_func = lambda x: torch.min(x, dim=0)[0]

        if args.box:
            self.R_offset = Embedding(args.num_relation, args.embedding_dim)
            # Regex query training
            if args.query_types is not [0]:
                if self.kleene_plus_op in [constants.GEOMETRIC, constants.FREE_PARAM]:
                    self.kleene_plus_R_offset = Embedding(
                        args.num_relation, args.embedding_dim)
                elif self.kleene_plus_op == constants.GQE:
                    self.kleene_plus_R_offset = Linear(args.embedding_dim,
                                                       args.embedding_dim, bias=False)
                    torch.nn.init.xavier_uniform_(
                        self.kleene_plus_R_offset.weight)
                else:
                    assert False

                if self.disjunction_op == constants.GQE:
                    self.disjunction_R_offset_premat = Linear(args.embedding_dim,
                                                              args.embedding_dim, bias=False)
                    torch.nn.init.xavier_uniform_(
                        self.disjunction_R_offset_premat.weight)

                    self.disjunction_R_offset_postmat = Linear(args.embedding_dim,
                                                               args.embedding_dim, bias=False)
                    torch.nn.init.xavier_uniform_(
                        self.disjunction_R_offset_postmat.weight)

        embedding_range = 2 * (args.margin + args.epsilon) / args.embedding_dim
        init_scale = 1e-3

        torch.nn.init.uniform_(
            self.E.weight, a=-embedding_range, b=embedding_range)
        torch.nn.init.uniform_(self.R.weight, a=-1.0, b=1.0)

        if args.query_types is not [0]:
            if self.kleene_plus_op == constants.GEOMETRIC:
                torch.nn.init.normal_(
                    self.kleene_plus_R.weight, 0.0, init_scale)
            elif self.kleene_plus_op == constants.FREE_PARAM:
                torch.nn.init.uniform_(
                    self.kleene_plus_R.weight, a=-1.0, b=1.0)
            elif self.kleene_plus_op == constants.GQE:
                # already initialized
                pass
            else:
                assert False, f'Please specify initialization for {self.kleene_plus_op}'

        if args.box:
            torch.nn.init.normal_(self.R_offset.weight, 0.0, init_scale)
            if args.query_types is not [0] and self.kleene_plus_op is not constants.GQE:
                torch.nn.init.normal_(
                    self.kleene_plus_R_offset.weight, 0.0, init_scale)

        self.loss = getattr(losses, args.loss)(args)
        self.epsilon = 1e-20
        self.minimum_value = -float("Inf")

    def load_from_checkpoint(self, path):
        data = torch.load(path)["state_dict"]
        # temporary
        if 'kleene_plus_R.weight' in data:
            del data['kleene_plus_R.weight']
        if 'kleene_plus_R_offset.weight' in data:
            del data['kleene_plus_R_offset.weight']
        # Load pre-trained KBC weights
        self.load_state_dict(data, strict=False)
        # If starting regex training from pre-trained KBC weights
        if self.hparams.kleene_plus_op == constants.FREE_PARAM:
            self.kleene_plus_R.weight.data = data['R.weight']
            if self.hparams.box:
                self.kleene_plus_R_offset.weight.data = data['R_offset.weight']

    def forward(self, s, r, o, rel_path_ids, query_type, mode):

        def kleene_op(r):
            if self.kleene_plus_op == constants.GEOMETRIC:
                return (1 + torch.abs(self.kleene_plus_R(r))) * self.R(r)
            elif self.kleene_plus_op == constants.FREE_PARAM:
                return self.kleene_plus_R(r)
            elif self.kleene_plus_op == constants.GQE:
                if query_type == 20:
                    return self.kleene_plus_R(r)
                else:
                    return self.kleene_plus_R(self.R(r))
            else:
                assert False, f'Please specify kleene_op for {self.kleene_plus_op}'

        def kleene_op_offset(r):
            if self.kleene_plus_op == constants.GEOMETRIC:
                return (1 + torch.abs(self.kleene_plus_R_offset(r))) * self.R_offset(r)
            elif self.kleene_plus_op == constants.FREE_PARAM:
                return self.kleene_plus_R_offset(r)
            elif self.kleene_plus_op == constants.GQE:
                if query_type == 20:
                    return self.kleene_plus_R_offset(r)
                else:
                    return self.kleene_plus_R_offset(self.R_offset(r))
            else:
                assert False, f'Please specify kleene_op_offset for {self.kleene_plus_op}'

        def gqe_disjunction_center(rels):
            temp1 = F.relu(self.disjunction_R_premat(rels))
            combined = self.agg_func(temp1)
            return self.disjunction_R_postmat(combined)

        def gqe_disjunction_offset(rels):
            temp1 = F.relu(self.disjunction_R_offset_premat(rels))
            combined = self.agg_func(temp1)
            return self.disjunction_R_offset_postmat(combined)

        pi = 3.14159265358979323846

        head = (self.E(s)).unsqueeze(0)
        tail = (self.E(o) if o is not None else self.E.weight).unsqueeze(0)

        head_re, head_im = torch.chunk(head, 2, dim=-1)
        tail_re, tail_im = torch.chunk(tail, 2, dim=-1)

        # (e1, r, e2)
        if query_type == 0:
            assert r.shape[1] == 1
            r = r.squeeze(1)
            rel = (self.R(r) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(self.R_offset(r)).unsqueeze(0)

        # (e1, r+, e2)
        elif query_type == 1:
            assert r.shape[1] == 1
            r = r.squeeze(1)
            rel = (kleene_op(r) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(r)).unsqueeze(0)

        # (e1, r1+, r2+, e2)
        elif query_type == 2:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = ((kleene_op(r1) + kleene_op(r2)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(
                    r1)) + torch.abs(kleene_op_offset(r2)).unsqueeze(0)

        # (e1, r1+, r2+, r3+, e2)
        elif query_type == 3:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((kleene_op(r1) + kleene_op(r2) +
                    kleene_op(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(
                    r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)

        # (e1, r1, r2, e2)
        elif query_type == 21:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = ((self.R(r1) + self.R(r2)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(self.R_offset(
                    r1)) + torch.abs(self.R_offset(r2)).unsqueeze(0)

        # (e1, r1, r2+, e2)
        elif query_type == 4:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = ((self.R(r1) + kleene_op(r2)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(self.R_offset(
                    r1)) + torch.abs(kleene_op_offset(r2)).unsqueeze(0)

        # (e1, r1+, r2, e2)
        elif query_type == 5:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            rel = ((kleene_op(r1) + self.R(r2)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(
                    r1)) + torch.abs(self.R_offset(r2)).unsqueeze(0)

        # (e1, r1+, r2+, r3, e2)
        elif query_type == 6:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((kleene_op(r1) + kleene_op(r2) + self.R(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(
                    r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1+, r2, r3+, e2)
        elif query_type == 7:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((kleene_op(r1) + self.R(r2) + kleene_op(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(
                    r1)) + torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)

        # (e1, r1, r2+, r3+, e2)
        elif query_type == 8:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((self.R(r1) + kleene_op(r2) + kleene_op(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(self.R_offset(
                    r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)

        # (e1, r1, r2, r3+, e2)
        elif query_type == 9:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((self.R(r1) + self.R(r2) + kleene_op(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(self.R_offset(
                    r1)) + torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset(r3)).unsqueeze(0)

        # (e1, r1, r2+, r3, e2)
        elif query_type == 10:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((self.R(r1) + kleene_op(r2) + self.R(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(self.R_offset(
                    r1)) + torch.abs(kleene_op_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1+, r2, r3, e2)
        elif query_type == 11:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            rel = ((kleene_op(r1) + self.R(r2) + self.R(r3)) * pi).unsqueeze(0)

            if self.hparams.box:
                query_offset = torch.abs(kleene_op_offset(
                    r1)) + torch.abs(self.R_offset(r2)) + torch.abs(self.R_offset(r3)).unsqueeze(0)

        # (e1, r1/r2, e2)
        elif query_type == 12:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            if self.disjunction_op == constants.GQE:
                rel = (gqe_disjunction_center(torch.stack([self.R(r1),
                                                           self.R(r2)], dim=0)) * pi).unsqueeze(0)
            else:
                # Disjunction as Aggregation
                rel = torch.stack([self.R(r1) * pi,
                                   self.R(r2) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                                 self.R_offset(r2)], dim=0))).unsqueeze(0)
                else:
                    query_offset = torch.stack([torch.abs(self.R_offset(r1)),
                                                torch.abs(self.R_offset(r2))], dim=0)

        # (e1, r1/r2, r3, e2)
        elif query_type == 13:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0))
                rel = ((rel + self.R(r3)) * pi).unsqueeze(0)
            else:
                rel = torch.stack([(self.R(r1) + self.R(r3)) * pi,
                                   (self.R(r2) + self.R(r3)) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                                 self.R_offset(r2)], dim=0)))
                    query_offset = query_offset + \
                        torch.abs(self.R_offset(r3)).unsqueeze(0)
                else:
                    query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(self.R_offset(r3)),
                                                torch.abs(self.R_offset(r2)) + torch.abs(self.R_offset(r3))], dim=0)

        # (e1, r1, r2/r3, e2)
        elif query_type == 14:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r2),
                                                          self.R(r3)], dim=0))
                rel = ((self.R(r1) + rel) * pi).unsqueeze(0)
            else:
                rel = torch.stack([(self.R(r1) + self.R(r2)) * pi,
                                   (self.R(r1) + self.R(r3)) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r2),
                                                                                 self.R_offset(r3)], dim=0)))
                    query_offset = torch.abs(self.R_offset(r1)).unsqueeze(0) + \
                        query_offset
                else:
                    query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(self.R_offset(r2)),
                                                torch.abs(self.R_offset(r1)) + torch.abs(self.R_offset(r3))], dim=0)

        # (e1, r1+/r2+, e2)
        elif query_type == 15:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            if self.disjunction_op == constants.GQE:
                rel = (gqe_disjunction_center(torch.stack([kleene_op(r1),
                                                           kleene_op(r2)], dim=0)) * pi).unsqueeze(0)
            else:
                rel = torch.stack([kleene_op(r1) * pi,
                                   kleene_op(r2) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([kleene_op_offset(r1),
                                                                                 kleene_op_offset(r2)], dim=0))).unsqueeze(0)
                else:
                    query_offset = torch.stack([torch.abs(kleene_op_offset(r1)),
                                                torch.abs(kleene_op_offset(r2))], dim=0)

        # (e1, r1/r2, r3+, e2)
        elif query_type == 16:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0))
                rel = ((rel + kleene_op(r3)) * pi).unsqueeze(0)
            else:
                rel = torch.stack([(self.R(r1) + kleene_op(r3)) * pi,
                                   (self.R(r2) + kleene_op(r3)) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                                 self.R_offset(r2)], dim=0)))
                    query_offset = query_offset + \
                        torch.abs(kleene_op_offset(r3)).unsqueeze(0)
                else:
                    query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(kleene_op_offset(r3)),
                                                torch.abs(self.R_offset(r2)) + torch.abs(kleene_op_offset(r3))], dim=0)

        # (e1, r1+/r2+, r3, e2)
        elif query_type == 17:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([kleene_op(r1),
                                                          kleene_op(r2)], dim=0))
                rel = ((rel + self.R(r3)) * pi).unsqueeze(0)
            else:
                rel = torch.stack([(kleene_op(r1) + self.R(r3)) * pi,
                                   (kleene_op(r2) + self.R(r3)) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([kleene_op_offset(r1),
                                                                                 kleene_op_offset(r2)], dim=0)))
                    query_offset = query_offset + \
                        torch.abs(self.R_offset(r3)).unsqueeze(0)
                else:
                    query_offset = torch.stack([torch.abs(kleene_op_offset(r1)) + torch.abs(self.R_offset(r3)),
                                                torch.abs(kleene_op_offset(r2)) + torch.abs(self.R_offset(r3))], dim=0)

        # (e1, r1+, r2/r3, e2)
        elif query_type == 18:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r2),
                                                          self.R(r3)], dim=0))
                rel = ((kleene_op(r1) + rel) * pi).unsqueeze(0)
            else:
                rel = torch.stack([(kleene_op(r1) + self.R(r2)) * pi,
                                   (kleene_op(r1) + self.R(r3)) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([self.R_offset(r2),
                                                                                 self.R_offset(r3)], dim=0)))
                    query_offset = torch.abs(kleene_op_offset(r1)).unsqueeze(0) + \
                        query_offset
                else:
                    query_offset = torch.stack([torch.abs(kleene_op_offset(r1)) + torch.abs(self.R_offset(r2)),
                                                torch.abs(kleene_op_offset(r1)) + torch.abs(self.R_offset(r3))], dim=0)

        # (e1, r1, r2+/r3+, e2)
        elif query_type == 19:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([kleene_op(r2),
                                                          kleene_op(r3)], dim=0))
                rel = ((self.R(r1) + rel) * pi).unsqueeze(0)
            else:
                rel = torch.stack([(self.R(r1) + kleene_op(r2)) * pi,
                                   (self.R(r1) + kleene_op(r3)) * pi], dim=0)

            if self.hparams.box:
                if self.disjunction_op == constants.GQE:
                    query_offset = torch.abs(gqe_disjunction_offset(torch.stack([kleene_op_offset(r2),
                                                                                 kleene_op_offset(r3)], dim=0)))
                    query_offset = torch.abs(self.R_offset(r1)).unsqueeze(0) + \
                        query_offset
                else:
                    query_offset = torch.stack([torch.abs(self.R_offset(r1)) + torch.abs(kleene_op_offset(r2)),
                                                torch.abs(self.R_offset(r1)) + torch.abs(kleene_op_offset(r3))], dim=0)

        # (e1, (r1/r2)+, e2)
        elif query_type == 20:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            if self.disjunction_op == constants.GQE and self.kleene_plus_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0))
                rel = (kleene_op(rel) * pi).unsqueeze(0)
            else:
                assert False

            if self.hparams.box:
                if self.disjunction_op == constants.GQE and self.kleene_plus_op == constants.GQE:
                    query_offset = gqe_disjunction_offset(torch.stack([self.R_offset(r1),
                                                                       self.R_offset(r2)], dim=0))
                    query_offset = torch.abs(
                        kleene_op_offset(query_offset)).unsqueeze(0)

        else:
            assert False

        rel_re = torch.cos(rel)
        rel_im = torch.sin(rel)

        query_re = head_re * rel_re - head_im * rel_im
        query_im = head_re * rel_im + head_im * rel_re

        if self.hparams.box:
            if self.hparams.debug:
                query_offset_re, query_offset_im = torch.chunk(
                    query_offset, 2, dim=-1)

                query_max_re = query_re + query_offset_re
                query_max_im = query_im + query_offset_im

                query_min_re = query_re - query_offset_re
                query_min_im = query_im - query_offset_im

                query_length = torch.abs(query_max_re - query_min_re) + \
                    torch.abs(query_max_im - query_min_im)

                if not (head.shape == tail.shape):
                    query_max_re = query_max_re.unsqueeze(2)
                    query_max_im = query_max_im.unsqueeze(2)
                    query_min_re = query_min_re.unsqueeze(2)
                    query_min_im = query_min_im.unsqueeze(2)
                    query_re = query_re.unsqueeze(2)
                    query_im = query_im.unsqueeze(2)
                    query_offset_re = query_offset_re.unsqueeze(2)
                    query_offset_im = query_offset_im.unsqueeze(2)
                    query_length = query_length.unsqueeze(2)

                zeros = torch.zeros_like(query_min_re)
                ones = torch.ones_like(query_min_re)

                t = ((tail_re - query_min_re) * (query_max_re - query_min_re)
                     + (tail_im - query_min_im) * (query_max_im - query_min_im)) / (query_length + self.epsilon)

                t = torch.max(zeros, torch.min(ones, t))

                projection_re = query_min_re + t * \
                    (query_max_re - query_min_re)
                projection_im = query_min_im + t * \
                    (query_max_im - query_min_im)

                dist_out = torch.abs(projection_re - tail_re) + \
                    torch.abs(projection_im - tail_im)
                dist_out = torch.norm(dist_out, p=1, dim=-1)

                dist_in = torch.abs(projection_re - query_re) + \
                    torch.abs(projection_im - query_im)
                dist_in = torch.norm(dist_in, p=1, dim=-1)

                return -1 * torch.min(dist_out + self.hparams.alpha * dist_in, dim=0)[0]

            elif self.hparams.debug2:
                query_offset_re, query_offset_im = torch.chunk(
                    query_offset, 2, dim=-1)

                query_max_re = query_re + query_offset_re
                query_max_im = query_im + query_offset_im

                query_min_re = query_re - query_offset_re
                query_min_im = query_im - query_offset_im

                if not (head.shape == tail.shape):
                    query_max_re = query_max_re.unsqueeze(2)
                    query_max_im = query_max_im.unsqueeze(2)
                    query_min_re = query_min_re.unsqueeze(2)
                    query_min_im = query_min_im.unsqueeze(2)
                    query_re = query_re.unsqueeze(2)
                    query_im = query_im.unsqueeze(2)
                    query_offset_re = query_offset_re.unsqueeze(2)
                    query_offset_im = query_offset_im.unsqueeze(2)

                dist_out = torch.min(
                    torch.abs(query_max_re - tail_re) +
                    torch.abs(query_max_im - tail_im),
                    torch.abs(query_min_re - tail_re) +
                    torch.abs(query_min_im - tail_im))

                dist_out = torch.norm(dist_out, p=1, dim=-1)

                dist_in = torch.abs(query_offset_re) + \
                    torch.abs(query_offset_im)

                dist_in = torch.norm(dist_in, p=1, dim=-1)
                return -1 * torch.min(dist_out + self.hparams.alpha * dist_in, dim=0)[0]

            else:
                query_center = torch.cat((query_re, query_im), dim=-1)
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

                return -1 * torch.min(dist_out + self.hparams.alpha * dist_in, dim=0)[0]

        if not (head.shape == tail.shape):
            query_re = query_re.unsqueeze(2)
            query_im = query_im.unsqueeze(2)

        re_score = query_re - tail_re
        im_score = query_im - tail_im

        score = torch.stack([re_score, im_score], dim=0)

        score = score.norm(dim=0)

        return -1 * torch.min(score.sum(dim=-1), dim=0)[0]
