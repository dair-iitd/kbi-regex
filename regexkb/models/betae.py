import torch
from torch.nn import Embedding, Linear
import torch.nn.functional as F

import losses

from .pl_model import Model
from regexkb import constants


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class BetaProjection(torch.nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = torch.nn.Linear(
            self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = torch.nn.Linear(
            self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), torch.nn.Linear(
                self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            torch.nn.init.xavier_uniform_(
                getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        # for disjunction
        if r_embedding.shape[0] == 2 and e_embedding.shape[0] == 1:
            e_embedding = e_embedding.expand(2,-1,-1)
        if r_embedding.shape[0] == 1 and e_embedding.shape[0] == 2:
            r_embedding = r_embedding.expand(2,-1,-1)
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x


class BetaE(Model):
    """
    Beta-E model for learning regular expressions.
    """

    def __init__(self, args):

        super().__init__()
        self.save_hyperparameters(args)
        self.kleene_plus_op = args.kleene_plus_op
        self.disjunction_op = args.disjunction_op
        self.gamma = args.margin
        init_scale = 1e-3

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

        # make sure the parameters of beta embeddings are positive
        self.entity_regularizer = Regularizer(1, 0.05, 1e9)
        # make sure the parameters of beta embeddings after relation projection are positive
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)

        embedding_range = (args.margin + args.epsilon) / args.embedding_dim
        torch.nn.init.uniform_(
            self.E.weight, a=-embedding_range, b=embedding_range)
        torch.nn.init.uniform_(
            self.R.weight, a=-embedding_range, b=embedding_range)

        hidden_dim, num_layers = args.beta_mode
        self.projection_net = BetaProjection(args.embedding_dim,
                                             args.embedding_dim // 2,
                                             hidden_dim,
                                             self.projection_regularizer,
                                             num_layers)
        self.loss = getattr(losses, args.loss)(args)
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

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(
            entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(
            alpha_embedding, beta_embedding)
        logit = torch.norm(torch.distributions.kl.kl_divergence(
            entity_dist, query_dist), p=1, dim=-1)
        return logit

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

        def gqe_disjunction_center(rels):
            temp1 = F.relu(self.disjunction_R_premat(rels))
            combined = self.agg_func(temp1)
            return self.disjunction_R_postmat(combined)

        head = self.entity_regularizer((self.E(s))).unsqueeze(0)
        # to be modified
        # tail = (self.E(o) if o is not None else self.E.weight).unsqueeze(0)

        # (e1, r, e2)
        if query_type == 0:
            assert r.shape[1] == 1
            r = r.squeeze(1)
            query = self.projection_net(head, self.R(r).unsqueeze(0))

        # (e1, r+, e2)
        elif query_type == 1:
            assert r.shape[1] == 1
            r = r.squeeze(1)
            query = self.projection_net(head, kleene_op(r).unsqueeze(0))

        # (e1, r1+, r2+, e2)
        elif query_type == 2:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r2).unsqueeze(0))

        # (e1, r1+, r2+, r3+, e2)
        elif query_type == 3:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r2).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r3).unsqueeze(0))

        # (e1, r1, r2, e2)
        elif query_type == 21:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            query = self.projection_net(head, self.R(r1).unsqueeze(0))
            query = self.projection_net(query, self.R(r2).unsqueeze(0))

        # (e1, r1, r2+, e2)
        elif query_type == 4:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            query = self.projection_net(head, self.R(r1).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r2).unsqueeze(0))

        # (e1, r1+, r2, e2)
        elif query_type == 5:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
            query = self.projection_net(query, self.R(r2).unsqueeze(0))

        # (e1, r1+, r2+, r3, e2)
        elif query_type == 6:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r2).unsqueeze(0))
            query = self.projection_net(query, self.R(r3).unsqueeze(0))

        # (e1, r1+, r2, r3+, e2)
        elif query_type == 7:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
            query = self.projection_net(query, self.R(r2).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r3).unsqueeze(0))

        # (e1, r1, r2+, r3+, e2)
        elif query_type == 8:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, self.R(r1).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r2).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r3).unsqueeze(0))

        # (e1, r1, r2, r3+, e2)
        elif query_type == 9:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, self.R(r1).unsqueeze(0))
            query = self.projection_net(query, self.R(r2).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r3).unsqueeze(0))

        # (e1, r1, r2+, r3, e2)
        elif query_type == 10:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, self.R(r1).unsqueeze(0))
            query = self.projection_net(query, kleene_op(r2).unsqueeze(0))
            query = self.projection_net(query, self.R(r3).unsqueeze(0))

        # (e1, r1+, r2, r3, e2)
        elif query_type == 11:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
            query = self.projection_net(query, self.R(r2).unsqueeze(0))
            query = self.projection_net(query, self.R(r3).unsqueeze(0))

        # (e1, r1/r2, e2)
        elif query_type == 12:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0)).unsqueeze(0)
                query = self.projection_net(head, rel)
            else:
                rel = torch.stack([self.R(r1),
                                   self.R(r2)], dim=0)
                query = self.projection_net(head, rel)

        # (e1, r1/r2, r3, e2)
        elif query_type == 13:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0)).unsqueeze(0)
                query = self.projection_net(head, rel)
                query = self.projection_net(query, self.R(r3).unsqueeze(0))
            else:
                rel = torch.stack([self.R(r1),
                                   self.R(r2)], dim=0)
                query = self.projection_net(head, rel)
                query = self.projection_net(query, self.R(r3).unsqueeze(0))

        # (e1, r1, r2/r3, e2)
        elif query_type == 14:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                query = self.projection_net(head, self.R(r1).unsqueeze(0))
                rel = gqe_disjunction_center(torch.stack([self.R(r2),
                                                          self.R(r3)], dim=0)).unsqueeze(0)
                query = self.projection_net(query, rel)
            else:
                query = self.projection_net(head, self.R(r1).unsqueeze(0))
                rel = torch.stack([self.R(r2), self.R(r3)], dim=0)
                query = self.projection_net(query, rel)

        # (e1, r1+/r2+, e2)
        elif query_type == 15:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([kleene_op(r1),
                                                          kleene_op(r2)], dim=0)).unsqueeze(0)
                query = self.projection_net(head, rel)
            else:
                rel = torch.stack([kleene_op(r1), kleene_op(r2)], dim=0)
                query = self.projection_net(head, rel)

        # (e1, r1/r2, r3+, e2)
        elif query_type == 16:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0)).unsqueeze(0)
                query = self.projection_net(head, rel)
                query = self.projection_net(query, kleene_op(r3).unsqueeze(0))
            else:
                rel = torch.stack([self.R(r1),
                                   self.R(r2)], dim=0)
                query = self.projection_net(head, rel)
                query = self.projection_net(query, kleene_op(r3).unsqueeze(0))

        # (e1, r1+/r2+, r3, e2)
        elif query_type == 17:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([kleene_op(r1),
                                                          kleene_op(r2)], dim=0)).unsqueeze(0)
                query = self.projection_net(head, rel)
                query = self.projection_net(query, self.R(r3).unsqueeze(0))
            else:
                rel = torch.stack([kleene_op(r1),
                                   kleene_op(r2)], dim=0)
                query = self.projection_net(head, rel)
                query = self.projection_net(query, self.R(r3).unsqueeze(0))

        # (e1, r1+, r2/r3, e2)
        elif query_type == 18:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
                rel = gqe_disjunction_center(torch.stack([self.R(r2),
                                                          self.R(r3)], dim=0)).unsqueeze(0)
                query = self.projection_net(query, rel)
            else:
                query = self.projection_net(head, kleene_op(r1).unsqueeze(0))
                rel = torch.stack([self.R(r2), self.R(r3)], dim=0)
                query = self.projection_net(query, rel)

        # (e1, r1, r2+/r3+, e2)
        elif query_type == 19:
            assert r.shape[1] == 3
            r1 = r[:, 0]
            r2 = r[:, 1]
            r3 = r[:, 2]
            if self.disjunction_op == constants.GQE:
                query = self.projection_net(head, self.R(r1).unsqueeze(0))
                rel = gqe_disjunction_center(torch.stack([kleene_op(r2),
                                                          kleene_op(r3)], dim=0)).unsqueeze(0)
                query = self.projection_net(query, rel)
            else:
                query = self.projection_net(head, self.R(r1).unsqueeze(0))
                rel = torch.stack([kleene_op(r2), kleene_op(r3)], dim=0)
                query = self.projection_net(query, rel)
        
        # (e1, (r1/r2)+, e2)
        elif query_type == 20:
            assert r.shape[1] == 2
            r1 = r[:, 0]
            r2 = r[:, 1]
            if self.disjunction_op == constants.GQE and self.kleene_plus_op == constants.GQE:
                rel = gqe_disjunction_center(torch.stack([self.R(r1),
                                                          self.R(r2)], dim=0))
                rel = kleene_op(rel).unsqueeze(0)
                query = self.projection_net(head, rel)   
            else:
                assert False              

        else:
            assert False

        query_alpha_embedding, query_beta_embedding = torch.chunk(
            query, 2, dim=-1)

        if o is None:
            # evaluation
            tail = self.entity_regularizer(
                self.E.weight).unsqueeze(0).unsqueeze(0)
            query_alpha_embedding = query_alpha_embedding.unsqueeze(2)
            query_beta_embedding = query_beta_embedding.unsqueeze(2)
        elif s.shape == o.shape:
            # positive sample
            tail = self.entity_regularizer(self.E(o)).unsqueeze(0)
        else:
            # negative sample
            tail = self.entity_regularizer(self.E(o)).unsqueeze(0)
            query_alpha_embedding = query_alpha_embedding.unsqueeze(2)
            query_beta_embedding = query_beta_embedding.unsqueeze(2)

        query_dist = torch.distributions.beta.Beta(
            query_alpha_embedding, query_beta_embedding)
        return -1 * torch.min(self.cal_logit_beta(tail, query_dist), dim=0)[0]
