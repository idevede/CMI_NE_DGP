import os
import numpy as np

import torch

from model.inference import Inference
from model.inference_cmi import InferenceCMI
from model.gumbel import GumbelMatrix, ConditionalGumbelMatrix
from utils.utils import to_numpy, preprocess_obs, postprocess_obs

EPS = 1e-4


class InferenceReg(InferenceCMI):
    def __init__(self, encoder, params):
        self.reg_params = reg_params = params.inference_params.reg_params
        self.flip_prob = reg_params.flip_prob_init
        super(InferenceReg, self).__init__(encoder, params)
        self.use_mask = reg_params.use_mask

        # regularization params
        self.lambda_M = reg_params.lambda_M_init
        self.lambda_I = reg_params.lambda_I_init

    def init_model(self):
        super(InferenceReg, self).init_model()

        reg_params = self.reg_params
        device = self.device
        flip_prob = self.flip_prob

        feature_dim = self.encoder.feature_dim

        self.adjacency = GumbelMatrix((feature_dim, feature_dim), reg_params.adjacency_init, flip_prob, device)
        self.interv_mask = GumbelMatrix((feature_dim, 1), reg_params.interv_mask_init, flip_prob, device)

        self.eye_mask = torch.eye(feature_dim, device=device).to(device)
        self.adjacency_mask = torch.ones(feature_dim, feature_dim, device=device) - self.eye_mask

    def init_graph(self, *args):
        pass

    def init_abstraction(self):
        self.abstraction_quested = False
        self.abstraction_graph = None
        self.should_update_abstraction = False
        self.should_update_abstracted_dynamics = False

    def init_cache(self):
        pass

    def reset_causal_graph_eval(self):
        pass

    def setup_training_mask_distribution(self):
        pass

    def forward_step(self, feature, action):
        """
        :param feature: if state space is continuous: (bs, feature_dim).
            Otherwise: [(bs, feature_i_dim)] * feature_dim
            if it is None, no need to forward it
        :param action: (bs, action_dim)
        :param mask: (bs, feature_dim, feature_dim + 1)
        :return: next step value for all state variables in the format of distribution,
            if state space is continuous: a Normal distribution of shape (bs, feature_dim)
            else: a list of distributions, [OneHotCategorical / Normal] * feature_dim, each of shape (bs, feature_i_dim)
        """
        # 1. extract features of all state variables and the action
        action_feature = self.extract_action_feature(action)            # (feature_dim, 1, bs, out_dim)
        state_feature = self.extract_state_feature(feature)             # (feature_dim, feature_dim, bs, out_dim)

        mask = None
        if self.use_mask:
            bs = action.shape[:-1]
            # adjacency mask
            M = self.adjacency(bs)                                      # (bs, feature_dim, feature_dim)
            M = M * self.adjacency_mask + self.eye_mask                 # (bs, feature_dim, feature_dim)

            # intervention mask
            I = self.interv_mask(bs)                                    # (bs, feature_dim, 1)

            mask = torch.cat([M, I], dim=-1)                            # (bs, feature_dim, feature_dim + 1)
            if self.training:
                mask = mask.permute(1, 2, 0)                            # (feature_dim, feature_dim + 1, bs)
            else:
                mask = mask.unsqueeze(dim=-1)
            mask = mask.unsqueeze(dim=-1)                               # (feature_dim, feature_dim + 1, bs, 1)

        # mask out unused features
        sa_feature = torch.cat([state_feature, action_feature], dim=1)  # (feature_dim, feature_dim + 1, bs, out_dim)
        if mask is not None:
            sa_feature = sa_feature * mask                              # (feature_dim, feature_dim + 1, bs, out_dim)
        sa_feature, sa_indices = sa_feature.max(dim=1)                  # (feature_dim, bs, out_dim)

        # 3. predict the distribution of next time step value
        dist = self.predict_from_sa_feature(sa_feature, feature)

        return dist

    def forward_with_feature(self, feature, actions, abstraction_mode=False):
        """

        :param feature: (bs, feature_dim) if state space is continuous else [(bs, feature_i_dim)] * feature_dim
            notice that bs can be a multi-dimensional batch size
        :param actions: (bs, n_pred_step, action_dim) if self.continuous_state else (bs, n_pred_step, 1)
        :param abstraction_mode: whether to only forward controllable & action-relevant state variables,
            used for model-based roll-out
        :return: distribution in the format of (sample, mean, log_std) if state space is continuous
            otherwise it is (sample, logits),
            the shape of each element is (bs, n_pred_step, feature_dim) if state space is continuous
            otherwise it is [(bs, n_pred_step, feature_i_dim)] * feature_dim
        """
        return Inference.forward_with_feature(self, feature, actions, abstraction_mode)

    def eval_prediction(self, obs, actions, next_obses):
        return Inference.eval_prediction(self, obs, actions, next_obses)

    def forward(self, obs, actions, abstraction_mode=False):
        feature = self.encoder(obs)
        return self.forward_with_feature(feature, actions, abstraction_mode)

    def setup_annealing(self, step):
        reg_params = self.reg_params
        reg_annealing_start = reg_params.reg_annealing_start
        reg_annealing_end = reg_params.reg_annealing_end
        reg_annealing_coef = np.clip((step - reg_annealing_start) / (reg_annealing_end - reg_annealing_start), 0, 1)

        lambda_M_init = reg_params.lambda_M_init
        lambda_M_final = reg_params.lambda_M_final
        lambda_I_init = reg_params.lambda_I_init
        lambda_I_final = reg_params.lambda_I_final

        self.lambda_M = lambda_M_init + reg_annealing_coef * (lambda_M_final - lambda_M_init)
        self.lambda_I = lambda_I_init + reg_annealing_coef * (lambda_I_final - lambda_I_init)

        flip_prob_init = reg_params.flip_prob_init
        flip_prob_final = reg_params.flip_prob_final
        self.flip_prob = flip_prob_init + reg_annealing_coef * (flip_prob_final - flip_prob_init)
        self.adjacency.flip_prob = self.flip_prob
        self.interv_mask.flip_prob = self.flip_prob

    def update(self, obses, actions, next_obses, eval=False):
        """
        :param obs: {obs_i_key: (bs, num_observation_steps, obs_i_shape)}
        :param actions: (bs, num_pred_steps, action_dim)
        :param next_obses: ({obs_i_key: (bs, num_pred_steps, obs_i_shape)}
        :return: {"loss_name": loss_value}
        """
        features = self.encoder(obses)
        next_features = self.encoder(next_obses)
        pred_next_dist = self.forward_with_feature(features, actions)

        # prediction loss in the state / latent space
        pred_loss = self.prediction_loss_from_dist(pred_next_dist, next_features)    # (bs, num_pred_steps)
        loss = pred_loss = pred_loss.sum(dim=-1).mean()
        loss_detail = {"pred_loss": pred_loss}

        if self.use_mask:
            # L1 reg for adjacency M and intervention mask I
            reg_M = self.lambda_M * self.adjacency.get_prob().sum()     # scalar
            reg_I = self.lambda_I * self.interv_mask.get_prob().sum()   # scalar
            loss += reg_M + reg_I
            loss_detail["reg_M"] = reg_M
            loss_detail["reg_I"] = reg_I

        if not eval:
            self.backprop(loss, loss_detail)

        if self.abstraction_quested:
            self.should_update_abstraction = True
            self.should_update_abstracted_dynamics = True

        return loss_detail

    def get_adjacency(self):
        if self.use_mask:
            M = self.adjacency.get_prob()
            M = M * self.adjacency_mask + self.eye_mask    # (feature_dim, feature_dim)
        else:
            M = torch.ones(self.feature_dim, self.feature_dim)
        return M

    def get_intervention_mask(self):
        if self.use_mask:
            return self.interv_mask.get_prob()
        else:
            return torch.ones(self.feature_dim, 1)

    def get_mask(self):
        M = self.get_adjacency()
        I = self.get_intervention_mask()
        mask = torch.cat([M, I], dim=-1) > self.reg_params.mask_threshold

        return mask

    def train(self, training=True):
        self.training = training

        flip_prob = self.flip_prob if training else 0

        self.adjacency.training = training
        self.interv_mask.training = training
        self.adjacency.flip_prob = flip_prob
        self.interv_mask.flip_prob = flip_prob

    def save(self, path):
        M = self.get_adjacency()
        I = self.get_intervention_mask()
        mask = torch.cat([M, I], dim=-1)
        torch.save({"model": self.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mask": mask,
                    }, path)

    def load(self, path, device):
        Inference.load(self, path, device)
