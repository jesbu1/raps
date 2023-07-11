import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.torch_17_distributions.one_hot_categorical import OneHotCategorical

from rad.curl_sac import OneHotDist, Actor, QFunction, Critic, CURL, RadSacAgent

import rad.data_augs as da
import rad.utils as utils
from rad.encoder import make_encoder

LOG_FREQ = 10000

class ObsPrior(nn.Module):
    def __init__(
        self,
        output_dim,
        obs_encoder_type,
        num_linear_layers,
        obs_shape,
        obs_encoder_output_dim,
        obs_encoder_num_layers,
        obs_encoder_num_filters,
        use_film,
        film_input_dim=0,
    ):
        super().__init__()
        self.encoder = make_encoder(
            obs_encoder_type,
            obs_shape,
            obs_encoder_output_dim,
            obs_encoder_num_layers,
            obs_encoder_num_filters,
            output_logits=True,
            film=use_film,
            film_input_dim=film_input_dim, 
        )
        self.linear = nn.Sequential(
            *[nn.Linear(obs_encoder_output_dim, obs_encoder_output_dim), nn.ReLU() for _ in range(num_linear_layers - 1)],
            nn.Linear(obs_encoder_output_dim, output_dim),
        )

    def forward(self, obs, one_hot=None):
        h = self.encoder(obs, film_input=one_hot)
        return self.linear(h)

class ActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        hidden_dim,
        output_dim,
        n_layers,
        model_type="rnn",
        one_hot_dim=0,
    ):
        super().__init__()
        if model_type == "rnn":
            self.model = nn.GRU(
                action_dim + one_hot_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
            )
        elif model_type == "transformer":
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=action_dim + one_hot_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim,
                    dropout=0.0,
                ),
                n_layers,
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.nonlinearity = nn.ReLU()
        self.linear = nn.Linear(hidden_dim + one_hot_dim, output_dim * 2)
        self.model_type = model_type

    def forward(self, action_traj, one_hot=None):
        if one_hot:
            action_traj = torch.cat((action_traj, one_hot), -1)
        if self.model_type == "rnn":
            _, h = self.model(action_traj)
            h = h[-1]
        elif self.model_type == "transformer":
            h = self.model(action_traj)

        h = self.norm(h)
        h = self.nonlinearity(h)
        if one_hot:
            h = torch.cat((h, one_hot), -1)
        mu, log_std = self.linear(h).chunk(2, dim=-1)
        return mu, log_std

class ActionDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        hidden_dim,
        output_dim,
        n_layers,
        model_type="rnn",
        closed_loop=False,
        cl_encoder=None,
        ):
        super().__init__()
        self.closed_loop = closed_loop
        if closed_loop:
            assert cl_encoder is not None
            self.cl_encoder = cl_encoder
            self.obs_input_size = cl_encoder.feature_dim
        else:
            self.obs_input_size = 0

        if model_type == "rnn":
            self.model = nn.GRU(
                latent_dim + self.obs_input_size,
                hidden_dim,
                n_layers,
                batch_first=True,
            )
        elif model_type == "transformer":
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim,
                    dropout=0.0,
                ),
                n_layers,
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.nonlinearity = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.model_type = model_type
        self.closed_loop = closed_loop
        self.cl_encoder = cl_encoder

    def forward(self, action_traj, obs=None, one_hot=None):
        if self.closed_loop:
            obs = self.cl_encoder(obs, one_hot)
            obs = obs.unsqueeze(1).expand(-1, action_traj.shape[1], -1)
            action_traj = torch.cat((action_traj, obs), -1)
        if self.model_type == "rnn":
            _, h = self.model(action_traj)
            h = h[-1]
        elif self.model_type == "transformer":
            h = self.model(action_traj)
        h = self.norm(h)
        h = self.nonlinearity(h)
        return self.linear(h)
    




class SPiRLRadSacAgent(RadSacAgent):
    """SPIRL built on top of RAD with SAC."""

    def __init__(
        self,
        obs_shape,
        discrete_continuous_dist,
        continuous_action_dim,
        discrete_action_dim,
        env_action_dim,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs="",
        # SPiRL specific parameters below
        spirl_latent_dim=10,
        spirl_encoder_type="pixel",
        spirl_closed_loop=False,
        use_film=False,
        spirl_architecture="rnn",
        spirl_beta = 0.1,
    ):
        torch.backends.cudnn.benchmark = True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.data_augs = data_augs

        self.augs_funcs = {}

        aug_to_func = {
            "crop": da.random_crop,
            "grayscale": da.random_grayscale,
            "cutout": da.random_cutout,
            "cutout_color": da.random_cutout_color,
            "flip": da.random_flip,
            "rotate": da.random_rotation,
            "rand_conv": da.random_convolution,
            "color_jitter": da.random_color_jitter,
            "translate": da.random_translate,
            "no_aug": da.no_aug,
        }

        for aug_name in self.data_augs.split("-"):
            assert aug_name in aug_to_func, "invalid data aug string"
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

        self.actor = Actor(
            obs_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
            discrete_continuous_dist,
            continuous_action_dim,
            discrete_action_dim,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            continuous_action_dim + discrete_action_dim,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target = Critic(
            obs_shape,
            continuous_action_dim + discrete_action_dim,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(continuous_action_dim + discrete_action_dim)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == "pixel":
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(
                obs_shape,
                encoder_feature_dim,
                self.latent_dim,
                self.critic,
                self.critic_target,
                output_type="continuous",
            ).to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(self.CURL.parameters(), lr=encoder_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

        # SPiRL specific stuff
        self.spirl_num_skills = discrete_action_dim
        self.spirl_latent_dim = spirl_latent_dim
        self.use_film = use_film
        self.spirl_closed_loop = spirl_closed_loop
        self.model_type = spirl_architecture
        self.spirl_beta = spirl_beta
        self.spirl_encoder = ActionEncoder(
            env_action_dim,
            hidden_dim,
            spirl_latent_dim,
            num_layers=1,  # just one processing layer is fine
            model_type=spirl_architecture,
        )
        self.spirl_prior = ObsPrior(
            spirl_latent_dim,
            spirl_encoder_type,
            2,
            obs_shape,
            encoder_feature_dim,
            num_layers,
            num_filters,
        )
        self.spirl_decoder = ActionDecoder(
            spirl_latent_dim,
            hidden_dim,
            env_action_dim,
            num_layers=1,
            closed_loop=spirl_closed_loop,
            cl_encoder=self.spirl_prior.encoder,
            model_type=spirl_architecture
        )
        self.spirl_optimizer = torch.optim.Adam(
            list(self.spirl_encoder.parameters()) + list(self.spirl_prior.parameters()) + list(self.spirl_decoder.parameters()),
            lr=encoder_lr,
        )
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == "pixel":
            self.CURL.train(training)

    def select_action(self, obs):
        # TODO: SPiRL integration
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        # TODO: SPiRL integration
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixel":
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                self.augs_funcs
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        # if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
        #    obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
        #    self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)



    def save_curl(self, model_dir, step):
        torch.save(self.CURL.state_dict(), "%s/curl_%s.pt" % (model_dir, step))


    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))
        torch.save(
            self.critic_target.state_dict(),
            "%s/critic_target_%s.pt" % (model_dir, step),
        )