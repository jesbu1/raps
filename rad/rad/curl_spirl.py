import copy
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.torch_17_distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Normal

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
        self.linear = []
        encoder_feature_dim = self.encoder.feature_dim
        for _ in range(num_linear_layers - 1):
            self.linear.append(nn.Linear(encoder_feature_dim, encoder_feature_dim))
            self.linear.append(nn.ReLU())
        self.linear.append(nn.Linear(encoder_feature_dim, output_dim * 2))
        self.linear = nn.Sequential(*self.linear)

    def forward(self, obs, one_hot=None):
        h = self.encoder(obs, film_input=one_hot)
        mu, log_sigma = self.linear(h).chunk(2, dim=-1)
        return mu, log_sigma


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
            self.model = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=action_dim + one_hot_dim,
                    nhead=3,
                    dim_feedforward=hidden_dim,
                    dropout=0.0,
                    batch_first=True,
                ),
                num_layers=n_layers,
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
            h = h[:, -1]
            import pdb

            pdb.set_trace()

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
    ):
        super().__init__()
        if model_type == "rnn":
            self.model = nn.GRU(
                latent_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
            )
        elif model_type == "transformer":
            self.model = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=latent_dim,
                    nhead=2,
                    dim_feedforward=hidden_dim,
                    dropout=0.0,
                    batch_first=True,
                ),
                num_layers=n_layers,
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.nonlinearity = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.model_type = model_type

    def forward(self, latents, obs=None, one_hot=None):
        if self.model_type == "rnn":
            decoded_actions, _ = self.model(latents)
        elif self.model_type == "transformer":
            decoded_actions = self.model(latents)
        # resize
        decoded_actions = decoded_actions.reshape(
            (
                decoded_actions.shape[0] * decoded_actions.shape[1],
                decoded_actions.shape[2],
            )
        )
        acs = self.norm(decoded_actions)
        acs = self.nonlinearity(acs)
        acs = self.linear(acs)
        acs = acs.reshape((latents.shape[0], latents.shape[1], acs.shape[1]))
        return acs


class ClosedLoopActionDecoder(nn.Module):
    # this will be an Action Decoder MLP
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        output_dim,
        num_linear_layers,
        obs_encoder_type,
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
        self.obs_input_size = self.encoder.feature_dim
        layers = []
        for i in range(num_linear_layers - 1):
            if i == 0:
                layers.append(nn.Linear(latent_dim + self.obs_input_size, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.linear = nn.Sequential(*layers)

    def forward(self, latents, obs, one_hot=None):
        # reshape obs as it's a sequence
        if len(obs.shape) == 5:
            # images
            obs = obs.reshape(
                (obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            )
        else:
            # state
            obs = obs.reshape((obs.shape[0] * obs.shape[1], obs.shape[2]))
        obs = self.encoder(obs, one_hot)
        original_shape = latents.shape
        latents = latents.reshape(
            (latents.shape[0] * latents.shape[1], latents.shape[2])
        )
        latents = torch.cat((latents, obs), -1)
        acs = self.linear(latents)
        acs = acs.reshape((original_shape[0], original_shape[1], acs.shape[1]))
        return acs


class SPiRLRadSacAgent(RadSacAgent, nn.Module):
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
        use_amp=True,
        # SPiRL specific parameters below
        spirl_latent_dim=10,
        spirl_closed_loop=False,
        use_film=False,
        spirl_architecture="rnn",
        spirl_beta=0.1,
        spirl_action_horizon=10,
        target_prior_divergence=5.0,
        # checkpoint loading
        ckpt_load_dir=None,
        **kwargs
    ):
        # nn.Module empty init
        nn.Module.__init__(self)
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

        if self.encoder_type == "identity":
            encoder_feature_dim = obs_shape[
                0
            ]  # this is a hack to ensure that state space a

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
        )

        self.critic = Critic(
            obs_shape,
            continuous_action_dim + discrete_action_dim,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        )

        self.critic_target = Critic(
            obs_shape,
            continuous_action_dim + discrete_action_dim,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        )

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        # self.target_entropy = -np.prod(continuous_action_dim + discrete_action_dim)
        self.target_kl = target_prior_divergence

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
            )

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
            n_layers=1,  # just one processing layer is fine
            model_type=spirl_architecture,
        )
        self.spirl_prior = Actor(
            obs_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
            discrete_continuous_dist=False,
            continuous_action_dim=spirl_latent_dim,
            discrete_action_dim=0,
        )
        if spirl_closed_loop:
            self.spirl_decoder = ClosedLoopActionDecoder(
                spirl_latent_dim,
                hidden_dim,
                env_action_dim,
                num_linear_layers=num_layers,
                obs_encoder_type=encoder_type,
                obs_shape=obs_shape,
                obs_encoder_output_dim=encoder_feature_dim,
                obs_encoder_num_layers=num_layers,
                obs_encoder_num_filters=num_filters,
                use_film=use_film,
            )
        else:
            self.spirl_decoder = ActionDecoder(
                spirl_latent_dim,
                hidden_dim,
                env_action_dim,
                n_layers=1,
                model_type=spirl_architecture,
            )
        self.spirl_optimizer = torch.optim.Adam(
            list(self.spirl_encoder.parameters())
            + list(self.spirl_prior.parameters())
            + list(self.spirl_decoder.parameters()),
            lr=encoder_lr,
        )
        self.use_amp = use_amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.spirl_action_horizon = spirl_action_horizon
        # now set the actor's parameters to the ObsPrior
        self.actor.load_state_dict(self.spirl_prior.state_dict())

        # load model from checkpoint
        if ckpt_load_dir:
            self.load(ckpt_load_dir)

        # TODO: discount according to variable length skills
        # TODO: freeze the closed loop decoder's encoder maybe? prob not
        self.reset()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == "pixel":
            self.CURL.train(training)

    def train_spirl(self, training=True):
        self.spirl_encoder.train(training)
        self.spirl_prior.train(training)
        self.spirl_decoder.train(training)

    def update_critic(self, obs, action, reward, next_obs, not_done, log_dict, step):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.no_grad():
                policy_mu, policy_action, _, log_std = self.actor(
                    next_obs, compute_log_pi=False
                )
                prior_mu, _, _, prior_log_std = self.spirl_prior(
                    next_obs, compute_log_pi=False
                )
                divergence_from_prior = utils.gaussian_kl_divergence(
                    policy_mu, log_std, prior_mu, prior_log_std
                )

                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
                target_V = (
                    torch.min(target_Q1, target_Q2)
                    - self.alpha.detach() * divergence_from_prior
                )
                target_Q = reward + (not_done * self.discount * target_V)
            # get current Q estimates
            current_Q1, current_Q2 = self.critic(
                obs, action, detach_encoder=self.detach_encoder
            )
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
            if step % self.log_interval == 0:
                log_dict["train_critic/loss"] = critic_loss.item()

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            self.grad_scaler.scale(critic_loss).backward()
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()

    def update_actor_and_alpha(self, obs, log_dict, step):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # detach encoder, so we don't update it with the actor loss
            _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)
            with torch.no_grad():
                prior_mu, _, _, prior_log_std = self.spirl_prior(
                    obs, compute_log_pi=False
                )
                entropy = 0.5 * log_std.shape[1] * (
                    1.0 + np.log(2 * np.pi)
                ) + log_std.sum(dim=-1)
            prior_kl_divergence = utils.gaussian_kl_divergence(
                pi, log_std, prior_mu, prior_log_std
            )

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * prior_kl_divergence - actor_Q).mean()

            if step % self.log_interval == 0:
                log_dict["train_actor/loss"] = actor_loss.item()
                log_dict["train_actor/target_kl"] = self.target_kl
                log_dict["train_actor/entropy"] = entropy.mean().item()
                log_dict[
                    "train_actor/prior_kl_divergence"
                ] = prior_kl_divergence.mean().item()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            self.grad_scaler.scale(actor_loss).backward()
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()

            # optimize the KL div regularization coef alpha
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_kl).detach()).mean()
            if step % self.log_interval == 0:
                log_dict["train_alpha/loss"] = alpha_loss.item()
                log_dict["train_alpha/value"] = self.alpha.item()
            self.grad_scaler.scale(alpha_loss).backward()
            self.grad_scaler.step(self.log_alpha_optimizer)
            self.grad_scaler.update()

    def spirl_update(self, replay_buffer, step):
        # TODO: integrate a multi-skill spirl version (not yet) as I need to first test straightforward spirl
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            obs, actions = replay_buffer.get_batch()

            # compute prior
            # select first obs for the prior
            first_obs = obs[:, 0]
            prior_mu, _, _, prior_log_std = self.spirl_prior(
                first_obs, compute_pi=False, compute_log_pi=False
            )

            # compute posterior
            posterior_mu, posterior_log_std = self.spirl_encoder(actions)

            # compute kl divergence for the prior
            kl_div_prior = utils.gaussian_kl_divergence(
                posterior_mu.detach(),
                posterior_log_std.detach(),
                prior_mu,
                prior_log_std,
            )

            # compute KL divergence for the encoder
            gaussian_mu = torch.zeros_like(posterior_mu)
            gaussian_log_std = torch.zeros_like(posterior_log_std)
            kl_div_encoder = utils.gaussian_kl_divergence(
                posterior_mu, posterior_log_std, gaussian_mu, gaussian_log_std
            )

            # sample from posterior gaussian
            z = Normal(posterior_mu, posterior_log_std.exp()).rsample()

            # compute reconstruction loss
            z = z.unsqueeze(1).expand(-1, actions.shape[1], -1)
            # get all but last from obs
            obs = obs[:, :-1]
            recon = self.spirl_decoder(z, obs)
            recon_loss = F.mse_loss(recon, actions)

            # compute total loss
            loss = self.spirl_beta * kl_div_encoder + recon_loss + kl_div_prior

            # update
            self.spirl_optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.spirl_optimizer)
            self.grad_scaler.update()

            # log
            log_dict = {}
            if step % self.log_interval == 0:
                # log reconstruction loss with spirl prior
                with torch.no_grad():
                    prior_mu = prior_mu.unsqueeze(1).expand(-1, actions.shape[1], -1)
                    prior_recon = self.spirl_decoder(prior_mu, obs)
                    prior_recon_loss = F.mse_loss(prior_recon, actions)
                log_dict = dict(
                    recon_loss=recon_loss.item(),
                    kl_div_prior=kl_div_prior.item(),
                    kl_div_encoder=kl_div_encoder.item(),
                    prior_recon_loss=prior_recon_loss.item(),
                )
        return log_dict

    def reset(self):
        # reset all of the online RL stuff
        self.current_action_trajs = []
        self.current_action_horizon = 0
        self.current_latent = None
        self.sampled_new_action = False

    def get_deterministic_action_from_decoder(self, input_obs, sample_hl_action):
        # TODO: handle one-hot case
        # TODO: also support parallel envs
        with torch.no_grad():
            if self.encoder_type == "pixel":
                obs = input_obs / 255.0
            else:
                obs = input_obs
            # now get action from LL policy (spirl decoder)
            self.current_action_horizon += 1
            if self.current_action_horizon == self.spirl_action_horizon:
                # reset
                self.reset()

            if self.current_latent is None:
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                # get action from HL policy
                if sample_hl_action:
                    _, pi, _, _ = self.actor(obs, compute_log_pi=False)
                    self.current_latent = pi
                else:
                    mu, _, _, _ = self.actor(
                        obs, compute_pi=False, compute_log_pi=False
                    )
                    self.current_latent = mu
                self.sampled_new_action = True
            else:
                self.sampled_new_action = False

        if obs.shape[-1] != self.image_size and self.encoder_type == "pixel":
            obs = utils.center_crop_image(obs, self.image_size)
        if self.spirl_closed_loop:
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
            ac = (
                self.spirl_decoder(
                    self.current_latent,
                    obs,
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        else:
            if len(self.current_action_trajs) == 0:
                # sample new action traj
                repeated_latent = self.current_latent.unsqueeze(1).expand(
                    -1, self.spirl_action_horizon, -1
                )
                ac_traj = self.spirl_decoder(repeated_latent)
                self.current_action_trajs = list(
                    ac_traj.squeeze(0).detach().cpu().numpy()
                )
            ac = self.current_action_trajs.pop(0).flatten()
        return ac

    def select_action(self, obs):
        return self.get_deterministic_action_from_decoder(obs, sample_hl_action=False)

    def sample_action(self, obs):
        return self.get_deterministic_action_from_decoder(obs, sample_hl_action=True)

    def update(self, replay_buffer, log_dict, step):
        if self.encoder_type == "pixel":
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(
                self.augs_funcs
            )
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            log_dict["train/batch_reward"] = reward.mean().item()

        self.update_critic(obs, action, reward, next_obs, not_done, log_dict, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, log_dict, step)

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
        if hasattr(self, "CURL"):
            torch.save(self.CURL.state_dict(), "%s/curl_%s.pt" % (model_dir, step))

    def load_curl(self, model_dir, step):
        if hasattr(self, "CURL"):
            self.CURL.load_state_dict(torch.load("%s/curl_%s.pt" % (model_dir, step)))

    def load(self, model_dir):
        # check if actor or critic or curl checkpoints exist. if they do use them
        model_ckpts = os.listdir(model_dir)
        all_actors = [x for x in model_ckpts if "actor" in x]
        # sort by step, decreasing
        all_actors = sorted(
            all_actors, key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True
        )
        if len(all_actors) > 0:
            step = all_actors[0].split("_")[-1].split(".")[0]
            self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
            self.critic.load_state_dict(
                torch.load("%s/critic_%s.pt" % (model_dir, step))
            )
            self.critic_target.load_state_dict(
                torch.load("%s/critic_target_%s.pt" % (model_dir, step))
            )
            self.load_curl(model_dir, step)
            self.log_alpha = nn.Parameter(
                torch.load("%s/log_alpha_%s.pt" % (model_dir, step))
            )
            self.log_alpha_optimizer.load_state_dict(
                torch.load("%s/log_alpha_optim_%s.pt" % (model_dir, step))
            )
            self.actor_optimizer.load_state_dict(
                torch.load("%s/actor_optim_%s.pt" % (model_dir, step))
            )
        # latest spirl checkpoint
        all_spirl = [x for x in model_ckpts if "spirl" in x]
        all_spirl = sorted(
            all_spirl, key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True
        )
        spirl_step = all_spirl[0].split("_")[-1].split(".")[0]
        self.load_spirl(model_dir, spirl_step)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))
        torch.save(
            self.critic_target.state_dict(),
            "%s/critic_target_%s.pt" % (model_dir, step),
        )
        torch.save(
            self.log_alpha_optimizer.state_dict(),
            "%s/log_alpha_optim_%s.pt" % (model_dir, step),
        )
        torch.save(
            self.actor_optimizer.state_dict(),
            "%s/actor_optim_%s.pt" % (model_dir, step),
        )
        self.save_curl(model_dir, step)
        self.save_spirl(model_dir, step)

    def save_spirl(self, model_dir, step):
        # remove old spirl
        model_ckpts = os.listdir(model_dir)
        all_spirl = [x for x in model_ckpts if "spirl" in x]
        for spirl_ckpt in all_spirl:
            os.remove(os.path.join(model_dir, spirl_ckpt))
        # save new spirl
        torch.save(
            self.spirl_encoder.state_dict(),
            "%s/spirl_encoder_%s.pt" % (model_dir, step),
        )
        torch.save(
            self.spirl_prior.state_dict(), "%s/spirl_prior_%s.pt" % (model_dir, step)
        )
        torch.save(
            self.spirl_decoder.state_dict(),
            "%s/spirl_decoder_%s.pt" % (model_dir, step),
        )
        torch.save(
            self.spirl_optimizer.state_dict(),
            "%s/spirl_optimizer_%s.pt" % (model_dir, step),
        )

    def load_spirl(self, model_dir, step):
        self.spirl_encoder.load_state_dict(
            torch.load("%s/spirl_encoder_%s.pt" % (model_dir, step))
        )
        self.spirl_prior.load_state_dict(
            torch.load("%s/spirl_prior_%s.pt" % (model_dir, step))
        )
        self.spirl_decoder.load_state_dict(
            torch.load("%s/spirl_decoder_%s.pt" % (model_dir, step))
        )
        self.spirl_optimizer.load_state_dict(
            torch.load("%s/spirl_optimizer_%s.pt" % (model_dir, step))
        )
