import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from rad.kitchen_spirl_pretrain import experiment

    experiment(variant)


if __name__ == "__main__":
    # TODO: Fill this out
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_load_dir", type=str, required=True)
    # misc
    # parser.add_argument("--exp_prefix", type=str, required=True)
    parser.add_argument("--run_group", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--env", type=str, default="kitchen-mixed-v0")
    # train
    parser.add_argument("--agent", default="rad_sac", type=str)
    parser.add_argument("--hidden_dim", default=128, type=int)
    # critic
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument(
        "--critic_target_update_freq", default=2, type=int
    )  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)
    parser.add_argument("--discrete_continuous_dist", default=0, type=int)
    # encoder
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--latent_dim", default=128, type=int)
    # sac
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)
    parser.add_argument("--detach_encoder", default=False, action="store_true")
    # spirl
    parser.add_argument("--spirl_latent_dim", default=10, type=int)
    parser.add_argument("--spirl_closed_loop", default=False, action="store_true")
    parser.add_argument("--spirl_architecture", default="rnn", type=str)
    parser.add_argument("--use_film", default=False, action="store_true")
    parser.add_argument("--spirl_beta", default=5e-4, type=float)
    parser.add_argument("--spirl_action_horizon", default=10, type=int)
    args = parser.parse_args()
    variant = dict(
        run_group=args.run_group,
        agent_kwargs=dict(
            discount=0.99,
            critic_lr=2e-4,
            actor_lr=2e-4,
            encoder_lr=2e-4,
            encoder_type="identity",
            data_augs="no_aug",
            use_amp=True,
            log_interval=50,
            env_action_dim=9,
            discrete_continuous_dist=False,
            target_prior_divergence=5.0,
            **vars(args),
        ),
        frame_stack=1,
        replay_buffer_capacity=int(2.5e6),
        action_repeat=1,
        num_eval_episodes=5,
        num_train_steps=int(1e6),
        init_steps=2500,
        pre_transform_image_size=64,
        image_size=64,
        env_name="kitchen-mixed-v0",  # slide-cabinet
        batch_size=512,  # 512 originally for online RL
        eval_freq=10000,
        log_interval=1000,
        env_kwargs=dict(
            dense=False,
            image_obs=False,
            action_scale=1,
            control_mode="joint_velocity",  # default joint velocity control
            frame_skip=40,  # default for D4RL kitchen
            imwidth=84,
            imheight=84,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=280,
                unflatten_images=True,
            ),
            image_kwargs=dict(),
        ),
        seed=-1,
        use_raw_actions=True,
        env_suite="kitchen",
    )

    search_space = {
        "agent_kwargs.data_augs": [
            "no_aug",
        ],
        "agent_kwargs.discrete_continuous_dist": [False],
        "env_name": [args.env],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id
            run_experiment(
                experiment,
                exp_prefix=args.run_group,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="none",
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
            )
