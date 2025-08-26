import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.policies.rt1.rt1_model import RT1Inference

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "pi0":
        from simpler_env.policies.pi0.pi0_server_model import OpenPiFastInference
        model = OpenPiFastInference(
            server_ip=args.server_ip,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_ensemble_temp=args.action_ensemble_temp,
        )
    elif args.policy_model == "lerobot":
        from simpler_env.policies.lerobot.lerobot_server_model import LerobotServerModel
        model = LerobotServerModel(
            server_ip=args.server_ip,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_ensemble_temp=args.action_ensemble_temp,
        )
    elif args.policy_model == "otter":
        # follow the same structure as LerobotServerModel
        from simpler_env.policies.lerobot.lerobot_server_model import LerobotServerModel
        model = LerobotServerModel(
            server_ip=args.server_ip,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_ensemble_temp=args.action_ensemble_temp,
        )
    else:
        raise NotImplementedError()

    # Initialize wandb logging if enabled
    # add date-time to group
    args.wandb_run_name = (
        f"{args.wandb_run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_run_group,
            entity=args.wandb_entity,
            config=vars(args),
        )
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    if args.use_wandb:
        # Log overall success rate
        success_rate = np.mean(success_arr)
        wandb.log(
            {
                "success_rate": success_rate,
                "total_episodes": len(success_arr),
                "successful_episodes": np.sum(success_arr),
            }
        )

        # Log detailed episode results
        for i, success in enumerate(success_arr):
            wandb.log({f"episode_{i}_success": success})

        wandb.finish()