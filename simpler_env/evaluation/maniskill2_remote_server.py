import numpy as np
import time
from multiprocessing.connection import Listener
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

def run_maniskill2_remote_server(args):
    """
    Run a remote server that exposes the ManiSkill2 environment via a simple socket interface.
    Allows an external process (e.g., an RL training script) to step the environment.
    """
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    
    # Build environment (once)
    # We assume the scene and basic configs don't change between episodes for RL training
    # If they do, we might need to rebuild env on reset, but that's slow.
    
    additional_env_build_kwargs = args.additional_env_build_kwargs or {}
    
    kwargs = dict(
        obs_mode="rgbd",
        robot=args.robot,
        sim_freq=args.sim_freq,
        control_mode=control_mode,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        scene_name=args.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=args.rgb_overlay_path,
    )
    
    if args.enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict
        
    print(f"Building environment: {args.env_name}")
    env = build_maniskill2_env(
        args.env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    
    # Setup listener
    host = '0.0.0.0'
    port = 6000
    
    if args.server_ip:
        # Handle IP:PORT format or just IP
        parts = args.server_ip.split(':')
        host = parts[0]
        if len(parts) > 1:
            port = int(parts[1])
            
    address = (host, port)
        
    print(f"Starting remote server on {address}...")
    listener = Listener(address, authkey=b'simpler_secret')
    
    try:
        while True:
            print("Waiting for connection...")
            conn = listener.accept()
            print(f"Connection accepted from {listener.last_accepted}")
            
            try:
                while True:
                    cmd = conn.recv()
                    
                    if cmd['type'] == 'RESET':
                        # Pick random init parameters from args ranges
                        robot_init_x = np.random.choice(args.robot_init_xs)
                        robot_init_y = np.random.choice(args.robot_init_ys)
                        robot_init_quat = args.robot_init_quats[np.random.randint(len(args.robot_init_quats))]
                        
                        env_reset_options = {
                            "robot_init_options": {
                                "init_xy": np.array([robot_init_x, robot_init_y]),
                                "init_rot_quat": robot_init_quat,
                            }
                        }
                        
                        if args.obj_variation_mode == "xy":
                            obj_init_x = np.random.choice(args.obj_init_xs)
                            obj_init_y = np.random.choice(args.obj_init_ys)
                            env_reset_options["obj_init_options"] = {
                                "init_xy": np.array([obj_init_x, obj_init_y]),
                            }
                        elif args.obj_variation_mode == "episode":
                            obj_episode_id = np.random.randint(args.obj_episode_range[0], args.obj_episode_range[1])
                            env_reset_options["obj_init_options"] = {
                                "episode_id": obj_episode_id,
                            }
                            
                        obs, _ = env.reset(options=env_reset_options)
                        
                        # Get instruction
                        task_description = env.get_language_instruction()
                        
                        # Process observation
                        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
                        
                        # Return obs
                        conn.send({
                            'obs': obs,
                            'image': image,
                            'instruction': task_description
                        })
                        
                    elif cmd['type'] == 'STEP':
                        action = cmd['action']
                        # action here is expected to be the processed action (concatenated vector)
                        # DSRL should output the action vector directly matching the env action space
                        
                        # Note: maniskill2_evaluator does:
                        # obs, reward, done, truncated, info = env.step(np.concatenate([action["world_vector"], ...]))
                        # But DSRL model outputs a flat vector usually. We assume client sends the correct flat vector.
                        
                        # However, SimplerEnv models output a dictionary usually.
                        # If DSRL outputs a flat vector, we pass it directly.
                        
                        obs, reward, done, truncated, info = env.step(action)
                        
                        # Check for success / termination logic from evaluator?
                        # The evaluator checks `action["terminate_episode"]` from the model. 
                        # But here the model is on the client side. The client decides when to terminate or the env does.
                        # We just return what the env says.
                        
                        # But we also need to handle `advance_to_next_subtask` if applicable (LongHorizon)
                        # This is tricky because it depends on model output "terminate_episode".
                        # We will assume for now the client handles policy-based termination, 
                        # but we need to handle subtasks if the ENV requires it.
                        # Actually maniskill2_evaluator calls advance_to_next_subtask when MODEL predicts termination.
                        # So we should expose a command for that or let client handle it?
                        # For simplicity, we just step.
                        
                        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
                        
                        conn.send({
                            'obs': obs,
                            'image': image,
                            'reward': reward,
                            'done': done,
                            'truncated': truncated,
                            'info': info
                        })
                        
                    elif cmd['type'] == 'CLOSE':
                        env.close()
                        conn.close()
                        break
                        
            except EOFError:
                print("Connection closed")
                pass
            except Exception as e:
                print(f"Error handling connection: {e}")
                pass
                
    except KeyboardInterrupt:
        print("Server stopping...")
    finally:
        listener.close()

