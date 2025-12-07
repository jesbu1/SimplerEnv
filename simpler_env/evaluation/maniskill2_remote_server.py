import numpy as np
from PIL import Image
import time
import multiprocessing
import socket
import pickle
import struct
from simpler_env.policies.pi0.geometry import quat2mat, mat2euler
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def send_msg(sock, msg):
    """Send a message with length prefix over socket."""
    msg_bytes = pickle.dumps(msg)
    msg_len = struct.pack('>I', len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)


def recv_msg(sock):
    """Receive a length-prefixed message from socket."""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return pickle.loads(recvall(sock, msglen))


def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF is hit."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def preprocess_widowx_proprio(eef_pos) -> np.array:
    """convert ee rotation to the frame of top-down"""
    # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]) 
    proprio = eef_pos
    rm_bridge = quat2mat(proprio[3:7])
    rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7] # from simpler, 0 for close, 1 for open
    raw_proprio = np.concatenate(
        [
            proprio[:3],
            rpy_bridge_converted,
            [gripper_openness],
        ]
    )
    return raw_proprio

def handle_client(client_sock, args):
    """
    Handle a single client connection in a separate process.
    Creates its own environment instance.
    """
    env = None
    task_description = None
    try:
        print(f"[Process {multiprocessing.current_process().pid}] Client connected, building env...")
        
        # Build environment for this client
        control_mode = get_robot_control_mode(args.robot, args.policy_model)
        additional_env_build_kwargs = args.additional_env_build_kwargs or {}
        
        kwargs = dict(
            obs_mode="rgbd",
            robot=args.robot,
            sim_freq=args.sim_freq,
            control_mode=control_mode,
            control_freq=args.control_freq,
            max_episode_steps=args.max_episode_steps,
            scene_name=args.scene_name,
            camera_cfgs={"add_segmentation": False},
            rgb_overlay_path=args.rgb_overlay_path,
        )
        
        if args.enable_raytracing:
            ray_tracing_dict = {"shader_dir": "rt"}
            ray_tracing_dict.update(additional_env_build_kwargs)
            additional_env_build_kwargs = ray_tracing_dict
            
        env = build_maniskill2_env(
            args.env_name,
            **additional_env_build_kwargs,
            **kwargs,
        )
        
        print(f"[Process {multiprocessing.current_process().pid}] Env built: {args.env_name}")
        
        while True:
            cmd = recv_msg(client_sock)
            if cmd is None:
                print(f"[Process {multiprocessing.current_process().pid}] Client disconnected")
                break
            
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
                eef_pos = obs['agent']['eef_pos']
                state = preprocess_widowx_proprio(eef_pos)
                task_description = env.get_language_instruction()
                
                image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)

                image = Image.fromarray(image)
                image = image.resize((224, 224))
                image = np.array(image)
                
                send_msg(client_sock, {
                    'state': state,
                    'observation.images.image_0': image,
                    'prompt': task_description  # Use 'prompt' key to match RemoteEnv expectations
                })
                
            elif cmd['type'] == 'STEP':
                action = cmd['action']
                action = np.array(action, dtype=np.float32)
                obs, reward, done, truncated, info = env.step(action)
                eef_pos = obs['agent']['eef_pos']
                state = preprocess_widowx_proprio(eef_pos)
                
                image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
                
                image = Image.fromarray(image)
                image = image.resize((224, 224))
                image = np.array(image)
                
                # Check for success in info
                success = info.get('success', False)
                
                send_msg(client_sock, {
                    'state': state,
                    'observation.images.image_0': image,
                    'reward': reward,
                    'done': done,
                    'prompt': task_description,
                    'truncated': truncated,
                    'success': success,
                    'info': info
                })
                
            elif cmd['type'] == 'CLOSE':
                env.close()
                break
                
    except Exception as e:
        print(f"[Process {multiprocessing.current_process().pid}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass
        try:
            client_sock.close()
        except:
            pass


def run_maniskill2_remote_server(args):
    """
    Run a remote server that spawns a new process with its own environment for each client.
    Uses raw TCP sockets for compatibility with RemoteEnv.
    """
    # Setup server socket
    host = '0.0.0.0'
    port = 6000
    
    if args.server_ip:
        parts = args.server_ip.split(':')
        host = parts[0]
        if len(parts) > 1:
            port = int(parts[1])
            
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
        
    print(f"Starting remote server on {host}:{port} (Multi-Process Mode, TCP)...")
    
    try:
        while True:
            print("Waiting for connection...")
            client_sock, client_addr = server_sock.accept()
            print(f"Connection accepted from {client_addr}, spawning process...")
            
            # Spawn a new process for this client
            p = multiprocessing.Process(target=handle_client, args=(client_sock, args))
            p.daemon = True
            p.start()
            
            # Close our handle to the connection in the parent process
            client_sock.close()
            
    except KeyboardInterrupt:
        print("Server stopping...")
    finally:
        server_sock.close()
