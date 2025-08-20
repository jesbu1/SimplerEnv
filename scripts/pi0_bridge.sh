server_ip=${1:-"0.0.0.0:8001"}
logging_dir=${2:-"./results"}
gpu_id=${3:-0}

ckpt_path="pi0_regular_lora_29999"
action_ensemble_temp=-0.8
policy_model=pi0
wandb_project="p-masked-vla"
wandb_entity="clvr"
wandb_run_group="SIMPLER-EVAL-${ckpt_path}"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028



CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --action-ensemble-temp ${action_ensemble_temp} --logging-dir ${logging_dir} --server-ip ${server_ip} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --ckpt-path ${ckpt_path} \
  --wandb-project ${wandb_project} \
  --wandb-entity ${wandb_entity} \
  --use-wandb \
  --wandb-run-group ${wandb_run_group} \
  --wandb-run-name ${wandb_run_group}_PutCarrotOnPlateInScene-v0 \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --action-ensemble-temp ${action_ensemble_temp} --logging-dir ${logging_dir} --server-ip ${server_ip} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --ckpt-path ${ckpt_path} \
  --wandb-project ${wandb_project} \
  --wandb-entity ${wandb_entity} \
  --use-wandb \
  --wandb-run-group ${wandb_run_group} \
  --wandb-run-name ${wandb_run_group}_StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --action-ensemble-temp ${action_ensemble_temp} --logging-dir ${logging_dir} --server-ip ${server_ip} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --ckpt-path ${ckpt_path} \
  --wandb-project ${wandb_project} \
  --wandb-entity ${wandb_entity} \
  --use-wandb \
  --wandb-run-group ${wandb_run_group} \
  --wandb-run-name ${wandb_run_group}_PutSpoonOnTableClothInScene-v0 \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;


scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --action-ensemble-temp ${action_ensemble_temp} --logging-dir ${logging_dir} --server-ip ${server_ip} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --ckpt-path ${ckpt_path} \
  --wandb-project ${wandb_project} \
  --wandb-entity ${wandb_entity} \
  --use-wandb \
  --wandb-run-group ${wandb_run_group} \
  --wandb-run-name ${wandb_run_group}_PutEggplantInBasketScene-v0 \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;
