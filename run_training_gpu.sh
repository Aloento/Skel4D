#!/bin/bash
#SBATCH -A c_mvcondif
#SBATCH --time=40:20:00
#SBATCH --job-name=fothi_skel_gpu
#SBATCH --output=slurm_out/%j_skel_gpu.out
#SBATCH --error=slurm_out/%j_skel_gpu.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fa2@inf.elte.hu


module load singularity
export HF_HOME="/hf_home/"

singularity exec --nv \
  --env WANDB_API_KEY="cac4802235eab1dcc55870599944f661d7697032" \
  --env WANDB_MODE=online \
  --env WANDB_DIR=/wandb_data \
  --env SSL_CERT_FILE=/venvs/venv_instantskel/lib/python3.10/site-packages/certifi/cacert.pem \
  --env REQUESTS_CA_BUNDLE=/venvs/venv_instantskel/lib/python3.10/site-packages/certifi/cacert.pem \
  -B /etc/ssl/certs:/etc/ssl/certs:ro \
    -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /project/c_mvcondif/wandb/:/wandb/ \
    -B /scratch/c_mvcondif/wandb_data/:/wandb_data/ \
    -B /project/c_mvcondif/hf_home/:/hf_home/ \
    -B /project/c_mvcondif/outputs/:/outputs/ \
      /project/c_mvcondif/containers/instantskel.sif \
      python train.py --base configs/zero123plus-finetune-sk.yaml --gpus 0,1,2,3 --num_nodes 1 --name "fothi_skel_gpu1_$(date +%m%d_%H%M)" --seed 42