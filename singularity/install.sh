singularity shell --nv  -B /project/c_mvcondif/venvs/:/venvs/ \
    -B /scratch/c_mvcondif/:/objaverse/ \
    -B /project/c_mvcondif/models/:/models/ \
    -B /project/c_mvcondif/logs/:/logs/ \
    -B /project/c_mvcondif/outputs/:/outputs/ /project/c_mvcondif/containers/instantskel.sif


python3 -m virtualenv /venvs/venv_instantskel
source /venvs/venv_instantskel/bin/activate
pip install -U pip

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7



