export PYTHONPATH=$PYTHONPATH:$HOME/anaconda3/envs/onepose/lib/python3.7/site-packages/DeepLM/build
export TORCH_USE_RTLD_GLOBAL=YES

python cv2_real_time_improved.py +experiment=cv2_GATsSPG.yaml
