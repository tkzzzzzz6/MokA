uv sync --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple

.\scripts\pretrain\prepare_audiocaps_500_100_windows.ps1 -CookiesFromBrowser chrome

wget.exe -c https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main/llava_image.zip

apt update
apt install -y aria2

export HF_ENDPOINT=https://hf-mirror.com

aria2c -x 16 -s 16 -c \
https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main/llava_image.zip

wget -c https://hf-mirror.com/datasets/LanguageBind/Video-LLaVA/resolve/main/llava_image.zip