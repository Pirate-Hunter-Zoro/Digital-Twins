from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen1.5-0.5B-Chat",
    local_dir="/home/librad.laureateinstitute.org/mferguson/models/Qwen1.5-0.5B-Chat",
    local_dir_use_symlinks=False
)
