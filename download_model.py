from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/biogpt",
    local_dir="/home/librad.laureateinstitute.org/mferguson/models/Microsoft-BioGPT",
    local_dir_use_symlinks=False
)
