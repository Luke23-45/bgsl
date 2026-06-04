# 3. Public Dataset Acquisition
import os
from huggingface_hub import snapshot_download

repo_id = "hellxhell/sepsis-clinical-28"
local_dir = "sepsis_clinical_28"

print(f"🚀 Downloading PUBLIC dataset from {repo_id}...")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.lmdb", "*.json"]
)

print("\n✅ Download Complete!")

