# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)

print(model)
