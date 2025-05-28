from huggingface_hub import snapshot_download
if __name__ == "__main__":
	snapshot_download(
		repo_id="HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit", 
		local_dir="/home/chwenjun225/.llama/checkpoints/siglip-so400m-14-980-flash-attn2-navit", 
		revision="main"
	)


# HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
# google/siglip-so400m-patch14-384