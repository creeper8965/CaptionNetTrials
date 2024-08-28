import os

# Base URL for the files
base_url = "https://huggingface.co/datasets/lmms-lab/COCO-Caption/resolve/main/data/"

#Save Path
save_path = "COCO-2014/"

# File pattern
file_pattern = "val-{:05d}-of-00013.parquet"

# Loop through the range 0-12
for i in range(13):
    # Format the filename
    filename = file_pattern.format(i)
    
    # Complete download URL
    download_url = f"{base_url}{filename}?download=true"
    
    # Command to run
    command = f"wget {download_url} -O {save_path}{filename}"
    
    # Execute the command
    os.system(command)

    print(f"Downloaded {filename}")
