# !pip install colab_ssh --upgrade

from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared

# Set your password here
PASSWORD = "antigravity_colab" 

print(f"Starting SSH Tunnel... Password: {PASSWORD}")

# Launch the SSH tunnel using Cloudflared (no ngrok token needed)
launch_ssh_cloudflared(password=PASSWORD)

# Optional: Link your Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
