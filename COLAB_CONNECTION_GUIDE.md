# Connecting Local VS Code to Google Colab

This guide explains how to connect your local VS Code to a Google Colab instance, allowing you to use your local extensions, themes, and settings while utilizing Colab's GPU resources.

## Prerequisites

1.  **VS Code** installed on your Mac.
2.  **Remote - SSH** extension installed in VS Code.
3.  **Cloudflared** installed on your Mac (required for the tunneling method used below).
    *   *Brew install command:* `brew install cloudflared`

## Step 1: Prepare Google Colab

1.  Open your Google Colab notebook.
2.  Create a new code cell and paste the content from `colab_launch.py` (included in this repo), or copy it from here:

    ```python
    !pip install colab_ssh --upgrade
    from colab_ssh import launch_ssh_cloudflared
    launch_ssh_cloudflared(password="antigravity_colab")
    ```

3.  **Run the cell.**
4.  Wait for the output. It will print a configuration block that looks like this:

    ```text
    Hostname: ...
    Username: root
    ...
    SSH Connect Command: ...
    ```

    **IMPORTANT:** It will give you a VS Code config snippet. Keep this tab open.

## Step 2: Configure Local VS Code

1.  Open VS Code.
2.  Press `Cmd + Shift + P` to open the Command Palette.
3.  Type `Remote-SSH: Open Configuration File...` and select it.
4.  Choose your primary config file (usually `/Users/yourname/.ssh/config`).
5.  **Copy the config snippet** provided by the Colab output (Step 1) and paste it at the bottom of this file. It typically looks like:

    ```ssh
    Host google_colab_ssh
        HostName try.cloudflare.com
        User root
        IdentityFile ...
        ProxyCommand ...
    ```

    *If Colab gives you a specific HostName like `google_colab_ssh`, use that.*

## Step 3: Connect!

1.  In VS Code, press `Cmd + Shift + P`.
2.  Type `Remote-SSH: Connect to Host...`.
3.  Select the host alias you just added (e.g., `google_colab_ssh`).
4.  When prompted for the platform, select **Linux**.
5.  When prompted for the password, enter: `antigravity_colab` (or whatever you changed it to).

## Success!

You are now connected! The file explorer on the left will show the files in the Colab VM.
- Your code is running on Google's GPU.
- You can access `/content/` to see your data.
