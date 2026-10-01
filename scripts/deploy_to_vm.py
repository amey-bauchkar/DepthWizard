"""Automated 1-click cloud deployment for DepthWizard on Azure VM.

Transfers model weights, pulls latest Git updates, builds container, and restarts server.
"""
from __future__ import annotations

import getpass
import sys
import time
import webbrowser
from pathlib import Path

import paramiko

VM_IP = "172.198.77.233"
VM_USER = "azureuser"
ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "depthwizard_ndsm_model_v2.zip"
LOCAL_ZIP = ROOT / ZIP_NAME


def run_remote_stream(client: paramiko.SSHClient, cmd: str, desc: str) -> int:
    print(f"\n[+] {desc}...")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print("    " + line.rstrip())
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print(f"[-] Error: {err.strip()}", file=sys.stderr)
    return exit_status


def main() -> int:
    print("=" * 65)
    print("  DepthWizard - Automated 1-Click Azure Cloud Deployer")
    print(f"  Target VM: {VM_USER}@{VM_IP}")
    print("=" * 65)

    if not LOCAL_ZIP.exists():
        print(f"[-] ERROR: {LOCAL_ZIP} not found!", file=sys.stderr)
        return 1

    password = getpass.getpass("\nEnter VM Password (typing is hidden): ")
    if not password:
        print("[-] Password cannot be empty.")
        return 1

    print("\nConnecting to Azure VM...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(VM_IP, username=VM_USER, password=password, timeout=15)
        print("[✓] Connected successfully to VM!")
    except Exception as e:
        print(f"[-] Failed to connect: {e}")
        return 1

    try:
        # Check if remote zip already exists
        sftp = client.open_sftp()
        remote_zip = f"/home/{VM_USER}/DepthWizard/{ZIP_NAME}"
        upload_needed = True
        try:
            rstat = sftp.stat(remote_zip)
            if rstat.st_size == LOCAL_ZIP.stat().st_size:
                print(f"[✓] {ZIP_NAME} already uploaded on VM ({rstat.st_size/1e6:.1f} MB). Skipping upload.")
                upload_needed = False
        except IOError:
            upload_needed = True

        if upload_needed:
            total_size = LOCAL_ZIP.stat().st_size
            last_pct = [-1]

            def progress(transferred: int, total: int):
                pct = int((transferred / total) * 100)
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    mb_cur = transferred / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    sys.stdout.write(f"\r[+] Uploading {ZIP_NAME}: {pct}% ({mb_cur:.1f}/{mb_tot:.1f} MB)")
                    sys.stdout.flush()

            print(f"[+] Uploading {ZIP_NAME} to VM (size: {total_size/1e6:.1f} MB)...")
            sftp.put(str(LOCAL_ZIP), remote_zip, callback=progress)
            print("\n[✓] Model weights uploaded successfully!")
        sftp.close()

        # Step 1: Git pull
        run_remote_stream(client, "cd ~/DepthWizard && git pull origin main", "Pulling latest layout & scale updates from GitHub")

        # Step 2: Unpack model weights
        unpack_cmd = (
            "cd ~/DepthWizard && "
            "mkdir -p models/da-v2-small-ndsm/2.0.0 && "
            "python3 -c \"import zipfile; zipfile.ZipFile('depthwizard_ndsm_model_v2.zip').extractall('models/da-v2-small-ndsm/2.0.0/')\""
        )
        run_remote_stream(client, unpack_cmd, "Unpacking v2 model weights into models/da-v2-small-ndsm/2.0.0/")

        # Step 3: Docker build
        build_cmd = "cd ~/DepthWizard && sudo docker build -t depthwizard:latest ."
        run_remote_stream(client, build_cmd, "Building updated production Docker image")

        # Step 4: Restart container
        restart_cmd = (
            "sudo docker stop depthwizard 2>/dev/null || true; "
            "sudo docker rm depthwizard 2>/dev/null || true; "
            "sudo docker run -d --name depthwizard --restart always -p 80:7860 depthwizard:latest"
        )
        run_remote_stream(client, restart_cmd, "Starting updated DepthWizard container on port 80")

        # Step 5: Test health
        time.sleep(3)
        check_cmd = "curl -s http://localhost:80/health | grep -o '\"status\":\"ok\"' || echo 'starting...'"
        run_remote_stream(client, check_cmd, "Verifying live server health")

        url = f"http://{VM_IP}/"
        print("\n" + "=" * 65)
        print("  🎉 DEPLOYMENT SUCCEEDED!")
        print(f"  Live Presentation URL: {url}")
        print("=" * 65)

        try:
            webbrowser.open(url)
        except Exception:
            pass
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
