# WSL Setup Guide for KALA

KALA runs on Linux. If you're on Windows, you'll need to use **Windows Subsystem for Linux (WSL)** to run it. This guide walks through installing WSL with Ubuntu and troubleshooting common problems.

## Requirements

- **Windows 10** version 2004+ (Build 19041+) or **Windows 11**
- Administrator access
- Virtualization enabled in BIOS/UEFI (usually labeled "Intel VT-x" or "AMD-V")

## Installation

### Step 1: Install WSL and Ubuntu

Open **PowerShell as Administrator** (right-click Start > "Terminal (Admin)" or search "PowerShell" > "Run as administrator") and run:

```powershell
wsl --install -d Ubuntu
```

This enables the required Windows features (Virtual Machine Platform, Windows Subsystem for Linux) and installs Ubuntu in a single command.

**Restart your computer when prompted.**

### Step 2: Complete Ubuntu Setup

After restarting, Ubuntu should launch automatically and ask you to create a username and password. If it does not launch automatically, see the troubleshooting section below.

### Step 3: Verify the Installation

In PowerShell, run:

```powershell
wsl --list --verbose
```

You should see Ubuntu listed with a STATE of "Running" or "Stopped" and VERSION 2.

## Troubleshooting: Ubuntu Not Opening / Not Listed as an App

This is a common issue. Work through these steps in order.

### 1. Check if WSL Itself is Installed

Open PowerShell as Administrator and run:

```powershell
wsl --status
```

If this command is not recognized, WSL is not installed. Run `wsl --install` and restart.

### 2. Check What Distributions Are Installed

```powershell
wsl --list --verbose
```

- **If Ubuntu appears in the list** but won't open as an app, you can launch it directly from PowerShell:
  ```powershell
  wsl -d Ubuntu
  ```
  Or simply type `wsl` if Ubuntu is your only distribution.

- **If nothing is listed**, Ubuntu did not fully install. Re-install it:
  ```powershell
  wsl --install -d Ubuntu
  ```
  Then restart your computer.

### 3. Ensure Required Windows Features Are Enabled

Open PowerShell as Administrator and run:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**Restart your computer** after running these commands, then try installing Ubuntu again:

```powershell
wsl --install -d Ubuntu
```

### 4. Update WSL

An outdated WSL installation can cause silent failures:

```powershell
wsl --update
```

Then restart WSL:

```powershell
wsl --shutdown
wsl --install -d Ubuntu
```

### 5. Set WSL 2 as the Default Version

WSL 2 is required for Docker and GPU passthrough (both needed by KALA). Make sure it's the default:

```powershell
wsl --set-default-version 2
```

If you already have Ubuntu installed under WSL 1, upgrade it:

```powershell
wsl --set-version Ubuntu 2
```

### 6. Check Virtualization

WSL 2 requires hardware virtualization. To verify it's enabled:

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Go to the **Performance** tab
3. Look for **Virtualization: Enabled**

If virtualization is disabled, you need to enable it in your BIOS/UEFI settings. The exact steps vary by motherboard manufacturer — search for your model + "enable virtualization" for specific instructions.

### 7. Re-register Ubuntu (Last Resort)

If Ubuntu is listed but completely broken, unregister and reinstall:

```powershell
wsl --unregister Ubuntu
wsl --install -d Ubuntu
```

**Warning**: This deletes all data inside that Ubuntu installation.

## After Ubuntu is Working

Once you can open an Ubuntu terminal, set up KALA:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and pip
sudo apt install -y python3 python3-pip python3-venv git

# (Optional) Install CUDA toolkit for GPU support
# See: https://developer.nvidia.com/cuda-downloads (select WSL-Ubuntu)

# Clone and set up KALA
git clone https://github.com/hewcarroll/KALA.git
cd KALA

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## GPU Support in WSL

KALA benefits significantly from GPU acceleration. To use your NVIDIA GPU inside WSL:

1. Install the latest **NVIDIA GPU driver** on Windows (not inside WSL) from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. WSL 2 automatically provides GPU access — you do not need to install a separate Linux driver
3. Inside Ubuntu, install the CUDA toolkit:
   ```bash
   # Check that your GPU is visible
   nvidia-smi

   # Install CUDA toolkit
   sudo apt install -y nvidia-cuda-toolkit
   ```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `WslRegisterDistribution failed with error: 0x80370102` | Virtualization not enabled | Enable VT-x/AMD-V in BIOS |
| `WslRegisterDistribution failed with error: 0x80370114` | Hyper-V not available | Enable "Virtual Machine Platform" in Windows Features |
| `WslRegisterDistribution failed with error: 0x8007019e` | WSL feature not enabled | Run the `dism.exe` commands in step 3 above |
| `wsl: command not found` | Windows too old for WSL | Update to Windows 10 2004+ or Windows 11 |
| `The virtual machine could not be started` | Not enough memory | Close other applications or increase WSL memory limit in `.wslconfig` |

## Configuring WSL Resources

KALA requires significant RAM. Create or edit `%USERPROFILE%\.wslconfig` on Windows:

```ini
[wsl2]
memory=24GB
swap=8GB
processors=8
```

Restart WSL for changes to take effect:

```powershell
wsl --shutdown
```

## Further Reading

- [Microsoft WSL Documentation](https://learn.microsoft.com/en-us/windows/wsl/)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [KALA Development Plan](DEVELOPMENT_PLAN.md)
