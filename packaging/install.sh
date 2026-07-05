#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing voicetype (user)"
# Install the [cuda] extra (CTranslate2 CUDA runtime libs) when an NVIDIA GPU is
# present, so the systemd service gets GPU acceleration too. Falls back to CPU
# at runtime if the GPU can't be used.
spec="."
if command -v nvidia-smi >/dev/null 2>&1 || lspci 2>/dev/null | grep -qi nvidia; then
  echo "    NVIDIA GPU detected -> installing with [cuda] extra"
  spec=".[cuda]"
fi
uv tool install --force "$spec" || pipx install --force "$spec"

echo "==> Installing systemd user units"
mkdir -p "$HOME/.config/systemd/user"
cp systemd/voicetype.service "$HOME/.config/systemd/user/voicetype.service"
cp systemd/ydotoold.service "$HOME/.config/systemd/user/ydotoold.service"
systemctl --user daemon-reload

# Some distros (e.g. the Arch `ydotool` package) ship their own enabled
# `ydotool.service` that runs a bare `ydotoold` on the SAME default socket path
# our unit uses. Two daemons then fight over the socket and both keep failing
# ("Another ydotoold is running with the same socket"). Disable the distro one
# so only our unit (explicit socket path + 0660 perms) owns the socket.
if systemctl --user list-unit-files ydotool.service >/dev/null 2>&1; then
  echo "    Disabling conflicting distro ydotool.service"
  systemctl --user disable --now ydotool.service 2>/dev/null || true
fi

echo "==> Checking for ydotool"
if ! command -v ydotoold >/dev/null; then
  echo "ydotool is not installed. Install it with your package manager, e.g.:"
  echo "    sudo pacman -S ydotool      # Arch / CachyOS"
  echo "    sudo apt install ydotool    # Debian / Ubuntu"
  echo "    sudo dnf install ydotool    # Fedora"
fi

echo "==> Checking for wl-clipboard (clipboard injection of Unicode text)"
if ! command -v wl-copy >/dev/null; then
  echo "wl-clipboard is not installed. Install it with your package manager, e.g.:"
  echo "    sudo pacman -S wl-clipboard      # Arch / CachyOS"
  echo "    sudo apt install wl-clipboard    # Debian / Ubuntu"
  echo "    sudo dnf install wl-clipboard    # Fedora"
fi

echo "==> Configuring uinput permission for ydotool (needs sudo)"
sudo cp packaging/70-uinput.rules /etc/udev/rules.d/70-uinput.rules
sudo groupadd -f input
sudo usermod -aG input "$USER" || true
sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput || true

echo "==> Enabling + starting services (survive reboot via default.target)"
# reenable rewrites the WantedBy symlink, so upgrades from an older unit that
# targeted graphical-session.target get re-anchored under default.target.wants.
systemctl --user reenable ydotoold.service voicetype.service
systemctl --user restart ydotoold.service voicetype.service

cat <<'EOF'

Installed and started. The services are enabled under default.target, so they
start automatically on every login/reboot.

Next steps:
  1. Bind a hotkey in your DE keyboard settings to:  voicetype toggle
  2. GNOME only: enable the AppIndicator/Ayatana extension for the tray icon.

Check status any time with:
  systemctl --user status voicetype ydotoold
  voicetype status

The udev rule grants the active user access to /dev/uinput immediately (via
uaccess), so no logout is required. If `ydotool type` reports a permission
error, run:  sudo udevadm trigger /dev/uinput
EOF
