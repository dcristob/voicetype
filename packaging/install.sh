#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing voicetype (user)"
uv tool install --force . || pipx install --force .

echo "==> Installing systemd user units"
mkdir -p "$HOME/.config/systemd/user"
cp systemd/voicetype.service "$HOME/.config/systemd/user/voicetype.service"
cp systemd/ydotoold.service "$HOME/.config/systemd/user/ydotoold.service"
systemctl --user daemon-reload

echo "==> Checking for ydotool"
if ! command -v ydotoold >/dev/null; then
  echo "ydotool is not installed. Install it with your package manager, e.g.:"
  echo "    sudo pacman -S ydotool      # Arch / CachyOS"
  echo "    sudo apt install ydotool    # Debian / Ubuntu"
  echo "    sudo dnf install ydotool    # Fedora"
fi

echo "==> Configuring uinput permission for ydotool (needs sudo)"
sudo cp packaging/70-uinput.rules /etc/udev/rules.d/70-uinput.rules
sudo groupadd -f input
sudo usermod -aG input "$USER" || true
sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput || true

cat <<'EOF'

Next steps:
  1. Start the input backend:  systemctl --user enable --now ydotoold
  2. Enable voicetype:         systemctl --user enable --now voicetype
  3. Bind a hotkey in your DE keyboard settings to:  voicetype toggle
  4. GNOME only: enable the AppIndicator/Ayatana extension for the tray icon.

The udev rule grants the active user access to /dev/uinput immediately (via
uaccess), so no logout is required. If `ydotool type` reports a permission
error, run:  sudo udevadm trigger /dev/uinput
EOF
