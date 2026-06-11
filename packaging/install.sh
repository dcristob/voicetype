#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing voicetype (user)"
uv tool install --force . || pipx install --force .

echo "==> Installing systemd user unit"
mkdir -p "$HOME/.config/systemd/user"
cp systemd/voicetype.service "$HOME/.config/systemd/user/voicetype.service"
systemctl --user daemon-reload

echo "==> Configuring uinput permission for ydotool (needs sudo)"
sudo cp packaging/70-uinput.rules /etc/udev/rules.d/70-uinput.rules
sudo groupadd -f input
sudo usermod -aG input "$USER"
sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput || true

echo "==> ydotool"
if ! command -v ydotool >/dev/null; then
  echo "Please install ydotool with your package manager (e.g. 'sudo pacman -S ydotool')."
fi

cat <<'EOF'

Next steps:
  1. Log out/in (so your 'input' group membership applies).
  2. Start ydotoold:   systemctl --user enable --now ydotool   (or run 'ydotoold &')
  3. Enable voicetype: systemctl --user enable --now voicetype
  4. Bind a hotkey in your DE keyboard settings to:  voicetype toggle
  5. GNOME only: enable the AppIndicator/Ayatana extension for the tray icon.
EOF
