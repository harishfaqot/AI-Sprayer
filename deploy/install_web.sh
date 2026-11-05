#!/bin/sh
# Jalankan di Orange Pi:  sudo sh install_web.sh
# Pastikan folder /home/orangepi/sprayer-weeder sudah berisi main_web.py, pid_control.py,
# runs/segment/train-9/weights/best.pt dan video/real1.mp4
set -e
cd "$(dirname "$0")"
install -m 644 sprayer-web.service /etc/systemd/system/sprayer-web.service
systemctl daemon-reload
systemctl enable sprayer-web.service
systemctl restart sprayer-web.service
echo "terpasang. log: journalctl -u sprayer-web -f"
