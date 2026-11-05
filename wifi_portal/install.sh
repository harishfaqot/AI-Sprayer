#!/bin/sh
# Run on the Pi:  sudo sh install.sh
set -e
cd "$(dirname "$0")"
install -d /opt/wifi-portal /var/lib/wifi-portal
install -m 755 wifi_portal.py /opt/wifi-portal/wifi_portal.py
install -m 644 wifi-portal.service /etc/systemd/system/wifi-portal.service
[ -f /etc/wifi-portal.json ] || echo '{}' > /etc/wifi-portal.json
install -m 644 sprayer-port80.service /etc/systemd/system/sprayer-port80.service
systemctl daemon-reload
systemctl enable wifi-portal.service
systemctl enable sprayer-port80.service
systemctl restart sprayer-port80.service
systemctl restart wifi-portal.service
echo "installed. logs: journalctl -u wifi-portal -f"
