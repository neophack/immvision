#!/bin/bash

# 首先，我们需要使用root权限运行这个脚本
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# 创建udev规则文件
echo 'ACTION=="add", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="0c2e", ATTRS{idProduct}=="0901", KERNEL=="hidraw*", GOTO="create_symlink"
GOTO="end_label"
LABEL="create_symlink"
TEST!="/dev/barscanner", MODE="0666", SYMLINK+="barscanner", GOTO="end_label"
LABEL="end_label"' > /etc/udev/rules.d/99-usb.rules

# 重载udev规则，使新规则生效
udevadm control --reload-rules
udevadm trigger
