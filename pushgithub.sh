#!/bin/bash

# 检查ps命令返回的字符串中是否包含"Qv2ray"和"AppImage"
if ps -aux | grep Qv2ray | grep AppImage > /dev/null; then
    # 如果包含，则设置环境变量
    export ALL_PROXY="socks5://127.0.0.1:1089"
    export all_proxy="socks5://127.0.0.1:1089"
    echo "已设置Git代理"
fi

git add --all
git commit -m "update"
git push -u origin main