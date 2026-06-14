#!/bin/bash

# 停止并删除旧容器
docker stop temp-proxy-8000 2>/dev/null
docker rm temp-proxy-8000 2>/dev/null

# 创建临时 nginx 配置文件
cat > /tmp/nginx_stream_8000.conf << 'EOF'
user root;
worker_processes auto;

events {
    worker_connections 1024;
}

stream {
    log_format proxy '$remote_addr [$time_local] '
                     '$protocol $status $bytes_sent $bytes_received '
                     '$session_time "$upstream_addr"';

    upstream backend {
        server 172.18.133.56:8000;
    }

    server {
        listen 8000;
        proxy_pass backend;
        proxy_connect_timeout 10s;
        proxy_timeout 300s;
    }
}
EOF

# 启动容器
docker run -d \
  --name temp-proxy-8000 \
  --restart always \
  --network host \
  -v /tmp/nginx_stream_8000.conf:/etc/nginx/nginx.conf:ro \
  172.18.0.140:28280/public/nginx:1.27-amd64

# 检查状态
if [ $? -eq 0 ]; then
    echo "容器启动成功"
    echo "端口转发规则: 本机 8000 -> 172.18.133.56:8000"
    echo "查看日志: docker logs temp-proxy-8000"
else
    echo "容器启动失败"
fi
