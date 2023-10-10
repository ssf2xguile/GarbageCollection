import socket
import datetime

PORT = 50000
BUFSIZE = 4096

# ソケットを作成してバインドする
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('', PORT))

# クライアントへの対応処理
while True:
    data, addr = server.recvfrom(BUFSIZE)
    msg = str(datetime.datetime.now())
    server.sendto(msg.encode('UTF-8'), addr)
    print(msg, "接続要求あり")
    print("アドレス:", addr)