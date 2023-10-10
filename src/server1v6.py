import socket
import datetime

PORT = 50000

# ソケットを作成してバインドする
server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
server.bind(('', PORT))

# 接続の待ち受け
server.listen()

# クライアントへの対応処理
while True:
    client, addr = server.accept()
    msg = str(datetime.datetime.now())
    client.sendall(msg.encode('UTF-8'))
    print(msg, "接続要求あり")
    print("クライアント情報:", client)
    print("アドレス:", addr)
    client.close()