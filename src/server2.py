import socket
import datetime

PORT = 50000
BUFSIZE = 4096

# ソケットを作成してバインドする
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('', PORT))

# 接続の待ち受け
server.listen()

# クライアントへの対応処理
while True:
    client, addr = server.accept()
    msg = str(datetime.datetime.now())
    print(msg, "接続要求あり")
    print("クライアント情報:", client)
    print("アドレス:", addr)

    # クライアントからのメッセージ受信
    data = client.recv(BUFSIZE)
    print(data.decode('UTF-8'))

    # クライアントへのメッセージ送信
    client.sendall(msg.encode('UTF-8'))
    client.close()