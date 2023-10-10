import socket
import datetime
import threading

PORT = 50000
BUFSIZE = 4096

def client_hander(client, client_no, msg):
    """クライアントとの接続処理スレッド"""
    deta = client.recv(BUFSIZE)
    print(f'クライアント{client_no}: {deta.decode("UTF-8")}')
    client.sendall(msg.encode('UTF-8'))
    client.close()


# ソケットを作成してバインドする
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('', PORT))

# 接続の待ち受け
server.listen()

# クライアントの受付番号の初期化
client_no = 0

# クライアントへの対応処理
while True:
    client, addr = server.accept()
    client_no += 1
    msg = str(datetime.datetime.now())
    print(msg, "接続要求あり")
    print("クライアント情報:", client)
    print("アドレス:", addr)

    p = threading.Thread(target=client_hander, args=(client, client_no, msg))
    p.start()
