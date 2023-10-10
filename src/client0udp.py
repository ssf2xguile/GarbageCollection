import socket

HOST = 'localhost'
PORT = 50000
BUFSIZE = 4096

# ソケットを作成してサーバに接続する
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# サーバにメッセージを送信する
client.sendto(b'Hello', (HOST, PORT))

# サーバからのメッセージを受信して表示する
data = client.recv(BUFSIZE)
print(data.decode('UTF-8'))

# コネクションの終了
client.close()