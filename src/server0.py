import socket

PORT = 50000

# ソケットを作成してバインドする
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('', PORT))

# 接続の待ち受け
server.listen()

# クライアントへの対応処理
client, addr = server.accept()
client.sendall(b'fuck off dude')
client.close()
server.close()