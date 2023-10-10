import socket
import sys

PORT = 50000
BUFSIZE = 4096

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = input('接続するホスト名を入力してください: ')
try:
    client.connect((host, PORT))
except socket.gaierror:
    print('接続に失敗しました')
    sys.exit()

# サーバへのメッセージ送信
msg = input('送信メッセージを入力してください: ')
client.sendall(msg.encode('UTF-8'))

# サーバからのメッセージ受信
data = client.recv(BUFSIZE)
print("サーバからのメッセージ: ")
print(data.decode('UTF-8'))

client.close()