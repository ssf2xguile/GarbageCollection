import socket
import sys

HOST = 'localhost'
PORT = 50000
BUFSIZE = 4096

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

fin = open('data/data.txt', 'rt')

try:
    client.connect((HOST, PORT))
except socket.gaierror:
    print('接続に失敗しました')
    sys.exit()

# サーバへのメッセージ送信
msg = fin.read()
client.sendall(msg.encode('UTF-8'))
client.close()