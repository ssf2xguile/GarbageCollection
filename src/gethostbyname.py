import socket

while True:
    try:
        hostname = input("ホスト名を入力してください(qで終了): ")
        if hostname == "q":
            break
        print(socket.gethostbyname(hostname))
    except socket.gaierror:
        print("ホスト名が見つかりませんでした")