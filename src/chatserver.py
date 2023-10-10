import socket 

PORT = 50000
BUFSIZE = 4096

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('', PORT))

client_list = []
while True:
    data, addr = server.recvfrom(BUFSIZE)
    if addr not in client_list:
        client_list.append(addr)
    if data.decode('UTF-8') == 'q':
        client_list.remove(addr)
    else:
        msg = str(addr) + '> ' + data.decode('UTF-8')
        print(msg)
        for client in client_list:
            server.sendto(msg.encode('UTF-8'), client)