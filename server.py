import socket

def receive_file(conn, filename):
    # First receive the size of the file
    file_size = int(conn.recv(1024).decode())
    conn.sendall(b'ACK')  # Send acknowledgement

    # Now receive the file itself
    with open(filename, 'wb') as file:
        bytes_received = 0
        while bytes_received < file_size:
            data = conn.recv(1024)
            if not data:
                break
            file.write(data)
            bytes_received += len(data)

def send_file(conn, filename):
    with open(filename, 'rb') as file:
        data = file.read()
        conn.sendall(data)  # Send the file data back

def main():
    host = 'localhost'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Server listening on port", port)

    conn, addr = server_socket.accept()
    print('Connected by', addr)
    
    filename = 'received_image.png'
    receive_file(conn, filename)
    send_file(conn, filename)
    
    conn.close()
    server_socket.close()

if __name__ == '__main__':
    main()

