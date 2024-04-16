import socket
import os

def send_file(sock, filename):
    # First send the size of the file
    file_size = os.path.getsize(filename)
    sock.sendall(str(file_size).encode())
    sock.recv(1024)  # Wait for server acknowledgement

    # Now send the file data
    with open(filename, 'rb') as file:
        sock.sendall(file.read())

def receive_file(sock, filename):
    with open(filename, 'wb') as file:
        data = sock.recv(1024)
        while data:
            file.write(data)
            data = sock.recv(1024)

def main():
    host = 'localhost'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    filename = 'your_img.jpg'  # This should be the path to your image
    send_file(client_socket, filename)
    
    received_filename = 'received_back_image.png'
    receive_file(client_socket, received_filename)
    print("Received image back from server.")

    client_socket.close()
    print("Connection closed.")

if __name__ == '__main__':
    main()

