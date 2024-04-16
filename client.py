import socket
import os
import argparse

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

def main(args):
    host = 'localhost'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    # Sending the command to the server
    command = args.exp
    client_socket.sendall(command.encode())
    ack = client_socket.recv(1024) # Wait for ack

    filename = 'your_img.jpg'  # This should be the path to your image
    send_file(client_socket, filename)
    
    received_filename = 'received_back_image.mat'
    receive_file(client_socket, received_filename)
    print("Received image back from server.")

    client_socket.close()
    print("Connection closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Flag for the experiment to run.')
    parser.add_argument('exp', type=str, choices=['none', 'pre', 'post', 'all', 's1', 's2', 'u1', 'u2'], help='Specify the amount of work done on client side and server side')
    args = parser.parse_args()
    main(args)

