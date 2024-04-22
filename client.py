import socket
import os
import argparse
from mprnext import mprnext
import torch
import hdf5storage
import cv2
import numpy as np
from memory_profiler import profile

@profile
def receive_array(sock):
    # First receive the shape and dtype
    shape_dtype = sock.recv(1024).decode().split("-")
    shape = tuple(map(int, shape_dtype[0].strip("()").split(",")))
    dtype = np.dtype(shape_dtype[1])
    # Calculate the number of bytes to receive
    data_size = np.prod(shape) * dtype.itemsize
    # Receive the data
    data = bytearray()
    while len(data) < data_size:
        packet = sock.recv(1024)
        if not packet:
            break
        data.extend(packet)
    # Convert the byte data to numpy array
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(array)


def send_array(sock, bgr):
    bgr_array = bgr.numpy()
    data = bgr_array.tobytes()
    # Send the shape and dtype first as plain text, followed by a special separator
    shape_dtype = f"{bgr_array.shape}-{bgr_array.dtype.name}"
    # Calculate the length of the header and send it first
    header = f"{shape_dtype}\n"
    sock.sendall(header.encode())
    # Send the actual data
    sock.sendall(data)

def post_processing(output):
    result = output.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.minimum(result, 1.0)
    result = np.maximum(result, 0)
    return result

def passthrough(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        # Remove the batch dimension
        output = output.squeeze(0)
    return output

def pre_processing(file_path):
    bgr = cv2.imread(file_path)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr = np.float32(bgr)
    bgr = (bgr - bgr.min())/(bgr.max()-bgr.min())
    bgr = np.transpose(bgr, [2, 0, 1])
    bgr = torch.from_numpy(bgr)
    bgr = bgr.unsqueeze(0)

    return bgr

def save_matv73(var):
    hdf5storage.savemat('image.mat', {'cube': var}, format='7.3', store_python_metadata=True)

def define_model():
    if torch.cuda.is_available():
        model = mprnext(in_c=121, out_c=121, n_feat=121, scale_unetfeats=121, scale_orsnetfeats=121, num_cab=4).cuda()
    else:
        model = mprnext(in_c=121, out_c=121, n_feat=121, scale_unetfeats=121, scale_orsnetfeats=121, num_cab=4)
    return model

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

    if command == 'none':
        model = define_model()
        bgr = pre_processing(filename)
        output = passthrough(model, bgr)
        result = post_processing(output)
        save_matv73(result)
        print('All work carried out on client and completed')
    
    elif command == 'pre':
        bgr = pre_processing(filename)
        send_array(client_socket, bgr)
        received_filename = 'received_back_image.mat'
        receive_file(client_socket, received_filename)
        print('Only pre processing done on client side')

    elif command == 'post':
        model = define_model()
        bgr = pre_processing(filename)
        output = passthrough(model, bgr)
        send_array(client_socket, output)
        print('Only post processing done on server')

    elif command == 'all':
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

