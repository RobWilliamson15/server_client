import socket
from mprnext import mprnext
import torch
import hdf5storage
import cv2
import numpy as np

def receive_array(conn):
    header = ""
    while True:
        chunk = conn.recv(1).decode()
        if chunk == '\n':  # We use newline as a delimiter for the end of the header
            break
        header += chunk
    shape, dtype = header.split('-')
    shape = tuple(map(int, shape.strip('()').split(',')))
    dtype = np.dtype(dtype)

    # Calculate the number of bytes to receive
    data_size = np.prod(shape) * dtype.itemsize

    # Receive the data
    data = bytearray()
    while len(data) < data_size:
        packet = conn.recv(1024)
        if not packet:
            break
        data.extend(packet)

    # Convert the byte data to numpy array
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(array)


def send_array(conn, bgr):
    bgr_array = bgr.numpy()
    data = bgr_array.tobytes()
    # Send the shape and dtype first as plain text
    shape_dtype = f"{bgr_array.shape}-{bgr_array.dtype}"
    conn.sendall(shape_dtype.encode() + b"\n")
    # Send the actual data
    conn.sendall(data)

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
    try:
        with open(filename, 'rb') as file:
            data = file.read()
            conn.sendall(data)  # Send the file data back
    except BrokenPipeError as e:
        print(f"Error sending data: {e}")

def main():
    host = 'localhost'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Server listening on port", port)
    
    while True:
        conn, addr = server_socket.accept()
        print('Connected by', addr)
    
        # Recieve the command from the client
        command = conn.recv(1024).decode()
        conn.sendall(b'ACK')

        if command == 'none':
            print('none')
        elif command == 'all':
            print('all')
            filename = 'server_image.png'
            receive_file(conn, filename)
            model = define_model()
            bgr = pre_processing(filename)
            output = passthrough(model, bgr)
            result = post_processing(output)
            save_matv73(result)
            filename = 'image.mat'
            send_file(conn, filename)
        elif command == 'pre':
            print('pre')
            model = define_model()
            bgr = receive_array(conn)
            output = passthrough(model, bgr)
            result = post_processing(output)
            save_matv73(result)
            filename = 'image.mat'
            send_file(conn, filename)
        elif command == 'post':
            print('post')
            output = receive_array(conn)
            result = post_processing(output)
            save_matv73(result)
            filename = 'image.mat'
            send_file(conn, filename)

        conn.close()
        #server_socket.close()

if __name__ == '__main__':
    main()

