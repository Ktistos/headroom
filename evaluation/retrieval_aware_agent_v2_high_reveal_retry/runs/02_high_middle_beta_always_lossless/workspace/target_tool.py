REQUEST = b"target"
import socket

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect('/tmp/.hr-cat-_s9eu7y2/catalog.sock')
client.sendall(REQUEST + b"\n")
chunks = []
while True:
    chunk = client.recv(65536)
    if not chunk:
        break
    chunks.append(chunk)
client.close()
print(b"".join(chunks).decode("utf-8"), end="")
