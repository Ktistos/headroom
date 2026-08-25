import socket

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect('/tmp/.hr-cat-h0y7j1j2/catalog.sock')
chunks = []
while True:
    chunk = client.recv(65536)
    if not chunk:
        break
    chunks.append(chunk)
client.close()
print(b"".join(chunks).decode("utf-8"), end="")
