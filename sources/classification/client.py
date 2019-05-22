import socket
import cv2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 8073)
sock.connect(server_address)

image_left = cv2.imread('signs_ready/stop/09370df0-4815-4e2f-bf3c-cf6d563f43d4.jpg')
image_right = cv2.imread('signs_ready/right/cfa98d36-6331-4a02-8ee0-cd0aa96c2dc2.jpg')
image_center = cv2.imread('signs_ready/woman/892342ad-4559-4cf8-817c-6c2a424d6c0c.jpg')

resized_left = cv2.resize(image_left, (55, 100))
resized_right = cv2.resize(image_right, (55, 100))
resized_center = cv2.resize(image_center, (55, 100))
bytes_left = resized_left.tobytes()
bytes_right = resized_right.tobytes()
bytes_center = resized_center.tobytes()

data = bytes_left + bytes_center + bytes_right

sock.sendto(data, server_address)

print(len(data))
