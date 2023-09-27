import requests
headers = {'Content-type': 'application/json'}
response = requests.post(' http://localhost:8080/process', data={"prompt": "The capital of france is "}, headers =headers)

if response.status_code == 201:
  print('Post created successfully!')
else:
  print('Error creating post: {}'.format(response.status_code))