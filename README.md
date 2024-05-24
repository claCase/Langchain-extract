# Project Structure 
To get the tree structure cd into the parent directory and paste the following linux command
```cmd
sudo apt-get install tree
sudo tree --dirsfirst -r ./
```
The project is structured in two folders, one for the content extractor and one for the dashboard. 
```cmd
├── extractor
│   ├── requirements.txt
│   ├── main.py
│   └── dockerfile
├── dashboard
│   ├── requirements.txt
│   ├── main.py
│   └── dockerfile
├── requirements.txt
├── main.py
├── docker-compose.yml
├── docker-compose.debug.yml
└── Dockerfile
```

# Linux Set Up 
We need to first get updates from linux packages. Then we need to install docker to run images inside containers.
```cmd
sudo apt-get update -y
sudo apt install python3.11 docker.io docker-compose-v2 -y
```

# Docker Set up 
We need to build the images for each service and install the required packages. 
This can be done by executing the following command. <br>
```cmd
sudo docker compose up --build
```
You can re-initialize the built services by executing the following command:
```cmd
sudo docker compose up
```
To clear built containers:
```cmd
docker rm -vf $(docker ps -aq)
```
To clear built images
```cmd
docker rmi -f $(docker images -aq)
```
The docker-compose.yml will make a share volume and create the folders app/data to save files so that the two services will be allowed to communicate<br>

**To build the app with Cohere model set the COHERE flag to True in line 24 of the docker-compose.yml file, otherwise the app will use the Anthropic model: Deafults to False** <br><br>
Unfortunately I didn't manage to make cohere model work as intended since the parser would always throw error as the model output was not complete and had missing informations/fields.
The following are free parameters to change in the docker-compose.yml file:

- API_KEY_COHERE
- API_KEY_ANTHROPIC
- FROM_PAGE
- TO_PAGE
- COHERE
- TEMPERATURE