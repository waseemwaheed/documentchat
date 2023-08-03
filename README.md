# README

## Project description
A document chat application with Streamlit based web application.

The application allows the user to upload one or more pdf documents and then interrogate them using natural english.

The project built with LangChain and uses OpenAI's API to generate the embeddings for the document(s) which are then stored in a FAISS vector database.

## How To

### Build and run

```shell
docker compose build --no-cache
docker compose up
```

### Build the image

```terminal
docker build -t 01 .
```

### Run the container

```terminal
docker run --name 01 -p 80:80 -d -v %cd%:/code 01
```

### Remove container

```terminal
docker stop 01
docker rm 01
```

### To build without cache

```shell
docker build --no-cache -t 01 --progress=plain >> docker-build.log .
```

### Coding with docker containers

[https://www.youtube.com/watch?v=6OxqiEeCvMI]
