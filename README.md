How to build & run:
```docker-compose up```

How to build:
```docker-compose build```

Development:

- The package is installed via pip with the -e option, and then this directory is mounted to that install location, so that any changes made are directly reflected.  Thus, just run the container, and make whatever changes as you like.  No need to re-build every time.
- To run tests as you develop, do ```docker exec 270033469fd0 pytest -s```, and be sure to replace ```270033469fd0``` with whatever the id of your running container of NPIV is (do ```docker container ls to see running containers```)