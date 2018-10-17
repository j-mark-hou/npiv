Implementation of the core idea behind the DeepIV paper, but with tree boosting and quantile regression instead of deep learning and mixtures of gaussians.

Just browsing:

- for more in depth discussion, see this blog post [http://jmarkhou.com/npiv/](http://jmarkhou.com/npiv/)
- look at [```example/example.ipynb```](https://github.com/j-mark-hou/npiv/blob/master/example/example.ipynb)

Run via docker:
```docker-compose up```


How to build:
```docker-compose build```

Development:

- dockerfile just installs the package via pip with the -e option, and docker-compose then mounts this directory to that install location, so that any changes made are directly reflected.  Thus, just run the container, and make whatever changes as you like.
- to run tests as you develop, do ```docker exec 270033469fd0 pytest -s```, excep with whatever the hash of the container is (do ```docker container ls to see running containers```)
- or even better, just use bash in the container: ``` docker exec 270033469fd0 bash ``` 