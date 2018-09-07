FROM jmarkhou/jupyter_xgb

# copy this package and pip install it in editable mode.
# this allows us to then map this local directory into the
# corresponding directory in the image, and directly do development
# my modifying the local files and then having them automatically
# reflected in the running container.
# this is relatively easier than having to re-build every time 
# things change.
USER root
COPY . /home/jovyan/npiv
WORKDIR /home/jovyan/npiv
RUN pip install -e .

RUN fix-permissions /home/jovyan/npiv

# reset permissions
USER $NB_UID