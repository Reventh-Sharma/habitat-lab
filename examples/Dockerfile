# Use the specified image as the base
FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

# Setup basic packages
RUN apt-get update && apt install git-lfs -y

# Set the working directory
WORKDIR /habitat

# Expose the specified port
EXPOSE 8850

# Run the specified commands
RUN /bin/bash -c "\
    conda env remove -n habitat && \
    conda create -n habitat python=3.9 cmake=3.14.0 && \
    echo 'source activate habitat' > ~/.bashrc && \
    /bin/bash -c \"source ~/.bashrc\" && \
    conda install habitat-sim withbullet -c conda-forge -c aihabitat && \
    cd habitat-lab && \
    pip install -e . && \
    pip install -e habitat-baselines && \
    pip install loguru && \
    pip install jupyterlab && \
    ipython kernel install --user --name=habitat"