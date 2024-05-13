# Use the specified image as the base
FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker

# Setup basic packages
RUN apt-get update && apt install git-lfs -y

# Set the working directory
WORKDIR /

# Expose the specified port
EXPOSE 8850

SHELL ["/bin/bash", "-c"]

ENV PATH /opt/conda/bin:$PATH

# Conda environment
RUN conda create -n habitat python=3.9 cmake=3.14.0
RUN rm -r /habitat*
RUN git clone https://github.com/Reventh-Sharma/habitat-lab.git
RUN /bin/bash --login -c ". activate habitat; conda install habitat-sim withbullet -c conda-forge -c aihabitat -y"
RUN /bin/bash --login -c ". activate habitat; cd habitat-lab; pip install -e habitat-lab; pip install -e habitat-baselines"
RUN /bin/bash --login -c ". activate habitat; pip install loguru; pip install jupyterlab; ipython kernel install --user --name=habitat"
