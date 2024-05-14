1) `git clone https://github.com/Reventh-Sharma/habitat-lab.git`
2) `docker run -it --runtime=nvidia -p 8850:8850 -v <currdir>:/habitat --name habitatenv_updated fairembodied/habitat-challenge:habitat_navigation_2023_base_docker`
3) The following set of commands are to be run inside container:

    `apt-get update && apt install git-lfs -y`\
    `apt install git-lfs -y`\
    `conda init`\
    `source ~/.bashr`\
    `cd /habitat/habitat-lab`\
    `conda env remove =n habitat`\
    `conda create -n habitat python=3.9 cmake=3.14.0`\
    `conda activate habitat`\
    `conda install habitat-sim withbullet -c conda-forge -c aihabitat -y`\
    `pip install -e habitat-lab`\
    `pip install -e habitat-baselines`\
    `pip install loguru`\
    `pip install jupyterlab`\
    `ipython kernel install --user --name=habitat`
4) `python examples/data_creation/main.py --plot_save_dir examples/data_creation/output/plots --video_save_dir examples/data_creation/output/videos`

