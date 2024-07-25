# ml_tests

Repository with some ml algorithms i tested  

venv creation  
python -m venv ./venv  

virtual env activation  
(linux) source venv/bin/activate  
(windows) venv\Scripts\activate.bat  

run jupyterlab  
jupyter lab  

dependencies  
(install) pip install -r requirements.txt  
(save) pip freeze > requirements.txt  

gpu on wsl2  
https://docs.nvidia.com/cuda/wsl-user-guide/index.html  
pip install tensorflow[and-cuda]  

tensorboard activation assuming logs folder is named my_logs in current directory  
tensorboard --logdir=./my_logs  