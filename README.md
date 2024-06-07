# ml_tests

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
(win pandas-profiling version err) pip install pandas-profiling

tensorboard activation assuming logs folder is named my_logs in current directory
tensorboard --logdir=./my_logs