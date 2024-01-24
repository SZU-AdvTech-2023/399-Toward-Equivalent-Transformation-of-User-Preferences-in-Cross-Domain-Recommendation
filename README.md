# README
**数据下载地址：**

http://jmcauley.ucsd.edu/data/amazon/index_2014.html

下载 ratings only



**Runing commands**  
Run the codes with the following commands on different datasets (amazon means "Movie & Book", amazon2 means "Movie & Music" and amazon3 means "Music & Book").  

-->on Movie & Book dataset: 
CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset amazon --reg 5.0  

-->on Movie & Music dataset:  
CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset amazon2 --reg 0.5  

-->on Music & Book dataset:  
CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset amazon3 --reg 1.0  



