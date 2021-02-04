KR>

해당 폴더에서는 Cifar100 데이터를 32 pixel로 생성시켜 놓았다.

데이터 폴더는 두가지로 만들어 놓았다.

방법 1> 아래 세개의 파일을 하나씩 실행시키면, train과 test 폴더에 cifar10 데이터가 생성될것이다. 

python 0_cifar100_create_folders.py 

python 1_cifar100_data_convert_n_save.py 

python 2_cifar100_modify_filenames.py

Cifar100의 모든 데이터를 JPG로 바꾸는데는 1시간 이상 소요될수도 있다.

네번재 파일은 잘 작동하는지 시험하기 위한 code이다. 

python 3_cifar100_datasets_from_directory.py

방법2> 

Cifar100_input_size_32_pixels.zip.001 ~ Cifar100_input_size_32_pixels.zip.008의 압축을 풀면 train과 test 폴더가 생성된다.

아래의 Github 저장소는 TF1.0과 Keras로 알고리즘 용으로 만들어진 code들 입니다.

TF2.0 재구성 후에 전문가용 코드를 업데이트 예정입니다.

https://github.com/RichardMinsooGo-ML : This is new repository for Machine Learning.

https://github.com/RichardMinsooGo-RL-Gym : This is new repository for Reinforcement Learning based on Open-AI gym.

https://github.com/RichardMinsooGo-RL-Single-agent
: This is new repository for Reinforcement Learning for Single Agent.

https://github.com/RichardMinsooGo-RL-Multi-agent : This new repository is for Reinforcement Learning for Multi Agents.




EN>

In this folder, cifar100 data was created with 32 pixels. You can create cifar100 data with 2 options.

Option 1> 

Execute below 3 python files. Then "train" and "test" folder will be created.

python 0_cifar100_create_folders.py 

python 1_cifar100_data_convert_n_save.py 

python 2_cifar100_modify_filenames.py

It might requires more than 1 hour.

Next code is to test this folder is well working.

python 3_cifar100_datasets_from_directory.py

Option2>

If you unzip "Cifar100_input_size_32_pixels.zip.001 ~ Cifar100_input_size_32_pixels.zip.008" then "train" and "test" folder will be created.


Below github repositories were built with tensorflow 1.X and Keras. 
Those will up updated witf TF2 as expert mode.

https://github.com/RichardMinsooGo-ML : This is new repository for Machine Learning.

https://github.com/RichardMinsooGo-RL-Gym : This is new repository for Reinforcement Learning based on Open-AI gym.

https://github.com/RichardMinsooGo-RL-Single-agent
: This is new repository for Reinforcement Learning for Single Agent.

https://github.com/RichardMinsooGo-RL-Multi-agent : This new repository is for Reinforcement Learning for Multi Agents.

