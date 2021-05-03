Hello!

Below you can find a outline of how to reproduce my solution for the **M5-forecasting - accuracy** competition.
if you run into any trouble with the setup/code or have any questions please contact me at **mbnb950513@gmail.com**.<br>
I live in South Korea, timezone UTC + 9 hr. 

# HW & SW & Data setup
## Hardware
I only used Kaggle notebooks. Only CPU, Not GPU.

Below is kaggle notebook specs of July 15, 2020
- CPU : Intel(R) Xeon(R) CPU @ 2.20GHz
- RAM : Max 16GB
- Socket(s) : 1
- Core(s) per socket : 2
- Thread(s) per core : 2

## Software
*Assume your working directory = top level dircetory.*
- python 3.7.6
- python packages are detailed separately in `"./1. document/requirements.txt"`

## Data setup
*Assume your working directory = top level dircetory.*

Raw data is located in `"./2. data/"`. You can also download from kaggle, or use Kaggle API to load data.

# About running code
All codes contain lines that specify your directory on the first line. This line **must represent** the top level of the SUBMISSION folder.<BR> In my case "C:/Users/yeonjun.in/Desktop/SUBMISSION MODEL/"<BR>It's cumbersome, but to eliminate the uncertainty of code execution.<br>Below is the line<br>
<img src='https://github.com/YeonJun-IN/-kaggle-M5---Accuracy-1st-place-solution/blob/master/directory_input.PNG' width=500>  
  
  
## Data Processing
*Assume your working directory = top level dircetory.*
- you can run a code `"./3. code/1. preprocessing/1. preprocessing.ipynb"`.
- The results of this code are saved in `"./2. data/processed/"`
  - `grid_part_1.pkl`
  - `grid_part_2.pkl`
  - `grid_part_3.pkl`
  - `lags_df_28.pkl`
  - `mean_encoding_df.pkl`
- There is nothing special to take care of, except that every time you turn the code, the result is overwritten.

## Model Build
*Assume your working directory = top level dircetory.*

There are 2 options to make submissions.
<a id="21"></a> <br>
### 1. Not to train, only to predict.
  - Model files are already included in `"./5. models/"`.
  - `lgb_model_000_v1.bin` means recursive model. `non_recur_model_000.bin` means non recursive model
  - Totally, there are 220 model files.
  - Therefore, all you have to do is run all code in `"./3. code/3. predict/"`
    - you must run `"./3. code/3. predict/3. Final ensemble.ipynb"` last. Except for it, the order within `"./3. code/3. predict/"` does not matter. 
  - Codes 1-1, 1-2, and 1-3 in `"./3. code/3. predict/"` produce results in `"./6. submission/before_ensemble/"`.
    - `submission_kaggle_recursive_store.csv`
    - `submission_kaggle_recursive_store_cat.csv`
    - `submission_kaggle_recursive_store_dept.csv`
  - Codes 2-1, 2-2, and 2-3 in `"./3. code/3. predict/"` produce results in `"./4. logs/"` and `"./6. submission/before_ensemble"`.
    - The files in `"./4. logs/"` are created first, and these files are processed to make the files in `"./6. submission/before_ensemble"`. 
    - In other words, the files in `"./4. logs/"` are intermediate processes, so you don't have to take care of it. 
    - The final result of codes is the files in `"./6. submission/before_ensemble"`.
      - `submission_kaggle_nonrecursive_store.csv`
      - `submission_kaggle_nonrecursive_store_cat.csv`
      - `submission_kaggle_nonrecursive_store_dept.csv`
      
  - Code 3-1 in `"./3. code/3. predict/"` use files in `"./6. submission/before_ensemble"` to make final submission in `"./6. submission/`
  
### 2. Retrain a model to predict.
  - If you want to retrain the models, you can run the code in `"./3. code/2. train/"` and then run the code in `"./3. code/3. predict/"`.
    - The order within `"./3. code/2. train/"` does not matter.
    - If you run the codes in `"./3. code/2. train/"`., model files that already existed will be overwritten.
  - Codes 1-1, 1-2, and 1-3 in `"./3. code/2. train/"` produce results in `"./5. models/"` and `"./2. data/processed/"`.
    - Files in `"./2. data/processed/"`
      - `test_CA_1.pkl`, `test_CA_1_FOODS.pkl`, `test_CA_1_FOODS_1.pkl` etc.
      - This files are already included in `"./2. data/processed/"`.
      - If you run, `test_000.pkl` files that already existed will be overwritten.
  - How to predict is same as [1. Not to train, only to predict.](#21)
  
