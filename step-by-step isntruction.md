# Machine Information for CT4 & CT3
**CT3**  
- IP: `10.112.110.51`  
- Username: `jarvis3@user`  
- Password: `G2d@jarvis3@2025`

**CT4**  
- IP: `10.112.110.50`  
- Username: `jarvis3@user`  
- Password: `yydsZ@2025`


## Environment Setup

1. Log in to CT4.
2. Set the proxy: `export https_proxy=http://child-igk.intel.com:912`
3. Navigate to the project directory and install dependencies:
    ```bash
    cd vllm-fork
    git checkout deepseek_r1
    pip install -v .
    ```
4. Install etcd: `sudo apt install etcd -y`
5. Install Mooncake Transfer Engine: `pip3 install mooncake-transfer-engine==0.3.0b3`

Repeat the same steps as above on CT3.

## Prepare the Following Scripts
You can find the scripts in the following path on CT4: `/home/jarvis3@user/lzw/vllm-fork/scrpts_for_ds_r1`
![image](https://github.com/user-attachments/assets/567073e6-1e3b-4635-b9af-64167cc9e768)


## Execution Steps

1. On **CT4**, run: `bash start_etcd_mc.sh`
2. On **CT4**, run: `bash start_prefill.sh`
3. On **CT3**, run: `bash start_decode.sh`
4. Wait for model initialization to complete on both CT4 and CT3.
5. On **CT4**, run: `bash start_proxy.sh`
6. On **CT4**, verify the output: `bash curl.sh`

