# Fast-QNN-Capstone
## Finn files folder

```
echo "export FINN_XILINX_PATH=\"/tools/Xilinx\"" | tee -a ~/.bashrc 

echo "export FINN_XILINX_VERSION=\"2022.2\"" | tee -a ~/.bashrc 

git clone https://github.com/Xilinx/finn/ 

cd finn 

git checkout 52c092568de3ac27102205d03335b17f8a66aee5 
```
Download the files from this folder and put them in their respective directories in your cloned finn directory.
```
./run-docker.sh quicktest 
```
If the above isn’t working run the following. Make sure to put the downloaded files in the respective folders again after the checkout (There should be 0 failed in the quicktest – everything else is fine)
```
git checkout dev 
./run-docker.sh quicktest 
```