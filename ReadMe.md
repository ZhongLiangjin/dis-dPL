# Source code for d2PL model

* [Overview](#overview)
* [How to Use](#How-to-Use)

## Overview

The code includes the distributed differentiable parameter learnable (dPL) model (abbreviated as the d2PL model) and CAMELS-based transfer learning dPL model (abbreviated as the tDPL model) presented in paper titled "**Development of a Distributed Physics-informed Deep Learning Hydrological Model for Data-scarce Regions**"  submitted to WRR.

If you have any questions or suggestions with the code or find a bug, please let us know. You are welcome to contact Liangjin Zhong at _zhonglj21@mails.tsinghua.edu.cn_

## How to Use

The code was built on Pytorch. To use this code, please do:

1. Install the dependencies use the following command:

   ```none
   pip install -r requirements.txt
   ```

2. Prepare your dataset 

   Due to confidentiality requirements, the runoff used in the paper are not available to the public. But readers can follow instructions to prepare data for their own study area.

   - Meteorological data (including temperature, precipitation, and pontential evaportranspiration) and static attributes (more details please refer to the paper) are needed to force the models, please organize the data by sub-basins (such as the format shown in 'd2PL/data/data.pkl').
   - Streamflow is required for parameterization. Please sort data in the format shown in the 'd2PL/data/streamflow.csv' .
   - The shapefile of sub-basins should be included in the folder 'd2PL/data/sub-basins'.
   - To replicate the CAMELS-based transfer learning dPL model, you need to download the CAMELS dataset from https://ral.ucar.edu/solutions/products/camels. Unzip the file in the folder 'tDPL/dPLCamels/Camels'. 

3. Run the d2PL model. 

   - Get all data prepared in the folder 'd2PL/data'.
   - Run the file 'd2PL/main.py'.

4. Run the tDPL model.

   - Get alldata prepared in the folders 'tDPL/dPLCamels/data' and 'tDPL/TL/data'

   - Run the file 'tDPL/dPLCamels/traindPL.py' to first get a pre-trained model.

   - Copy the checkpoint file of the  pre-trained to replace the file 'tDPL/TL/data/model_Ep50.pt'.

   - Run the file 'tDPL/TL/main.py' to retrain the model with local data.

5. More details about hyperparameter tuning could be found in the paper and codes.

