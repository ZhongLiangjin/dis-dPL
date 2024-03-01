# Source code for dis-dPL model

* [Overview](#overview)
* [How to Use](#How-to-Use)

## Overview

The code includes the distributed differentiable parameter learnable (dPL) model (abbreviated as the dis-dPL model) and CAMELS-based transfer learning dPL model (including lumped and distributed variants) presented in paper titled "**Development of a Distributed Physics-informed Deep Learning Hydrological Model for Data-scarce Regions**"  submitted to WRR.

If you have any questions or suggestions with the code or find a bug, please let us know. You are welcome to contact Liangjin Zhong at _zhonglj21@mails.tsinghua.edu.cn_

## How to Use

The code was built on Pytorch. To use this code, please do:

1. Install the dependencies use the following command:

   ```none
   pip install -r requirements.txt
   ```

2. Prepare your dataset 

   Due to confidentiality requirements, the runoff used in the paper are not available to the public. But readers can follow instructions to prepare data for their own study area.

   - Meteorological data (including temperature, precipitation, and pontential evaportranspiration) and static attributes (more details please refer to the paper) are needed to force the models, please organize the data by sub-basins (such as the format shown in 'dis-dPL/data/data.pkl').
   - Streamflow is required for parameterization. Please sort data in the format shown in the 'dis-dPL/data/streamflow.csv' .
   - The shapefile of sub-basins should be included in the folder 'dis-dPL/data/sub-basins'.
   - To replicate the CAMELS-based transfer learning dPL model, you need to download the CAMELS dataset from https://ral.ucar.edu/solutions/products/camels. Unzip the file in the folder 'tDPL/dPLCamels/Camels'. 

3. Run the dis-dPL model. 

   - Get all data prepared in the folder 'dis-dPL/data'.
   - Run the file 'dis-dPL/main.py'.

4. Run the lumped TL model.

   - Get alldata prepared in the folders 'TL/dPLCamels/data' and 'TL/TL-lumped/data'

   - Run the file 'TL/dPLCamels/traindPL.py' to first get a pre-trained model.

   - Copy the checkpoint file of the  pre-trained model to replace the file 'TL/TL-lumped/data/model_Ep50.pt'.

   - Run the file 'TL/TL-lumped/main.py' to retrain the model with local data.

5. Run the distributed TL model.

   - Copy the checkpoint file of the  pre-trained model to replace the file 'TL/TL-distributed/data/model_Ep50.pt'.

   - Run the file 'TL/TL-distributed/main.py' to retrain the model with local data.

6. More details about hyperparameter tuning could be found in the paper and codes.

