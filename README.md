# Noise-tolerant Universal Representation Learning for Multivariate Time Series from global-to-local Perspective (KBS 2025)
This repository provides a PyTorch implementation of TSG2L ([paper](https://pdf.sciencedirectassets.com/271505/1-s2.0-S0950705125X00022/1-s2.0-S0950705125001844/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCMV6xn4MZ0JE031C5amNBDLetvkpydkn1AuPrkzLL5LgIgTTazPtk5iHU2pqPmSVycjSzArpda9pu8yejdv0u5y%2FIqswUIJRAFGgwwNTkwMDM1NDY4NjUiDD%2BKrvw%2BtPVRDjBvDCqQBURSeTG6Iw6KRRxWMxFMuKCOsaoN17bMAVROZKHYhYTDxc%2Bp3%2BSRirTQsSy3irscl5lJ0KqzqAb8NuppNjxGd0uDg2VKW8jLYmrOksYooswohdlHhYOzZVyeVGy0ItMdqmJE4TFNlB3ScxvyTl%2FgtaXdUM%2FxiMn2r8tQqfJiL5qX0xUcAC1zq8a5ZSALiJCA2vEb%2F0EMarl0kyi7ruXLy06GoBIDqv34hEAYoZWjV%2BIs9GePmyNpY1wbH1GMYRaxr6%2Blh2QeleMauhOYFuvi73Xyd9iFQH31JPNkAXGOUvtTmLETCq577IiZ83nuYPgQ78aeG8qfIc8Agg3wtyYLW%2FSPFXrqWXH9VQDHZs4rJ0F9YIjjEfaefZeD2hz657Adn6118W6keiVnHaTdTRJeo1bJVIxRPfFHJ%2F49hH2R%2B7hjkZ1fBAeKYEXbaejA8YLXzL0suupCbZ6Ih4Kuu%2BHQbBm63lbAzZVJS0o4iwGjZ5umYtqQjA7xeL%2Bls0I3XFK3E%2BCokOiFS1ACSd4G4wWZlHqloog5%2BK3rrP%2BeUnAtQDbPvLcEAsVAOil%2Fbu5WumU%2F230fG21XBSLdWR61qkFt2WdNc1jDe3JgnuG9bVbPuGg4f9WvPkw08yDnNL29uSvypzi%2FMEYBYhWoes22KlATnVbqrbEFxYLQOSR7AlZiqKIgKT6F2BfG%2F%2FY%2BEf4%2F8aQ5iT6M8qCIA5AOHcIXR4ORw0yUP7G%2BQdcNw37T57N5Ne%2FH4Hir5LDwi5u0lb95lPdUvYh0Ys%2BNjn3B0iAOAkWsLVkti5rolLLUj8GogMvMRrbOdern2i1K7dMStnvQkVpozHVOrfJwvZ1Fb%2BaPXZV9j%2F%2BZpUy%2BPE2Qy8QvHS%2BJFRKGMJ%2BvpL4GOrEBTI0HuiMghTXikt5CTdf6wyI8q0v%2FBMLhonYFJoqf4EjwxdUItBewar8TUJm7mNfcFjwagaz4Jodd93KHDgjXk9vYXVZ0uL1fDKK%2FNdip4AEdugmdtEyxYwQzDxw3qaozWxOc9ETbYQUt%2ByXdPmmTY7WdgpvXzBha3MjAHXzvCzzc3%2Be9slOrVMtSTATFKozfNLzHDU6vC0osWw8Ztnn3aereTEhGmE89m75jZ%2Fxi6QEm&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250306T041355Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5CAPX4ZG%2F20250306%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8d83f16bfa37adf5325130d5215cc8d6907afa2ce717bb9cd22778f8d576cb44&hash=6b0545151c4240d9863c9e9cecd244bb2ae1a80b6fcb0250c5eb21d08fb5a221&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0950705125001844&tid=spdf-987f48c5-039f-46f3-8f1f-73fd32914116&sid=1760c6c82bd4f343ba791930b9d31fce0b0cgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=050d5b51070454560751&rr=91bf05f48986d041&cc=cn)) .

## Requirements
The recommended requirements for TSG2L are specified as follows:

- Python 3.7
- torch==1.12.0
- numpy==1.21.6
- pandas==1.0.1
- scikit-learn==0.24.2
- scipy==1.7.3

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
## Data 
The datasets can be obtained and put into datasets/ folder in the following way:
### forecast
- [4 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/forecast/ETTh1.csv`, `datasets/forecast/ETTh2.csv`, `datasets/forecast/ETTm1.csv` and `datasets/forecast/ETTm1.csv`.
- [weather](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)should be placed at `datasets/forecast/weather.csv`.
- [airquality](https://archive.ics.uci.edu/dataset/360/air+quality)should be placed at `datasets/forecast/airquality.csv`.
### classification
- [128UCRdatasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.
Such as Chinatown, ItalyPowerDemand, ArrowHead.
- [30UEAdatasets](http://www.timeseriesclassification.com/dataset.php)) should be put into `datasets/UEA/` so that each data file can be located by `datasets/UEA/<dataset_name>/<dataset_name>_*.csv`.
Such as RacketSports, SharePriceIncrease, BasciMotions.
### anomaly
- [MSL](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/anomaly/MSL.csv`.
- [SMD](https://github.com/NetManAIOps/OmniAnomaly) should be placed at `datasets/anomaly/SMD.csv`.
- [SMAP](https://en.wikipedia.org/wiki/Soil_Moisture_Active_Passive) should be placed at `datasets/anomaly/SMAP.csv`.
- [MBA](https://paperswithcode.com/dataset/mit-bih-arrhythmia-database) should be placed at `datasets/anomaly/MBA.csv`.
- [SwaT](https://drive.google.com/drive/folders/1ABZKdclka3e2NXBSxS9z2YF59p7g2Y5I) should be placed at `datasets/anomaly/SwaT.csv`.
## Usage
To train and evaluate TSG2L on a dataset, run the following command:
```bash
python train_forecast.py --dataset <dataset_name>  --run_name <run_name> --loader <loader> --gpu <gpu> 
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| gpu | The gpu no. used for training and inference (defaults to 0) |

(For descriptions of more arguments, run `python train.py -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 
**Scripts:** The scripts for reproduction are provided in scripts/ folder.
