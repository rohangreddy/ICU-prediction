# Prediction of Acute Kidney Injury and Sepsis using ICU data

This repository explores the prediction of both Acute Kidney Injury (AKI) and Sepsis using time-series ICU patient data.

AKI and Sepsis are two conditions which have a negative effect on patient morbidity and mortality in the intensive care unit. Early detection of these two conditions may allow for life-saving early interventions that improve patient outcomes and reduce complication-related costs that stem from these conditions.  

# Dataset

[PhysioNet/Computing in Cardiology 2019 clinical dataset](https://physionet.org/content/challenge-2019/1.0.0/)

The training and testing datasets used for AKI and Sepsis prediction are drawn from the dataset above. The datasets consist of patient ICU records each stored as an individual CSV file. Each individual CSV file contains the hour by hour data of various clinical features, along with several demographic features. 


The training dataset that will be used for AKI prediction consists of 32,336 patient records with a total of 1,244,421 rows and 36 features. Summary statistics for each of the features can be seen below:


| features              | min   | max    | mean                | median | std                 | missing |
|-----------------------|-------|--------|---------------------|--------|---------------------|---------|
| HR                    | 20.0  | 280.0  | 84.61473047022297   | 84.0   | 17.276917271088003  | 122634  |
| O2Sat                 | 20.0  | 100.0  | 97.19517371681457   | 98.0   | 2.9192667396187266  | 162056  |
| Temp                  | 20.9  | 50.0   | 36.97995902271426   | 37.0   | 0.7699307449455322  | 822968  |
| SBP                   | 20.0  | 300.0  | 123.74999620610186  | 121.0  | 23.213788967470734  | 182189  |
| MAP                   | 20.0  | 300.0  | 82.38307317431263   | 80.0   | 16.3518384199459    | 154626  |
| DBP                   | 20.0  | 300.0  | 63.83839808721236   | 62.0   | 14.000919680702681  | 391216  |
| Resp                  | 1.0   | 100.0  | 18.735898023684662  | 18.0   | 5.0988806005183145  | 190490  |
| EtCO2                 | 10.0  | 100.0  | 32.97396400714628   | 33.0   | 7.931639600200454   | 1198523 |
| BaseExcess            | -32.0 | 100.0  | -0.6692614154244216 | 0.0    | 4.284286544624071   | 1176399 |
| HCO3                  | 0.0   | 55.0   | 24.074651038790396  | 24.0   | 4.384801763216848   | 1191908 |
| FiO2                  | -50.0 | 4000.0 | 0.561040546487821   | 0.5    | 12.37430735073774   | 1139899 |
| pH                    | 6.62  | 7.93   | 7.379023527514059   | 7.38   | 0.07453494394952005 | 1157629 |
| PaCO2                 | 10.0  | 100.0  | 41.023684324013516  | 40.0   | 9.269229936115897   | 1174856 |
| SaO2                  | 23.0  | 100.0  | 92.67975698635229   | 97.0   | 10.864319830793923  | 1201337 |
| AST                   | 3.0   | 9961.0 | 263.1675816023739   | 41.0   | 865.2394373300145   | 1224201 |
| BUN                   | 1.0   | 268.0  | 23.85605378551103   | 17.0   | 19.83643140091416   | 1158673 |
| Alkalinephos          | 7.0   | 3833.0 | 101.80419685236073  | 74.0   | 121.01546848169838  | 1224406 |
| Calcium               | 1.0   | 27.0   | 7.558656211645981   | 8.3    | 2.433340432636586   | 1171091 |
| Chloride              | 26.0  | 145.0  | 105.81701686941562  | 106.0  | 5.862784411299719   | 1187454 |
| Creatinine            | 0.1   | 46.6   | 1.50897979319599    | 0.94   | 1.8113814524853356  | 1168407 |
| Bilirubin_direct      | 0.01  | 37.5   | 1.8218285956724647  | 0.5    | 3.6283137900026694  | 1242064 |
| Glucose               | 13.0  | 952.0  | 136.77015645090145  | 126.0  | 50.940927999352226  | 1031319 |
| Lactate               | 0.2   | 31.0   | 2.6764491282541196  | 1.82   | 2.56532741026917    | 1210925 |
| Magnesium             | 0.2   | 9.8    | 2.0519764150344635  | 2.0    | 0.398652189771607   | 1165642 |
| Phosphate             | 0.2   | 18.8   | 3.5461314509053214  | 3.3    | 1.4204563924046056  | 1194273 |
| Potassium             | 1.0   | 27.5   | 4.13668191275341    | 4.1    | 0.6435192571781231  | 1128107 |
| Bilirubin_total       | 0.1   | 49.6   | 2.143699450016176   | 0.9    | 4.386306520270476   | 1225875 |
| TroponinI             | 0.01  | 440.0  | 8.226923934482459   | 0.3    | 24.706607782677512  | 1232760 |
| Hct                   | 5.5   | 71.7   | 30.797260458473442  | 30.3   | 5.482085375909788   | 1133793 |
| Hgb                   | 2.2   | 32.0   | 10.430294887245804  | 10.3   | 1.9650860799104346  | 1152318 |
| PTT                   | 12.5  | 250.0  | 41.15813956658034   | 32.4   | 26.10209331815181   | 1207736 |
| WBC                   | 0.1   | 440.0  | 11.429603700462554  | 10.3   | 7.307838403865409   | 1164431 |
| Fibrinogen            | 35.0  | 1383.0 | 284.5556071341315   | 248.0  | 152.0304761198496   | 1236235 |
| Platelets             | 1.0   | 2322.0 | 195.84763715188677  | 180.0  | 103.9246785181922   | 1170379 |
| Age                   | 14.0  | 100.0  | 61.95546994144267   | 64.0   | 16.44907123435386   | 0       |
| Hours_Since_Admission | 1.0   | 336.0  | 27.111325668724653  | 21.0   | 29.38038393520359   | 0       |

Similarly, the training dataset that will be used for sepsis predictions consists of 31,834 patient records with a total of 1,231,811 rows and 41 features.


