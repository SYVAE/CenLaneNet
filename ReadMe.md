# CenLaneNet

This is an implementation of  "":

​                                                                                         <img src="ReadMe.assets/151_ins.jpg" style="zoom:33%;" />    <img src="ReadMe.assets/151_res.jpg" style="zoom:33%;" /> 

![](ReadMe.assets/151.jpg)

### Quick starts:
链接: https://pan.baidu.com/s/1lakhm5LCABcMDVoTTzXN8g code: wn1u
1) Evaluating tusimple

```
python A2_Test_tusimple.py --TusimpleTesting_root "yourTusimplePath/test_set/"
```

2) Evaluating culane
```
python A1_Test_culane.py --CUlane_dataroot "/home/sunyi/sy/data/LaneDetection/CULane/"
```

3) Post-Processing comparison

```
python A3_Postprocess_cost_test.py --TusimpleTesting_root  "yourTusimplePath/test_set/" --show 1
```



### Training your own model:

1) preparing ground_truth--tusimple

```
python D0_CreateTusimpleGt.py --tusimple_root "/home/sunyi/sy/data/LaneDetection/tusimple_raw/tosunyi/train_set/" --GtDataroot  "./"
```

2) Training

```
python A0_Train.py --GtDataroot "/home/sunyi/sy/data/LaneDetection/tusimple/"
```

Some parameters are selected from validation datasets...
(Code for generating the Gt of Culane will come soon....)





Please contact me:sunyi13@nudt.edu.cn if you have any questions about this implementation。If you find this project is useful, please cite this work if possible:
@ARTICLE{9735393,
  author={Yi, Sun and Li, Jian and Xu, Xin and Shi, Yifei},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Adaptive Multi-Lane Detection based on Robust Instance Segmentation for Intelligent Vehicles}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIV.2022.3158750}}
