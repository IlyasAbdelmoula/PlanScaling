# PlanScaling
scripts to estimate scaling factor for floor plan images **(context: MID programme)**

## 1- floor plan detection
Using the inference script from [TrainYourOwnYOLO](https://github.com/AntonMu/TrainYourOwnYOLO).

Adding an algorithm to optimze floor plan cropping boundaries after their detection. (image_crop_growth + image_crop_optimize() )

(sample images from CVC-FP dataset)

## 2- layering + scale
creating several floor plan layers based on floor plan image + wall masks.

Estimating a scaling factor, based on detected doors (the door with the highest probability)

(sample images from [DeepFloorPlan](https://github.com/zlzeng/DeepFloorplan))
