import splitfolders

splitfolders.ratio(
    "C:/Users/sachu/Desktop/mobilenet/data", 
    output="mask_dataset", 
    seed=42, 
    ratio=(.7, .2, .1)
)
