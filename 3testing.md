root@5587ebe0b507:/workspace/project# python train3.py 2>/dev/null
Using TensorFlow backend
✅ Success! TensorFlow has found and is using the following GPU(s): [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

================================================================================
    STARTING EXPERIMENT FOR MODEL: ResNet50
================================================================================

Preparing tf.data.Dataset pipeline...
Found 19759 files belonging to 10 classes.
Using 15808 files for training.
Found 19759 files belonging to 10 classes.
Using 3951 files for validation.
Calculating class weights to handle data imbalance...
Calculated Class Weights:
{0: 2.0476683937823834, 1: 1.9588599752168525, 2: 1.0805194805194804, 3: 0.3696048632218845, 4: 0.6478688524590164, 5: 1.9184466019417477, 6: 1.1841198501872658, 7: 0.9998734977862113, 8: 1.0062380649267981, 9: 2.1420054200542005}

Building model with ResNet50 base...

--- Training head for ResNet50 ---
Epoch 1/10
247/247 [==============================] - 31s 97ms/step - loss: 0.5423 - accuracy: 0.8434 - val_loss: 0.2285 - val_accuracy: 0.9251
Epoch 2/10
247/247 [==============================] - 25s 99ms/step - loss: 0.3049 - accuracy: 0.9085 - val_loss: 0.2034 - val_accuracy: 0.9299
Epoch 3/10
247/247 [==============================] - 27s 106ms/step - loss: 0.2513 - accuracy: 0.9227 - val_loss: 0.1901 - val_accuracy: 0.9344
Epoch 4/10
247/247 [==============================] - 27s 106ms/step - loss: 0.2281 - accuracy: 0.9291 - val_loss: 0.1688 - val_accuracy: 0.9418
Epoch 5/10
247/247 [==============================] - 27s 107ms/step - loss: 0.1905 - accuracy: 0.9376 - val_loss: 0.1625 - val_accuracy: 0.9489
Epoch 6/10
247/247 [==============================] - 27s 107ms/step - loss: 0.1798 - accuracy: 0.9419 - val_loss: 0.1565 - val_accuracy: 0.9509
Epoch 7/10
247/247 [==============================] - 27s 107ms/step - loss: 0.1617 - accuracy: 0.9487 - val_loss: 0.1583 - val_accuracy: 0.9522
Epoch 8/10
247/247 [==============================] - 27s 108ms/step - loss: 0.1511 - accuracy: 0.9495 - val_loss: 0.1555 - val_accuracy: 0.9517
Epoch 9/10
247/247 [==============================] - 27s 107ms/step - loss: 0.1479 - accuracy: 0.9498 - val_loss: 0.1577 - val_accuracy: 0.9527
Epoch 10/10
247/247 [==============================] - 28s 109ms/step - loss: 0.1375 - accuracy: 0.9545 - val_loss: 0.1761 - val_accuracy: 0.9481

--- Fine-tuning ResNet50 ---
Epoch 10/30
247/247 [==============================] - 35s 113ms/step - loss: 0.1854 - accuracy: 0.9382 - val_loss: 0.1616 - val_accuracy: 0.9537
Epoch 11/30
247/247 [==============================] - 28s 115ms/step - loss: 0.1163 - accuracy: 0.9591 - val_loss: 0.1509 - val_accuracy: 0.9582
Epoch 12/30
247/247 [==============================] - 28s 110ms/step - loss: 0.0890 - accuracy: 0.9683 - val_loss: 0.1479 - val_accuracy: 0.9577
Epoch 13/30
247/247 [==============================] - 28s 108ms/step - loss: 0.0706 - accuracy: 0.9755 - val_loss: 0.1496 - val_accuracy: 0.9595
Epoch 14/30
247/247 [==============================] - 30s 117ms/step - loss: 0.0675 - accuracy: 0.9749 - val_loss: 0.1477 - val_accuracy: 0.9608
Epoch 15/30
247/247 [==============================] - 29s 122ms/step - loss: 0.0577 - accuracy: 0.9799 - val_loss: 0.1468 - val_accuracy: 0.9620
Epoch 16/30
247/247 [==============================] - 30s 119ms/step - loss: 0.0525 - accuracy: 0.9820 - val_loss: 0.1445 - val_accuracy: 0.9628
Epoch 17/30
247/247 [==============================] - 27s 107ms/step - loss: 0.0458 - accuracy: 0.9836 - val_loss: 0.1471 - val_accuracy: 0.9628
Epoch 18/30
247/247 [==============================] - 27s 107ms/step - loss: 0.0407 - accuracy: 0.9853 - val_loss: 0.1438 - val_accuracy: 0.9653
Epoch 19/30
247/247 [==============================] - 30s 118ms/step - loss: 0.0364 - accuracy: 0.9877 - val_loss: 0.1494 - val_accuracy: 0.9633
Epoch 20/30
247/247 [==============================] - 27s 107ms/step - loss: 0.0332 - accuracy: 0.9881 - val_loss: 0.1490 - val_accuracy: 0.9638
Epoch 21/30
247/247 [==============================] - 30s 118ms/step - loss: 0.0276 - accuracy: 0.9902 - val_loss: 0.1537 - val_accuracy: 0.9646
Epoch 22/30
247/247 [==============================] - 27s 107ms/step - loss: 0.0279 - accuracy: 0.9903 - val_loss: 0.1549 - val_accuracy: 0.9653
Epoch 23/30
247/247 [==============================] - 27s 106ms/step - loss: 0.0264 - accuracy: 0.9910 - val_loss: 0.1554 - val_accuracy: 0.9653
Epoch 24/30
247/247 [==============================] - 30s 119ms/step - loss: 0.0212 - accuracy: 0.9925 - val_loss: 0.1554 - val_accuracy: 0.9656
Epoch 25/30
247/247 [==============================] - 28s 110ms/step - loss: 0.0223 - accuracy: 0.9918 - val_loss: 0.1525 - val_accuracy: 0.9656
Epoch 26/30
247/247 [==============================] - 27s 104ms/step - loss: 0.0205 - accuracy: 0.9934 - val_loss: 0.1559 - val_accuracy: 0.9661
Epoch 27/30
247/247 [==============================] - 30s 117ms/step - loss: 0.0188 - accuracy: 0.9927 - val_loss: 0.1553 - val_accuracy: 0.9658
Epoch 28/30
247/247 [==============================] - 27s 105ms/step - loss: 0.0166 - accuracy: 0.9940 - val_loss: 0.1611 - val_accuracy: 0.9661
Epoch 29/30
247/247 [==============================] - 27s 104ms/step - loss: 0.0146 - accuracy: 0.9941 - val_loss: 0.1663 - val_accuracy: 0.9658
Epoch 30/30
247/247 [==============================] - 26s 103ms/step - loss: 0.0147 - accuracy: 0.9956 - val_loss: 0.1677 - val_accuracy: 0.9643
Saved final model to ResNet50_final_model.h5

================================================================================
    STARTING EXPERIMENT FOR MODEL: MobileNetV2
================================================================================

Preparing tf.data.Dataset pipeline...
Found 19759 files belonging to 10 classes.
Using 15808 files for training.
Found 19759 files belonging to 10 classes.
Using 3951 files for validation.
Calculating class weights to handle data imbalance...
Calculated Class Weights:
{0: 2.0476683937823834, 1: 1.9588599752168525, 2: 1.0805194805194804, 3: 0.3696048632218845, 4: 0.6478688524590164, 5: 1.9184466019417477, 6: 1.1841198501872658, 7: 0.9998734977862113, 8: 1.0062380649267981, 9: 2.1420054200542005}

Building model with MobileNetV2 base...

--- Training head for MobileNetV2 ---
Epoch 1/10
247/247 [==============================] - 24s 84ms/step - loss: 0.6156 - accuracy: 0.8229 - val_loss: 0.2708 - val_accuracy: 0.9117
Epoch 2/10
247/247 [==============================] - 21s 83ms/step - loss: 0.3453 - accuracy: 0.8961 - val_loss: 0.2276 - val_accuracy: 0.9261
Epoch 3/10
247/247 [==============================] - 23s 91ms/step - loss: 0.2845 - accuracy: 0.9128 - val_loss: 0.2139 - val_accuracy: 0.9301
Epoch 4/10
247/247 [==============================] - 21s 82ms/step - loss: 0.2592 - accuracy: 0.9200 - val_loss: 0.2109 - val_accuracy: 0.9329
Epoch 5/10
247/247 [==============================] - 21s 82ms/step - loss: 0.2312 - accuracy: 0.9250 - val_loss: 0.1980 - val_accuracy: 0.9393
Epoch 6/10
247/247 [==============================] - 21s 83ms/step - loss: 0.2101 - accuracy: 0.9326 - val_loss: 0.1966 - val_accuracy: 0.9377
Epoch 7/10
247/247 [==============================] - 23s 88ms/step - loss: 0.1836 - accuracy: 0.9393 - val_loss: 0.1963 - val_accuracy: 0.9365
Epoch 8/10
247/247 [==============================] - 21s 84ms/step - loss: 0.1826 - accuracy: 0.9379 - val_loss: 0.2041 - val_accuracy: 0.9342
Epoch 9/10
247/247 [==============================] - 24s 93ms/step - loss: 0.1631 - accuracy: 0.9455 - val_loss: 0.2081 - val_accuracy: 0.9362
Epoch 10/10
247/247 [==============================] - 21s 83ms/step - loss: 0.1566 - accuracy: 0.9477 - val_loss: 0.1990 - val_accuracy: 0.9403

--- Fine-tuning MobileNetV2 ---
Epoch 10/30
247/247 [==============================] - 31s 91ms/step - loss: 0.3327 - accuracy: 0.8930 - val_loss: 0.2159 - val_accuracy: 0.9370
Epoch 11/30
247/247 [==============================] - 23s 92ms/step - loss: 0.2313 - accuracy: 0.9220 - val_loss: 0.2177 - val_accuracy: 0.9350
Epoch 12/30
247/247 [==============================] - 24s 93ms/step - loss: 0.1912 - accuracy: 0.9344 - val_loss: 0.2105 - val_accuracy: 0.9360
Epoch 13/30
247/247 [==============================] - 25s 99ms/step - loss: 0.1771 - accuracy: 0.9388 - val_loss: 0.2084 - val_accuracy: 0.9382
Epoch 14/30
247/247 [==============================] - 24s 92ms/step - loss: 0.1506 - accuracy: 0.9469 - val_loss: 0.1970 - val_accuracy: 0.9410
Epoch 15/30
247/247 [==============================] - 24s 95ms/step - loss: 0.1383 - accuracy: 0.9527 - val_loss: 0.1932 - val_accuracy: 0.9441
Epoch 16/30
247/247 [==============================] - 24s 93ms/step - loss: 0.1225 - accuracy: 0.9591 - val_loss: 0.1900 - val_accuracy: 0.9458
Epoch 17/30
247/247 [==============================] - 24s 94ms/step - loss: 0.1158 - accuracy: 0.9576 - val_loss: 0.1890 - val_accuracy: 0.9463
Epoch 18/30
247/247 [==============================] - 24s 93ms/step - loss: 0.1083 - accuracy: 0.9633 - val_loss: 0.1883 - val_accuracy: 0.9461
Epoch 19/30
247/247 [==============================] - 293s 1s/step - loss: 0.1060 - accuracy: 0.9639 - val_loss: 0.1839 - val_accuracy: 0.9481
Epoch 20/30
247/247 [==============================] - 30s 109ms/step - loss: 0.0955 - accuracy: 0.9660 - val_loss: 0.1856 - val_accuracy: 0.9484
Epoch 21/30
247/247 [==============================] - 25s 99ms/step - loss: 0.0897 - accuracy: 0.9672 - val_loss: 0.1864 - val_accuracy: 0.9453
Epoch 22/30
247/247 [==============================] - 26s 101ms/step - loss: 0.0834 - accuracy: 0.9696 - val_loss: 0.1876 - val_accuracy: 0.9466
Epoch 23/30
247/247 [==============================] - 25s 96ms/step - loss: 0.0826 - accuracy: 0.9717 - val_loss: 0.1874 - val_accuracy: 0.9468
Epoch 24/30
247/247 [==============================] - 24s 94ms/step - loss: 0.0759 - accuracy: 0.9732 - val_loss: 0.1796 - val_accuracy: 0.9491
Epoch 25/30
247/247 [==============================] - 25s 96ms/step - loss: 0.0741 - accuracy: 0.9743 - val_loss: 0.1730 - val_accuracy: 0.9506
Epoch 26/30
247/247 [==============================] - 25s 98ms/step - loss: 0.0691 - accuracy: 0.9728 - val_loss: 0.1750 - val_accuracy: 0.9494
Epoch 27/30
247/247 [==============================] - 23s 92ms/step - loss: 0.0673 - accuracy: 0.9758 - val_loss: 0.1684 - val_accuracy: 0.9514
Epoch 28/30
247/247 [==============================] - 24s 94ms/step - loss: 0.0648 - accuracy: 0.9770 - val_loss: 0.1726 - val_accuracy: 0.9517
Epoch 29/30
247/247 [==============================] - 24s 94ms/step - loss: 0.0616 - accuracy: 0.9789 - val_loss: 0.1710 - val_accuracy: 0.9544
Epoch 30/30
247/247 [==============================] - 26s 103ms/step - loss: 0.0555 - accuracy: 0.9801 - val_loss: 0.1704 - val_accuracy: 0.9512
Saved final model to MobileNetV2_final_model.h5

================================================================================
    STARTING EXPERIMENT FOR MODEL: EfficientNetB0
================================================================================

Preparing tf.data.Dataset pipeline...
Found 19759 files belonging to 10 classes.
Using 15808 files for training.
Found 19759 files belonging to 10 classes.
Using 3951 files for validation.
Calculating class weights to handle data imbalance...
Calculated Class Weights:
{0: 2.0476683937823834, 1: 1.9588599752168525, 2: 1.0805194805194804, 3: 0.3696048632218845, 4: 0.6478688524590164, 5: 1.9184466019417477, 6: 1.1841198501872658, 7: 0.9998734977862113, 8: 1.0062380649267981, 9: 2.1420054200542005}

Building model with EfficientNetB0 base...
Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
16705208/16705208 [==============================] - 1s 0us/step

--- Training head for EfficientNetB0 ---
Epoch 1/10
247/247 [==============================] - 28s 91ms/step - loss: 0.4577 - accuracy: 0.8700 - val_loss: 0.1945 - val_accuracy: 0.9377
Epoch 2/10
247/247 [==============================] - 25s 99ms/step - loss: 0.2392 - accuracy: 0.9260 - val_loss: 0.1663 - val_accuracy: 0.9458
Epoch 3/10
247/247 [==============================] - 22s 87ms/step - loss: 0.1912 - accuracy: 0.9403 - val_loss: 0.1501 - val_accuracy: 0.9512
Epoch 4/10
247/247 [==============================] - 22s 87ms/step - loss: 0.1611 - accuracy: 0.9493 - val_loss: 0.1375 - val_accuracy: 0.9560
Epoch 5/10
247/247 [==============================] - 22s 86ms/step - loss: 0.1380 - accuracy: 0.9539 - val_loss: 0.1311 - val_accuracy: 0.9593
Epoch 6/10
247/247 [==============================] - 22s 86ms/step - loss: 0.1186 - accuracy: 0.9613 - val_loss: 0.1239 - val_accuracy: 0.9625
Epoch 7/10
247/247 [==============================] - 23s 92ms/step - loss: 0.1032 - accuracy: 0.9643 - val_loss: 0.1301 - val_accuracy: 0.9593
Epoch 8/10
247/247 [==============================] - 23s 89ms/step - loss: 0.0972 - accuracy: 0.9677 - val_loss: 0.1210 - val_accuracy: 0.9615
Epoch 9/10
247/247 [==============================] - 22s 87ms/step - loss: 0.0940 - accuracy: 0.9678 - val_loss: 0.1226 - val_accuracy: 0.9638
Epoch 10/10
247/247 [==============================] - 22s 86ms/step - loss: 0.0827 - accuracy: 0.9727 - val_loss: 0.1304 - val_accuracy: 0.9610

--- Fine-tuning EfficientNetB0 ---
Epoch 10/30
247/247 [==============================] - 37s 103ms/step - loss: 0.3447 - accuracy: 0.8739 - val_loss: 0.2088 - val_accuracy: 0.9332
Epoch 11/30
247/247 [==============================] - 26s 102ms/step - loss: 0.2375 - accuracy: 0.9169 - val_loss: 0.1955 - val_accuracy: 0.9375
Epoch 12/30
247/247 [==============================] - 26s 100ms/step - loss: 0.2012 - accuracy: 0.9289 - val_loss: 0.1775 - val_accuracy: 0.9431
Epoch 13/30
247/247 [==============================] - 26s 101ms/step - loss: 0.1729 - accuracy: 0.9405 - val_loss: 0.1656 - val_accuracy: 0.9466
Epoch 14/30
247/247 [==============================] - 26s 103ms/step - loss: 0.1590 - accuracy: 0.9466 - val_loss: 0.1603 - val_accuracy: 0.9476
Epoch 15/30
247/247 [==============================] - 25s 98ms/step - loss: 0.1342 - accuracy: 0.9512 - val_loss: 0.1518 - val_accuracy: 0.9522
Epoch 16/30
247/247 [==============================] - 37s 146ms/step - loss: 0.1275 - accuracy: 0.9554 - val_loss: 0.1472 - val_accuracy: 0.9534
Epoch 17/30
247/247 [==============================] - 26s 103ms/step - loss: 0.1207 - accuracy: 0.9561 - val_loss: 0.1435 - val_accuracy: 0.9552
Epoch 18/30
247/247 [==============================] - 26s 103ms/step - loss: 0.1104 - accuracy: 0.9605 - val_loss: 0.1390 - val_accuracy: 0.9570
Epoch 19/30
247/247 [==============================] - 26s 103ms/step - loss: 0.1053 - accuracy: 0.9633 - val_loss: 0.1370 - val_accuracy: 0.9577
Epoch 20/30
247/247 [==============================] - 26s 103ms/step - loss: 0.0982 - accuracy: 0.9657 - val_loss: 0.1353 - val_accuracy: 0.9580
Epoch 21/30
247/247 [==============================] - 28s 109ms/step - loss: 0.0910 - accuracy: 0.9679 - val_loss: 0.1325 - val_accuracy: 0.9593
Epoch 22/30
247/247 [==============================] - 27s 105ms/step - loss: 0.0842 - accuracy: 0.9689 - val_loss: 0.1296 - val_accuracy: 0.9610
Epoch 23/30
247/247 [==============================] - 26s 103ms/step - loss: 0.0848 - accuracy: 0.9712 - val_loss: 0.1292 - val_accuracy: 0.9608
Epoch 24/30
247/247 [==============================] - 27s 105ms/step - loss: 0.0845 - accuracy: 0.9712 - val_loss: 0.1268 - val_accuracy: 0.9615
Epoch 25/30
247/247 [==============================] - 26s 103ms/step - loss: 0.0771 - accuracy: 0.9732 - val_loss: 0.1245 - val_accuracy: 0.9625
Epoch 26/30
247/247 [==============================] - 27s 107ms/step - loss: 0.0715 - accuracy: 0.9742 - val_loss: 0.1231 - val_accuracy: 0.9630
Epoch 27/30
247/247 [==============================] - 26s 100ms/step - loss: 0.0738 - accuracy: 0.9743 - val_loss: 0.1230 - val_accuracy: 0.9630
Epoch 28/30
247/247 [==============================] - 27s 107ms/step - loss: 0.0745 - accuracy: 0.9734 - val_loss: 0.1220 - val_accuracy: 0.9630
Epoch 29/30
247/247 [==============================] - 26s 105ms/step - loss: 0.0674 - accuracy: 0.9763 - val_loss: 0.1203 - val_accuracy: 0.9633
Epoch 30/30
247/247 [==============================] - 26s 101ms/step - loss: 0.0611 - accuracy: 0.9796 - val_loss: 0.1201 - val_accuracy: 0.9638
Saved final model to EfficientNetB0_final_model.h5

================================================================================
    ✅✅✅ FINAL EXPERIMENT SUMMARY ✅✅✅
================================================================================

Model                | Validation Accuracy    | Validation Loss    | Training Time (min) 
-------------------------------------------------------------------------------------
ResNet50             | 0.9643                 | 0.1677             | 14.63               
MobileNetV2          | 0.9512                 | 0.1704             | 16.95               
EfficientNetB0       | 0.9638                 | 0.1201             | 13.72               

-------------------------------------------------------------------------------------
🏆 Best performing model based on Validation Accuracy: ResNet50
-------------------------------------------------------------------------------------