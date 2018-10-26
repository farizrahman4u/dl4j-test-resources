
Code: (1.0.0-beta2, nd4j-native, Windows)

```
    public static void main(String[] args) throws Exception {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize norm = new NormalizerStandardize();
        norm.fit(iter);
        iter.setPreProcessor(norm);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .list()
            .layer(new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.fit(iter, 15);

        Evaluation e = net.evaluate(iter);
        ROCMultiClass r = net.evaluateROCMultiClass(iter);
        RegressionEvaluation re = net.evaluateRegression(iter);


        System.out.println(e.stats());
        System.out.println(r.stats());
        System.out.println(re.stats());

        File rootDir = new File("C:\\DL4J\\Git\\dl4j-test-resources\\src\\main\\resources\\regression_testing\\eval_100b");

        File f1 = new File(rootDir, "evaluation.json");
        File f2 = new File(rootDir, "rocMultiClass.json");
        File f3 = new File(rootDir, "regressionEvaluation.json");

        FileUtils.writeStringToFile(f1, e.toJson(), StandardCharsets.UTF_8);
        FileUtils.writeStringToFile(f2, r.toJson(), StandardCharsets.UTF_8);
        FileUtils.writeStringToFile(f3, re.toJson(), StandardCharsets.UTF_8);
    }
```


Output:
```
========================Evaluation Metrics========================
 # of classes:    3
 Accuracy:        0.7800
 Precision:       0.8000
 Recall:          0.7800
 F1 Score:        0.7753
Precision, recall & F1: macro-averaged (equally weighted avg. of 3 classes)


=========================Confusion Matrix=========================
  0  1  2
----------
 45  5  0 | 0 = 0
  0 26 24 | 1 = 1
  0  4 46 | 2 = 2

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
Label               AUC         # Pos     # Neg     
0                   0.9838      50        100       
1                   0.7934      50        100       
2                   0.9544      50        100       Average AUC: 0.9105      
Column    MSE            MAE            RMSE           RSE            PC             R^2            
col_0     6.53809e-02    2.19112e-01    2.55697e-01    2.94214e-01    9.10015e-01    7.05786e-01    
col_1     1.71850e-01    3.46236e-01    4.14547e-01    7.73323e-01    4.83498e-01    2.26677e-01    
col_2     1.03194e-01    2.69759e-01    3.21239e-01    4.64374e-01    7.72518e-01    5.35626e-01   
```